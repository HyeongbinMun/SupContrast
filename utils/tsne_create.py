# utils/classifier_activation_map.py
import os, argparse, glob, warnings
from typing import List
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import cm
from tqdm import tqdm

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Truncated 파일도 최대한 열기
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

from networks.resnet_big import SupConResNet, LinearClassifier

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

def parse_args():
    p = argparse.ArgumentParser("Visualize classifier activation maps (Grad-CAM) on a directory")
    p.add_argument("--ckpt", type=str, required=True, help="linear-trained checkpoint (.pth) with 'model' & 'classifier'")
    p.add_argument("--data_dir", type=str, required=True, help="directory containing images (flat or nested)")
    p.add_argument("--output_dir", type=str, default="./activation_maps", help="where to save heatmaps")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1, help="Grad-CAM은 보통 1 권장")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target_layer", type=str, default="layer4", help="backbone encoder target layer (e.g., layer4)")
    p.add_argument("--normalize_mean", nargs="+", type=float, default=[0.5,0.5,0.5])
    p.add_argument("--normalize_std",  nargs="+", type=float, default=[0.5,0.5,0.5])
    return p.parse_args()

def strip_module(sd: dict) -> dict:
    new = {}
    for k, v in sd.items():
        new[k.replace("module.", "")] = v
    return new

def list_images(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No image files found under: {root}")
    return files

class ImageDirDataset(Dataset):
    def __init__(self, paths: List[str], tfm):
        self.paths = paths
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            # 손상 파일은 None으로 표시 → 상위 루프에서 스킵
            return None, p, str(e)
        return self.tfm(img), p, None

def build_models(args):
    # checkpoint에서 클래스 수 유추 (classifier의 weight shape)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ("model" not in ckpt) or ("classifier" not in ckpt):
        raise KeyError("Checkpoint must contain keys: 'model' and 'classifier'")
    model_sd = strip_module(ckpt["model"])
    clf_sd   = strip_module(ckpt["classifier"])
    # weight key 찾기
    w_key = None
    for k in clf_sd.keys():
        if k.endswith(".weight") or "weight" in k:
            w_key = k; break
    if w_key is None:
        raise KeyError("No weight tensor found in classifier state_dict")
    num_classes = clf_sd[w_key].shape[0]

    backbone = SupConResNet(name=args.model)
    classifier = LinearClassifier(name=args.model, num_classes=num_classes)

    device = torch.device(args.device)
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    missing, unexpected = backbone.load_state_dict(model_sd, strict=False)
    if missing or unexpected:
        print(f"ℹ️ backbone loaded with missing={len(missing)} unexpected={len(unexpected)}")
    classifier.load_state_dict(clf_sd, strict=True)

    backbone.eval(); classifier.eval()
    return backbone, classifier, num_classes

class GradCAM:
    def __init__(self, backbone: SupConResNet, classifier: nn.Module, target_layer_name: str):
        self.encoder = backbone.encoder  # conv trunk
        self.classifier = classifier
        self.device = next(self.encoder.parameters()).device

        modules = dict(self.encoder.named_modules())
        if target_layer_name not in modules:
            raise KeyError(f"Target layer '{target_layer_name}' not found in encoder. Available: {list(modules.keys())[:20]}...")
        self.target_layer = modules[target_layer_name]

        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, m, i, o): self.activations = o.detach()
    def _save_gradient(self, m, gi, go): self.gradients = go[0].detach()

    @torch.no_grad()
    def forward_scores(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        scores = self.classifier(feats)
        return scores

    def generate(self, x: torch.Tensor, class_idx: int = None):
        # forward (grad 필요)
        feats = self.encoder(x)
        scores = self.classifier(feats)

        if class_idx is None:
            class_idx = int(scores.argmax(dim=1).item())

        self.encoder.zero_grad()
        self.classifier.zero_grad()
        one_hot = torch.zeros_like(scores)
        one_hot[0, class_idx] = 1.0
        scores.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM 계산
        grads = self.gradients.mean(dim=[2,3], keepdim=True)  # [B,C,1,1]
        cam = torch.sum(grads * self.activations, dim=1)      # [B,H,W]
        cam = torch.relu(cam)[0]                              # [H,W]
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))       # to (W,H)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def overlay_heatmap(img_tensor: torch.Tensor, cam: np.ndarray):
    # img_tensor: [3,H,W] (normalized)
    img = img_tensor.detach().cpu().permute(1,2,0).numpy()
    # 역정규화 (대략적인 시각화 목적)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)

    heatmap = (cm.jet(cam)[:,:,:3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return overlay  # RGB

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 파일 나열 (flat / nested 둘 다 지원)
    paths = list_images(args.data_dir)

    # 전처리
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.normalize_mean, std=args.normalize_std)
    ])
    ds = ImageDirDataset(paths, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # 모델 로드
    backbone, classifier, num_classes = build_models(args)
    gradcam = GradCAM(backbone, classifier, target_layer_name=args.target_layer)

    # 진행 상황 tqdm
    skipped, saved = 0, 0
    pbar = tqdm(total=len(ds), desc="Activations", unit="img")
    with torch.no_grad():
        for batch in dl:
            imgs, batch_paths, errs = batch
            # 손상 이미지(None) 처리
            if isinstance(imgs, list) or (imgs is None):
                for p, e in zip(batch_paths, errs):
                    skipped += 1
                pbar.update(len(batch_paths))
                continue

            imgs = imgs.to(args.device, non_blocking=True)
            # Grad-CAM은 배치 1 권장. batch>1이면 루프 처리
            B = imgs.size(0)
            for b in range(B):
                x = imgs[b:b+1]  # [1,3,H,W]
                try:
                    cam, pred = gradcam.generate(x, class_idx=None)
                    overlay = overlay_heatmap(x[0], cam)
                    # 저장 파일명
                    base = os.path.splitext(os.path.basename(batch_paths[b]))[0]
                    out_path = os.path.join(args.output_dir, f"{base}_pred{pred}.png")
                    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    saved += 1
                except Exception as e:
                    skipped += 1
                pbar.update(1)

            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    print(f"✅ Done. saved={saved}, skipped={skipped}, out_dir={args.output_dir}")

if __name__ == "__main__":
    main()
