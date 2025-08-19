# inference_linear.py
from __future__ import annotations
import os, sys, argparse, glob
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 학습 코드와 동일 패키지 구조 가정
from networks.resnet_big import SupConResNet, LinearClassifier

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    p = argparse.ArgumentParser("SupCon linear-classifier inference")
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to trained checkpoint (.pth) having keys: model/classifier/optimizer/epoch/best_acc")
    p.add_argument("--input", type=str, required=True,
                   help="image file OR a directory (recursive)")
    p.add_argument("--model", type=str, default="resnet50",
                   help="backbone used in training: resnet18/50/101/200 ...")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mean", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    p.add_argument("--std",  nargs="+", type=float, default=[0.5, 0.5, 0.5])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--classes", type=str, default="",
                   help="optional text file: one class name per line (index order)")
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def list_images(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(path, "**", f"*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(path, "**", f"*{ext.upper()}"), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No images found under: {path}")
    return files

class ImageDataset(Dataset):
    def __init__(self, paths: List[str], tfm):
        self.paths = paths
        self.tfm = tfm
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        return self.tfm(img), p

def strip_module(sd: dict) -> dict:
    """remove 'module.' prefix at any depth (top-level or after dots)"""
    new = {}
    for k, v in sd.items():
        k2 = k.replace("module.", "")
        k2 = k2.replace(".module.", ".")  # handle 'encoder.module.layer...'
        new[k2] = v
    return new

def detect_weight_key(sd: dict) -> str:
    """find the first key that ends with '.weight'"""
    cand = [k for k in sd.keys() if k.endswith(".weight")]
    if not cand:
        # fallback: any key containing 'weight'
        cand = [k for k in sd.keys() if "weight" in k]
    if not cand:
        raise KeyError("No weight tensor found in classifier state_dict")
    return cand[0]

def load_class_names(path: str, num_classes: int|None=None):
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"classes file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip() != ""]
    if (num_classes is not None) and (len(names) != num_classes):
        print(f"⚠️ classes size ({len(names)}) != num_classes ({num_classes}); using provided names anyway.")
    return names

def build_models(args) -> Tuple[nn.Module, nn.Module, int]:
    """load backbone+classifier from a training checkpoint shaped as:
       {'model','classifier','optimizer','epoch','best_acc'}"""
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if ("model" not in ckpt) or ("classifier" not in ckpt):
        raise KeyError("Checkpoint must contain keys: 'model' and 'classifier'")

    model_sd = strip_module(ckpt["model"])
    clf_sd   = strip_module(ckpt["classifier"])

    # num_classes from classifier weight shape
    w_key = detect_weight_key(clf_sd)
    num_classes = clf_sd[w_key].shape[0]  # [num_classes, feat_dim]

    # build modules (LinearClassifier infers in_features by 'name')
    backbone   = SupConResNet(name=args.model)
    classifier = LinearClassifier(name=args.model, num_classes=num_classes)

    device = torch.device(args.device)
    backbone   = backbone.to(device)
    classifier = classifier.to(device)

    # load safely (keys may not match strictly due to buffers/unused heads)
    miss, unexp = backbone.load_state_dict(model_sd, strict=False)
    if miss or unexp:
        print(f"ℹ️ backbone loaded with missing={len(miss)} unexpected={len(unexp)} keys")
    classifier.load_state_dict(clf_sd, strict=True)

    backbone.eval()
    classifier.eval()
    return backbone, classifier, num_classes

@torch.no_grad()
def run_inference(args):
    device = torch.device(args.device)

    backbone, classifier, num_classes = build_models(args)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    paths = list_images(args.input)
    ds = ImageDataset(paths, tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    names = load_class_names(args.classes, num_classes=num_classes)
    if names is None:
        names = [str(i) for i in range(num_classes)]

    softmax = nn.Softmax(dim=1)
    all_rows = []
    topk = min(args.topk, num_classes)

    for images, batch_paths in dl:
        images = images.to(device, non_blocking=True)
        feats  = backbone.encoder(images)    # same as training
        logits = classifier(feats)
        probs  = softmax(logits)

        pv, pi = probs.topk(topk, dim=1)  # values, indices
        for i in range(images.size(0)):
            p = batch_paths[i]
            idxs = pi[i].tolist()
            vals = pv[i].tolist()

            print(f"\n{p}")
            for k, (ci, v) in enumerate(zip(idxs, vals), start=1):
                print(f"  Top{k}: {names[ci]}  prob={v:.4f}")

            row = {"path": p}
            for k in range(topk):
                row[f"top{k+1}_class"] = names[idxs[k]]
                row[f"top{k+1}_prob"]  = float(vals[k])
            all_rows.append(row)

    if args.output_csv:
        import csv
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        cols = ["path"] + sum([[f"top{i}_class", f"top{i}_prob"] for i in range(1, topk+1)], [])
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(f"\n✅ Saved predictions to: {args.output_csv}")

def main():
    args = parse_args()
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    run_inference(args)

if __name__ == "__main__":
    main()
