# tsne_from_ckpt.py
from __future__ import annotations
import os, argparse, glob
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 프로젝트 내부 모듈 (SupContrast 구조 가정)
from networks.resnet_big import SupConResNet

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def parse_args():
    p = argparse.ArgumentParser("t-SNE from trained SupCon checkpoint over an image directory")
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to pretrained checkpoint (.pth) containing 'model' or 'state_dict'")
    p.add_argument("--data_dir", type=str, required=True,
                   help="root dir of images (ImageFolder layout: class subfolders)")
    p.add_argument("--model", type=str, default="resnet50",
                   help="backbone name used during pretraining (e.g., resnet50)")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--mean", nargs="+", type=float, default=[0.5, 0.5, 0.5])
    p.add_argument("--std",  nargs="+", type=float, default=[0.5, 0.5, 0.5])

    p.add_argument("--max_per_class", type=int, default=-1,
                   help="max samples per class for t-SNE; -1 uses all")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=42)

    p.add_argument("--output_png", type=str, default="tsne.png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        out[k.replace("module.", "")] = v
    return out

def load_backbone(args) -> nn.Module:
    model = SupConResNet(name=args.model)
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # 다양한 저장 포맷 지원
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        # 바로 state_dict일 가능성
        state = ckpt

    state = strip_module(state)

    device = torch.device(args.device)
    if torch.cuda.is_available() and device.type == "cuda":
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.to(device)
    cudnn.benchmark = True

    # 엄격하지 않게 로드 (헤드 등 불일치 허용)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ℹ️ Loaded with missing={len(missing)}, unexpected={len(unexpected)} keys")

    model.eval()
    return model

class PlainImageFolder(datasets.ImageFolder):
    # torchvision ImageFolder 그대로 사용
    pass

def build_loader(args) -> Tuple[DataLoader, Dict[int, str]]:
    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])
    ds = PlainImageFolder(root=args.data_dir, transform=tfm)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    return dl, idx_to_class

@torch.no_grad()
def extract_embeddings(model: nn.Module, dl: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    feats_list = []
    labels_list = []
    for images, labels in dl:
        images = images.to(device, non_blocking=True)
        # SupCon에서 분류기 학습 전에 쓰는 encoder 임베딩
        emb = model.encoder(images)      # [B, D]
        emb = emb.detach().cpu().numpy()
        feats_list.append(emb)
        labels_list.append(labels.numpy())
    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return feats, labels

def subsample_per_class(feats: np.ndarray, labels: np.ndarray, max_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_per_class is None or max_per_class < 0:
        return feats, labels
    sel_feats, sel_labels = [], []
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        if len(idxs) > max_per_class:
            choose = np.random.choice(idxs, size=max_per_class, replace=False)
        else:
            choose = idxs
        sel_feats.append(feats[choose])
        sel_labels.append(labels[choose])
    return np.concatenate(sel_feats, axis=0), np.concatenate(sel_labels, axis=0)

def run_tsne_plot(feats: np.ndarray, labels: np.ndarray, idx_to_class: Dict[int, str],
                  perplexity: float, n_iter: int, random_state: int,
                  output_png: str, dpi: int):
    print(f"t-SNE fitting on {feats.shape[0]} samples (dim={feats.shape[1]}) ...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state, init="pca")
    tsne_xy = tsne.fit_transform(feats)

    plt.figure(figsize=(8, 8))
    classes = np.unique(labels)
    for c in classes:
        mask = labels == c
        plt.scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], s=4, alpha=0.6, label=idx_to_class.get(int(c), str(c)))
    plt.legend(markerscale=3, frameon=True, fontsize=8)
    plt.title(f"t-SNE (perplexity={perplexity}, n_iter={n_iter})  N={feats.shape[0]}")
    plt.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved t-SNE plot to: {output_png}")

def main():
    args = parse_args()
    device = torch.device(args.device)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    model = load_backbone(args)
    dl, idx_to_class = build_loader(args)

    feats, labels = extract_embeddings(model, dl, torch.device(args.device))
    print(f"Collected features: {feats.shape}, labels: {labels.shape} (num_classes={len(np.unique(labels))})")

    feats_s, labels_s = subsample_per_class(feats, labels, args.max_per_class)
    if feats_s.shape[0] != feats.shape[0]:
        print(f"Subsampled per class to max {args.max_per_class}: N {feats.shape[0]} -> {feats_s.shape[0]}")

    run_tsne_plot(
        feats_s, labels_s, idx_to_class,
        perplexity=args.perplexity, n_iter=args.n_iter, random_state=args.random_state,
        output_png=args.output_png, dpi=args.dpi
    )

if __name__ == "__main__":
    main()
