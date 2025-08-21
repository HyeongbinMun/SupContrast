import os, argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import cv2
from matplotlib import cm

from networks.resnet_big import SupConResNet, LinearClassifier

def parse_args():
    p = argparse.ArgumentParser("Visualize classifier activation maps (Grad-CAM)")
    p.add_argument("--ckpt", type=str, required=True,
                   help="trained checkpoint (.pth, with keys: model, classifier)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="root dir of images (ImageFolder 형태)")
    p.add_argument("--output_dir", type=str, default="./activation_maps")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Grad-CAM은 보통 한 장씩 처리")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def strip_module(sd):
    new = {}
    for k,v in sd.items():
        new[k.replace("module.","")] = v
    return new

def load_models(args, num_classes):
    backbone = SupConResNet(name=args.model)
    classifier = LinearClassifier(name=args.model, num_classes=num_classes)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_sd = strip_module(ckpt["model"])
    clf_sd   = strip_module(ckpt["classifier"])
    device = torch.device(args.device)
    backbone, classifier = backbone.to(device), classifier.to(device)
    backbone.load_state_dict(model_sd, strict=False)
    classifier.load_state_dict(clf_sd, strict=True)
    backbone.eval(); classifier.eval()
    return backbone, classifier

class GradCAM:
    def __init__(self, backbone, classifier, target_layer="layer4"):
        self.backbone = backbone.encoder
        self.classifier = classifier
        self.device = next(backbone.parameters()).device
        self.target_layer = dict([*self.backbone.named_modules()])[target_layer]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, x, class_idx=None):
        # Forward
        feats = self.backbone(x)
        scores = self.classifier(feats)
        if class_idx is None:
            class_idx = scores.argmax(dim=1).item()

        # Backward
        self.backbone.zero_grad(); self.classifier.zero_grad()
        one_hot = torch.zeros_like(scores)
        one_hot[0, class_idx] = 1
        scores.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM
        grads = self.gradients.mean(dim=[2,3], keepdim=True)  # GAP over H,W
        cam = torch.sum(grads * self.activations, dim=1)      # [B,H,W]
        cam = torch.relu(cam)
        cam = cam[0].cpu().numpy()
        cam = cv2.resize(cam, (x.shape[2], x.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

def overlay_heatmap(img_tensor, cam):
    img = img_tensor.permute(1,2,0).cpu().numpy()
    img = (img - img.min()) / (img.max()-img.min()+1e-8)
    img = (img*255).astype(np.uint8)
    heatmap = (cm.jet(cam)[:,:,:3]*255).astype(np.uint8)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    return overlay

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    idx_to_class = {v:k for k,v in ds.class_to_idx.items()}
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers)

    num_classes = len(idx_to_class)
    backbone, classifier = load_models(args, num_classes)
    gradcam = GradCAM(backbone, classifier, target_layer="layer4")

    for i,(imgs,labels) in enumerate(dl):
        imgs = imgs.to(args.device)
        cam, pred = gradcam.generate(imgs, None)
        overlay = overlay_heatmap(imgs[0].cpu(), cam)
        cls_name = idx_to_class[pred]
        save_path = os.path.join(args.output_dir, f"{i:06d}_{cls_name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            print(f"[{i}/{len(ds)}] saved {save_path}")

if __name__=="__main__":
    main()
