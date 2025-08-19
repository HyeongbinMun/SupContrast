import os
import shutil
from tqdm import tqdm

def move_all_images(src_root, dst_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff")):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 먼저 이동할 이미지 총 개수 세기
    image_paths = []
    for root, _, files in os.walk(src_root):
        for file in files:
            if file.lower().endswith(exts):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    # tqdm으로 이동 진행 표시
    for idx, src_path in enumerate(tqdm(image_paths, desc="Moving images")):
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, f"{idx:06d}_{file_name}")  # 이름 중복 방지
        shutil.move(src_path, dst_path)

    print(f"{len(image_paths)} images moved to '{dst_dir}'")

# 사용 예시
if __name__ == "__main__":
    source_dir = "/ssd/hbmun/supcon/anchor/N2"
    target_dir = "/ssd/hbmun/supcon/etri/normal"
    move_all_images(source_dir, target_dir)
