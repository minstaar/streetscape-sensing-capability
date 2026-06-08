# -*- coding: utf-8 -*-
"""
08_extract_features.py — DINOv2 이미지 피처 추출
=============================================================================
[역할]
    07c_restore_flagged.py 이후 단계.
    유효 이미지(valid_image_sangkwon.csv, flagged=False)를 대상으로
    DINOv2-base CLS 토큰(768차원)을 추출, 상권별 평균 풀링 → 상권 1개 = 벡터 1개.
    (보고서 능력지도의 canonical 블랙박스 피처)

[처리 흐름]
    1. valid_image_sangkwon.csv 에서 flagged=False 상권 목록 로드
    2. data/images_poi/{상권_코드}/*.jpg 로드
    3. DINOv2 CLS 토큰 추출 (768-dim, L2 정규화)
    4. 이미지별 피처를 상권 단위로 평균 풀링
    5. CSV 저장 (RESUME=append/skip)

[출력]
    data/processed/image_features_dino_poi.csv   ← canonical (보고서 사용)
        columns: 상권_코드, dino_0 … dino_767

[모델]
    DINOv2-base: facebook/dinov2-base (HuggingFace), CLS 토큰 768-dim, ImageNet 정규화.
    처음 실행 시 모델 자동 다운로드 (~330MB).

실행:
    python scripts/08_extract_features.py
의존성:
    pip install torch torchvision transformers Pillow tqdm
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).resolve().parents[1]
IMG_DIR      = ROOT / "data/images_poi"
VALID_CSV    = ROOT / "data/processed/valid_image_sangkwon.csv"
OUT_DIR      = ROOT / "data/processed"
DINO_OUT     = "image_features_dino_poi.csv"

DINO_MODEL_ID = "facebook/dinov2-base"   # 768-dim CLS 토큰

# ImageNet 정규화
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


def load_dino(device):
    print(f"  DINOv2 로드 중 ({DINO_MODEL_ID}) ...")
    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    return model.to(device).eval()


@torch.no_grad()
def extract_dino(image_paths, model, device):
    """이미지 리스트 → DINOv2 CLS 토큰 평균. 반환: np.ndarray(768,) 또는 None."""
    feats = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            x   = TRANSFORM(img).unsqueeze(0).to(device)
            out = model(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # (768,)
            feats.append(cls)
        except Exception:
            continue
    if not feats:
        return None
    arr = np.stack(feats)
    mean_feat = arr.mean(axis=0)
    return mean_feat / (np.linalg.norm(mean_feat) + 1e-8)   # L2 정규화


def main():
    print("=" * 65)
    print("08_extract_features.py — DINOv2 피처 추출")
    print("=" * 65)

    valid_df    = pd.read_csv(VALID_CSV)
    valid_codes = (valid_df[valid_df["flagged"] == False]["상권_코드"]
                   .astype(str).tolist())
    print(f"\n  유효 상권: {len(valid_codes)}개  (flagged=False 기준)")

    dino_path = OUT_DIR / DINO_OUT
    done_dino = set()
    if dino_path.exists():
        done_dino = set(pd.read_csv(dino_path)["상권_코드"].astype(str))
        print(f"  DINOv2 기완료: {len(done_dino)}개 → 건너뜀")

    todo_dino = [c for c in valid_codes if c not in done_dino]
    if not todo_dino:
        print("\n  모든 상권 추출 완료. 재실행 불필요.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  디바이스: {device}\n")
    dino_model = load_dino(device)
    print()

    dino_records = []
    for code in tqdm(todo_dino, desc="DINOv2 추출"):
        img_dir = IMG_DIR / code
        if not img_dir.exists():
            continue
        image_paths = sorted(img_dir.glob("*.jpg"))
        if not image_paths:
            continue
        feat = extract_dino(image_paths, dino_model, device)
        if feat is not None:
            row = {"상권_코드": code}
            row.update({f"dino_{i}": float(v) for i, v in enumerate(feat)})
            dino_records.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if dino_records:
        new_df = pd.DataFrame(dino_records)
        if dino_path.exists():
            old_df = pd.read_csv(dino_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates("상권_코드", keep="last")
        new_df.to_csv(dino_path, index=False, encoding="utf-8-sig")

    total_dino = len(pd.read_csv(dino_path)) if dino_path.exists() else 0
    print(f"\n{'=' * 65}")
    print(f"  DINOv2-base: {total_dino}개 상권, 768차원  →  {dino_path.name}")
    print("=" * 65)
    print("\n다음 단계: 08b_extract_segmentation.py → 08c_extract_handcrafted.py → 10_capability_map.py")


if __name__ == "__main__":
    main()
