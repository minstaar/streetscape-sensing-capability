# -*- coding: utf-8 -*-
"""
08_extract_features.py — ResNet-50 & DINOv2 이미지 피처 추출
─────────────────────────────────────────────────────────────────────────────
[역할]
    07_filter_images.py 이후 단계.
    유효 이미지(valid_image_sangkwon.csv, flagged=False)를 대상으로:
      ① ResNet-50 (ImageNet pretrained) — 2048차원 피처 추출
      ② DINOv2-base (self-supervised pretrained) — 768차원 CLS 토큰 추출
    각 상권의 이미지들을 평균 풀링(mean pooling)하여
    상권 1개 = 벡터 1개 로 집계

[처리 흐름]
    1. valid_image_sangkwon.csv 에서 flagged=False 상권 목록 로드
    2. data/images/{상권_코드}/*.jpg 로드
    3. ResNet-50 avgpool 출력 추출 (2048-dim)
    4. DINOv2 CLS 토큰 추출 (768-dim)
    5. 이미지별 피처를 상권 단위로 평균 풀링
    6. CSV 저장

[출력]
    data/processed/image_features_resnet.csv
        columns: 상권_코드, resnet_0 … resnet_2047
    data/processed/image_features_dino.csv
        columns: 상권_코드, dino_0 … dino_767

[모델 정보]
    ResNet-50  : torchvision, IMAGENET1K_V2 weights
                 avgpool 출력 → 2048-dim, L2 정규화
    DINOv2-base: facebook/dinov2-base (HuggingFace)
                 CLS 토큰 → 768-dim, L2 정규화
                 patch size 14, 입력 224×224

[주의]
    - GPU 강력 권장 (GTX 4070 Super 등)
    - 처음 실행 시 DINOv2 모델 자동 다운로드 (~330MB)
    - 이미 추출된 상권은 건너뜀 (재실행 안전)

실행:
    python scripts/08_extract_features.py

의존성:
    pip install torch torchvision transformers Pillow tqdm --break-system-packages
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
IMG_DIR      = ROOT / "data/images"
VALID_CSV    = ROOT / "data/processed/valid_image_sangkwon.csv"
OUT_DIR      = ROOT / "data/processed"

# ── 모델 설정 ──────────────────────────────────────────────────────────────────
DINO_MODEL_ID = "facebook/dinov2-base"   # 768-dim CLS 토큰

# ── 이미지 전처리 (ImageNet 정규화 — ResNet/DINOv2 공통) ─────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ══════════════════════════════════════════════════════════════════════════════
# 모델 로드
# ══════════════════════════════════════════════════════════════════════════════
def load_resnet(device):
    print("  ResNet-50 로드 중 (IMAGENET1K_V2) ...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # FC 레이어 제거 → avgpool 출력: (batch, 2048, 1, 1)
    model = nn.Sequential(*list(model.children())[:-1])
    return model.to(device).eval()


def load_dino(device):
    print(f"  DINOv2 로드 중 ({DINO_MODEL_ID}) ...")
    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════════════════
# 피처 추출
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_resnet(image_paths, model, device):
    """
    이미지 리스트 → ResNet avgpool 피처 평균
    반환: np.ndarray (2048,) 또는 None
    """
    feats = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            x   = TRANSFORM(img).unsqueeze(0).to(device)
            f   = model(x).squeeze().cpu().numpy()   # (2048,)
            feats.append(f)
        except Exception:
            continue
    if not feats:
        return None
    arr = np.stack(feats)                             # (N, 2048)
    mean_feat = arr.mean(axis=0)                      # 평균 풀링
    mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)  # L2 정규화
    return mean_feat


@torch.no_grad()
def extract_dino(image_paths, model, device):
    """
    이미지 리스트 → DINOv2 CLS 토큰 피처 평균
    반환: np.ndarray (768,) 또는 None
    """
    feats = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            x   = TRANSFORM(img).unsqueeze(0).to(device)
            out = model(pixel_values=x)
            # last_hidden_state: (1, num_patches+1, 768)
            # 0번 토큰 = CLS 토큰
            cls = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()  # (768,)
            feats.append(cls)
        except Exception:
            continue
    if not feats:
        return None
    arr = np.stack(feats)                             # (N, 768)
    mean_feat = arr.mean(axis=0)                      # 평균 풀링
    mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)  # L2 정규화
    return mean_feat


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("08_extract_features.py — ResNet-50 & DINOv2 피처 추출")
    print("=" * 65)

    # ── 유효 상권 로드 ────────────────────────────────────────────────────────
    valid_df    = pd.read_csv(VALID_CSV)
    valid_codes = (
        valid_df[valid_df["flagged"] == False]["상권_코드"]
        .astype(str).tolist()
    )
    print(f"\n  유효 상권: {len(valid_codes)}개  (flagged=False 기준)")

    # ── 이미 추출된 상권 건너뜀 (재실행 안전) ────────────────────────────────
    resnet_path = OUT_DIR / "image_features_resnet.csv"
    dino_path   = OUT_DIR / "image_features_dino.csv"

    done_resnet = set()
    done_dino   = set()
    if resnet_path.exists():
        done_resnet = set(pd.read_csv(resnet_path)["상권_코드"].astype(str))
        print(f"  ResNet 기완료: {len(done_resnet)}개 → 건너뜀")
    if dino_path.exists():
        done_dino = set(pd.read_csv(dino_path)["상권_코드"].astype(str))
        print(f"  DINOv2 기완료: {len(done_dino)}개 → 건너뜀")

    todo_resnet = [c for c in valid_codes if c not in done_resnet]
    todo_dino   = [c for c in valid_codes if c not in done_dino]

    if not todo_resnet and not todo_dino:
        print("\n  모든 상권 추출 완료. 재실행 불필요.")
        return

    # ── 디바이스 & 모델 ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  디바이스: {device}\n")

    resnet_model = load_resnet(device) if todo_resnet else None
    dino_model   = load_dino(device)   if todo_dino   else None
    print()

    # ── 추출 루프 ─────────────────────────────────────────────────────────────
    resnet_records = []
    dino_records   = []

    all_todo = sorted(set(todo_resnet) | set(todo_dino))

    for code in tqdm(all_todo, desc="피처 추출"):
        img_dir     = IMG_DIR / code
        if not img_dir.exists():
            continue
        image_paths = sorted(img_dir.glob("*.jpg"))
        if not image_paths:
            continue

        if code in todo_resnet and resnet_model is not None:
            feat = extract_resnet(image_paths, resnet_model, device)
            if feat is not None:
                row = {"상권_코드": code}
                row.update({f"resnet_{i}": float(v) for i, v in enumerate(feat)})
                resnet_records.append(row)

        if code in todo_dino and dino_model is not None:
            feat = extract_dino(image_paths, dino_model, device)
            if feat is not None:
                row = {"상권_코드": code}
                row.update({f"dino_{i}": float(v) for i, v in enumerate(feat)})
                dino_records.append(row)

    # ── 저장 (append 방식) ────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if resnet_records:
        new_df = pd.DataFrame(resnet_records)
        if resnet_path.exists():
            old_df = pd.read_csv(resnet_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(resnet_path, index=False, encoding="utf-8-sig")

    if dino_records:
        new_df = pd.DataFrame(dino_records)
        if dino_path.exists():
            old_df = pd.read_csv(dino_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(dino_path, index=False, encoding="utf-8-sig")

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    total_resnet = len(pd.read_csv(resnet_path)) if resnet_path.exists() else 0
    total_dino   = len(pd.read_csv(dino_path))   if dino_path.exists()   else 0

    print(f"\n{'=' * 65}")
    print("피처 추출 완료")
    print(f"  ResNet-50  : {total_resnet}개 상권, 2048차원  →  {resnet_path.name}")
    print(f"  DINOv2-base: {total_dino}개 상권,  768차원  →  {dino_path.name}")
    print(f"{'=' * 65}")
    print("\n다음 단계: python scripts/09_multimodal_regression.py")


if __name__ == "__main__":
    main()
