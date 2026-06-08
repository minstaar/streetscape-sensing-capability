# -*- coding: utf-8 -*-
"""
08b_extract_segmentation.py — ADE20K 시맨틱 세그멘테이션 해석가능 피처 (RQ2)
=============================================================================
[역할]
    08_extract_features.py(DINOv2 블랙박스)와 병렬로,
    각 상권 이미지의 *해석가능* 가로경관 구성비를 산출한다.
    능력지도 RQ2: "이미지가 직장인구/유형을 맞히는 신호가
    어떤 시각요소(건물밀도/점포정면/간판/녹지/개방감)에서 오는가"

[방법]
    SegFormer (nvidia/segformer-b0-finetuned-ade-512-512, ADE20K 150클래스)
    각 이미지 픽셀을 분류 -> 관심 클래스군 비율 산출 -> 상권별 평균 풀링.

[관심 피처(상권별 컬럼)]
    seg_building  건물(building/house/skyscraper/wall)   건물밀도/상업등급
    seg_window    창문/문(window/door)                    점포 정면(storefront) 대리
    seg_sign      간판/차양/기둥(signboard/awning/pole)   상업 활성
    seg_road      도로(road/path)                          가로 폭/차량중심
    seg_sidewalk  보도(sidewalk)                           보행환경
    seg_green     녹지(tree/grass/plant/palm)              경관 쾌적성
    seg_sky       하늘(sky)                                개방감(SVF 대리)
    seg_entropy   클래스 분포 엔트로피                      시각 복잡도
    (보행자·차량은 촬영시점 의존(일시적)이라 가로경관 속성이 아님 → 추출 제외)

[입력]
    data/images_poi/{상권_코드}/*.jpg     (06d POI 수집 결과 = canonical)
    data/processed/valid_image_sangkwon.csv   (flagged=False 대상)
[출력]
    data/processed/image_segmentation_poi.csv
        columns: 상권_코드, n_images, seg_building ... seg_car

[실행]  (로컬 .venv - torch 필요)
    pip install transformers torch pillow tqdm     # 최초 1회
    python scripts/08b_extract_segmentation.py
    # GPU 자동 사용. CPU도 동작하나 느림(이미지당 ~1초).

[의존성]  transformers, torch, pillow, numpy, pandas, tqdm
"""

import os
import glob
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
IMG_DIR    = ROOT / "data/images_poi"                       # canonical: POI 수집
VALID_CSV  = ROOT / "data/processed/valid_image_sangkwon.csv"
OUT_CSV    = ROOT / "data/processed/image_segmentation_poi.csv"

MODEL_ID   = "nvidia/segformer-b0-finetuned-ade-512-512"
RESIZE     = 512
RESUME     = True   # 이미 처리한 상권은 건너뜀(중단 후 재개 가능)

# ADE20K 클래스명 -> 해석가능 피처군 매핑(부분문자열 매칭)
GROUPS = {
    "building": ["building", "house", "skyscraper", "wall", "hovel"],
    "window":   ["window", "door", "windowpane"],
    "sign":     ["signboard", "awning", "pole", "streetlight", "bulletin", "booth"],
    "road":     ["road", "path", "earth"],
    "sidewalk": ["sidewalk", "pavement"],
    "green":    ["tree", "grass", "plant", "palm", "flower", "field"],
    "sky":      ["sky"],
}
# 보행자(person)·차량(car)은 촬영시점 의존(일시적)이라 가로경관 속성이 아님 → 추출에서 제외.


def build_class_index(id2label):
    """ADE20K id->label 에서 각 피처군에 속하는 클래스 id 집합을 만든다."""
    grp_ids = {g: set() for g in GROUPS}
    for cid, name in id2label.items():
        nm = str(name).lower()
        for g, keys in GROUPS.items():
            if any(k in nm for k in keys):
                grp_ids[g].add(int(cid))
    return grp_ids


def seg_fractions(pred, grp_ids, n_classes):
    """픽셀 클래스맵 -> 피처군별 비율 + 클래스 엔트로피."""
    flat = pred.ravel()
    total = flat.size
    counts = np.bincount(flat, minlength=n_classes).astype(float)
    fr = {g: counts[list(ids)].sum() / total if ids else 0.0
          for g, ids in grp_ids.items()}
    p = counts / total
    p = p[p > 0]
    fr["entropy"] = float(-(p * np.log(p)).sum())
    return fr


def _flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV)
        old["상권_코드"] = old["상권_코드"].astype(str)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("상권_코드", keep="last")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


def main():
    import torch
    from PIL import Image
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[08b] device={device}  model={MODEL_ID}")
    proc  = SegformerImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID).to(device).eval()
    id2label = model.config.id2label
    n_classes = len(id2label)
    grp_ids = build_class_index(id2label)
    for g, ids in grp_ids.items():
        print(f"  {g:<9}: {len(ids)} ADE20K 클래스")

    valid = pd.read_csv(VALID_CSV)
    valid["상권_코드"] = valid["상권_코드"].astype(str).str.strip()
    codes = valid[valid["flagged"] == False]["상권_코드"].tolist()

    done = set()
    if RESUME and OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        prev["상권_코드"] = prev["상권_코드"].astype(str)
        done = set(prev["상권_코드"])
        print(f"  재개: 이미 처리 {len(done)}개 상권 건너뜀")

    rows = []
    feat_cols = list(GROUPS.keys()) + ["entropy"]
    for code in tqdm(codes, desc="상권"):
        if code in done:
            continue
        imgs = glob.glob(str(IMG_DIR / code / "*.jpg"))
        if not imgs:
            continue
        acc = defaultdict(list)
        for ip in imgs:
            try:
                im = Image.open(ip).convert("RGB").resize((RESIZE, RESIZE))
            except Exception:
                continue
            inputs = proc(images=im, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits          # (1,150,H/4,W/4)
            up = torch.nn.functional.interpolate(
                logits, size=(RESIZE, RESIZE), mode="bilinear", align_corners=False)
            pred = up.argmax(1)[0].cpu().numpy()
            fr = seg_fractions(pred, grp_ids, n_classes)
            for k, v in fr.items():
                acc[k].append(v)
        if not acc:
            continue
        row = {"상권_코드": code, "n_images": len(acc["entropy"])}
        for k in feat_cols:
            row[f"seg_{k}"] = float(np.mean(acc[k]))
        rows.append(row)

        if len(rows) % 25 == 0:                          # 주기적 저장(중단 대비)
            _flush(rows)
            done |= {r["상권_코드"] for r in rows}
            rows = []

    _flush(rows)
    print(f"[08b] 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
