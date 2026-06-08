# -*- coding: utf-8 -*-
"""
06b_test_new_collection.py — 새 수집 방법론 테스트
─────────────────────────────────────────────────────────────────────────────
[목적]
    기존 방법 (수직 ±90°, pitch=-10°, 20m 간격) vs
    새 방법 (4방향 고정, pitch=0°, 50m 간격)
    → 소수 상권(TEST_CODES)에서만 테스트 수집 후 CLIP 점수 비교

[결과물]
    data/images_test_new/{상권코드}/   ← 새 방법 이미지
    test_comparison_report.csv         ← CLIP 점수 비교표

[실행]
    python scripts/06b_test_new_collection.py

[비용]
    상권 6개 × 평균 6포인트 × 4방향 ≈ 144장 × $0.007 ≈ $1.0
"""

import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import osmnx as ox
import torch
from PIL import Image
from shapely.geometry import Point
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

# ── 환경변수 ───────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
SHP_PATH     = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
LIST_PATH    = ROOT / "data/processed/final_sangkwon_list.csv"
OLD_IMG_DIR  = ROOT / "data/images_test"        # 기존 방법 이미지 (이미 있음)
NEW_IMG_DIR  = ROOT / "data/images_test_new"    # 새 방법 이미지 (여기서 수집)
REPORT_PATH  = ROOT / "data/processed/test_comparison_report.csv"

# ── 테스트 대상 상권 (images_test에 이미 있는 6개) ────────────────────────────
TEST_CODES = ["3110006", "3110009", "3110017", "3120046", "3120102", "3120105"]

# ── 새 수집 파라미터 ───────────────────────────────────────────────────────────
HEADINGS_NEW    = [0, 90, 180, 270]   # 4방향 고정 (도로 방향 무관)
PITCH_NEW       = 0                   # 수평 시야 (기존: -10)
INTERVAL_NEW    = 50                  # 포인트 간격 m (기존: 20)
N_PTS_NEW       = 6                   # 상권당 포인트 수 (기존: 8~15)
MIN_SPREAD_NEW  = 50                  # 포인트 간 최소 거리 m
SV_RADIUS       = 50                  # 파노라마 탐색 반경 m
MAX_SNAP_M      = 50
IMG_SIZE        = "640x640"
FOV             = 90

# ── CLIP ──────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"
POSITIVE_PROMPT = "street view of Korean commercial district with shops and store signs"
NEGATIVE_PROMPT = "empty road highway with no stores or pedestrians"

# ── Street View API ────────────────────────────────────────────────────────────
SV_BASE = "https://maps.googleapis.com/maps/api/streetview"
SV_META = "https://maps.googleapis.com/maps/api/streetview/metadata"


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════
def dist_m(lat1, lng1, lat2, lng2):
    return math.hypot(lat1 - lat2, lng1 - lng2) * 111_000


def get_pano_location(lat, lng):
    try:
        r = requests.get(
            SV_META,
            params={"location": f"{lat},{lng}", "radius": SV_RADIUS, "key": API_KEY},
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "OK":
            return None
        loc = data["location"]
        return float(loc["lat"]), float(loc["lng"]), data.get("date", "")
    except Exception:
        return None


def download_image(lat, lng, heading, pitch, out_path):
    if out_path.exists():
        return True
    try:
        r = requests.get(
            SV_BASE,
            params={
                "size": IMG_SIZE, "location": f"{lat},{lng}",
                "heading": heading, "pitch": pitch,
                "fov": FOV, "key": API_KEY,
            },
            timeout=15,
        )
        if r.status_code == 200 and len(r.content) > 5000:
            out_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 샘플 포인트 추출 (격자형 — 도로 무관)
# ══════════════════════════════════════════════════════════════════════════════
def sample_grid_points(polygon, n_pts, min_spread_m):
    """
    폴리곤 내부를 격자 탐색 → n_pts개 공간 분산 포인트 선택
    도로 방향에 의존하지 않음 (4방향 고정이므로 도로 방향 무의미)
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    step = INTERVAL_NEW / 111_000

    candidates = []
    x = bounds[0]
    while x <= bounds[2]:
        y = bounds[1]
        while y <= bounds[3]:
            if polygon.contains(Point(x, y)):
                candidates.append((y, x))  # (lat, lng)
            y += step
        x += step

    if not candidates:
        # 폴리곤이 너무 작으면 중심점만
        c = polygon.centroid
        return [(c.y, c.x)]

    # 공간 분산 샘플링 (greedy)
    min_deg = min_spread_m / 111_000
    selected = []
    for pt in candidates:
        if len(selected) >= n_pts:
            break
        if not selected or min(
            math.hypot(pt[0] - s[0], pt[1] - s[1]) for s in selected
        ) >= min_deg:
            selected.append(pt)

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# CLIP 점수 계산
# ══════════════════════════════════════════════════════════════════════════════
def load_clip(device):
    print(f"  CLIP 모델 로드 중... (device: {device})")
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model.eval()
    return model, processor


@torch.no_grad()
def compute_clip_score(img_path, model, processor, text_features, device):
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        vis_out = model.vision_model(**inputs)
        img_feat = model.visual_projection(vis_out.pooler_output)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        score = (img_feat @ text_features.T).squeeze()
        return score[0].item() - score[1].item() * 0.5
    except Exception:
        return -1.0


def score_directory(img_dir, code, model, processor, text_features, device):
    """디렉토리 내 모든 jpg CLIP 점수 계산 → 리스트 반환"""
    results = []
    img_paths = list((img_dir / code).glob("*.jpg"))
    for p in img_paths:
        s = compute_clip_score(p, model, processor, text_features, device)
        results.append({"file": p.name, "clip_score": s})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("06b_test_new_collection.py — 새 수집 방법론 테스트")
    print("=" * 65)

    if not API_KEY:
        print("  ⚠ GOOGLE_STREETVIEW_API_KEY 없음. .env 파일 확인")
        return

    # ── 상권 폴리곤 로드 ──────────────────────────────────────────────────────
    print("\n[1] 상권 폴리곤 로드...")
    gdf = gpd.read_file(SHP_PATH, encoding="euc-kr")

    # 컬럼명 정규화 (SHP 원본 컬럼: TRDAR_CD)
    gdf = gdf.rename(columns={"TRDAR_CD": "상권_코드"})
    gdf["상권_코드"] = gdf["상권_코드"].astype(str)
    gdf = gdf[gdf["상권_코드"].isin(TEST_CODES)].copy()
    gdf = gdf.to_crs(epsg=4326)
    print(f"  대상 상권: {len(gdf)}개")

    NEW_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 새 방법으로 이미지 수집 ───────────────────────────────────────────────
    print("\n[2] 새 방법론으로 이미지 수집 (4방향, pitch=0°, 50m 간격)")
    print(f"    Headings: {HEADINGS_NEW}  |  Pitch: {PITCH_NEW}°  |  Interval: {INTERVAL_NEW}m")
    total_downloaded = 0
    total_skipped    = 0

    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="상권 수집"):
        code    = row["상권_코드"]
        polygon = row.geometry
        out_dir = NEW_IMG_DIR / code
        out_dir.mkdir(exist_ok=True)

        # 격자 샘플 포인트
        pts = sample_grid_points(polygon, N_PTS_NEW, MIN_SPREAD_NEW)
        print(f"\n  {code}: {len(pts)}개 포인트")

        for (lat, lng) in pts:
            # 파노라마 스냅 확인
            result = get_pano_location(lat, lng)
            if result is None:
                continue
            p_lat, p_lng, capture_date = result
            if dist_m(lat, lng, p_lat, p_lng) > MAX_SNAP_M:
                print(f"    스킵 (파노라마 너무 멀리: {dist_m(lat,lng,p_lat,p_lng):.0f}m)")
                continue

            print(f"    포인트 ({p_lat:.5f}, {p_lng:.5f})  촬영: {capture_date}")

            for h in HEADINGS_NEW:
                fname = f"{p_lat:.6f}_{p_lng:.6f}_{h}.jpg"
                out_path = out_dir / fname
                ok = download_image(p_lat, p_lng, h, PITCH_NEW, out_path)
                if ok:
                    total_downloaded += 1
                    print(f"      ✓ heading={h}°")
                else:
                    total_skipped += 1
                time.sleep(0.05)

    print(f"\n  다운로드 완료: {total_downloaded}장  |  스킵: {total_skipped}장")

    # ── CLIP 점수 비교 ────────────────────────────────────────────────────────
    print("\n[3] CLIP 점수 계산 (기존 vs 새 방법)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_clip(device)

    texts = [POSITIVE_PROMPT, NEGATIVE_PROMPT]
    t_inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_out = model.text_model(**t_inputs)
        text_features = model.text_projection(text_out.pooler_output)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    records = []
    for code in TEST_CODES:
        # 기존 방법
        old_dir = OLD_IMG_DIR / code
        if old_dir.exists():
            old_scores = score_directory(OLD_IMG_DIR, code, model, processor, text_features, device)
        else:
            old_scores = []

        # 새 방법
        new_dir = NEW_IMG_DIR / code
        if new_dir.exists():
            new_scores = score_directory(NEW_IMG_DIR, code, model, processor, text_features, device)
        else:
            new_scores = []

        old_vals = [r["clip_score"] for r in old_scores]
        new_vals = [r["clip_score"] for r in new_scores]

        OLD_THRESH = 0.12
        NEW_THRESH = 0.22

        records.append({
            "상권_코드"          : code,
            "기존_이미지수"       : len(old_vals),
            "기존_평균CLIP"       : round(np.mean(old_vals), 4) if old_vals else None,
            "기존_중앙CLIP"       : round(np.median(old_vals), 4) if old_vals else None,
            f"기존_0.12통과율(%)" : round(100 * np.mean([s >= OLD_THRESH for s in old_vals]), 1) if old_vals else None,
            f"기존_0.22통과율(%)" : round(100 * np.mean([s >= NEW_THRESH for s in old_vals]), 1) if old_vals else None,
            "새_이미지수"         : len(new_vals),
            "새_평균CLIP"         : round(np.mean(new_vals), 4) if new_vals else None,
            "새_중앙CLIP"         : round(np.median(new_vals), 4) if new_vals else None,
            f"새_0.12통과율(%)"   : round(100 * np.mean([s >= OLD_THRESH for s in new_vals]), 1) if new_vals else None,
            f"새_0.22통과율(%)"   : round(100 * np.mean([s >= NEW_THRESH for s in new_vals]), 1) if new_vals else None,
        })

    df = pd.DataFrame(records)
    df.to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("CLIP 점수 비교 결과")
    print(f"{'=' * 65}")
    print(f"\n{'상권':>10}  {'기존 평균':>8}  {'기존 0.22통과':>12}  {'새 평균':>8}  {'새 0.22통과':>11}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(
            f"{r['상권_코드']:>10}  "
            f"{r['기존_평균CLIP']:>8.4f}  "
            f"{r['기존_0.22통과율(%)']:>12.1f}%  "
            f"{r['새_평균CLIP']:>8.4f}  "
            f"{r['새_0.22통과율(%)']:>11.1f}%"
        )
    print("-" * 60)

    # 전체 평균
    all_old = []
    all_new = []
    for code in TEST_CODES:
        old_dir = OLD_IMG_DIR / code
        new_dir = NEW_IMG_DIR / code
        if old_dir.exists():
            all_old += [compute_clip_score(p, model, processor, text_features, device)
                        for p in old_dir.glob("*.jpg")]
        if new_dir.exists():
            all_new += [compute_clip_score(p, model, processor, text_features, device)
                        for p in new_dir.glob("*.jpg")]

    print(f"\n  [전체 평균]")
    print(f"  기존 방법: {len(all_old)}장  평균={np.mean(all_old):.4f}  0.22통과율={100*np.mean([s>=0.22 for s in all_old]):.1f}%")
    print(f"  새  방법: {len(all_new)}장  평균={np.mean(all_new):.4f}  0.22통과율={100*np.mean([s>=0.22 for s in all_new]):.1f}%")
    print(f"\n  비교 결과 → {REPORT_PATH}")
    print(f"\n  ✅ 새 방법 0.22 통과율이 기존보다 높으면 → 재수집 진행")
    print(f"  ❌ 비슷하면 → 다른 방법 검토 필요")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
