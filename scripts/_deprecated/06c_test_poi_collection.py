# -*- coding: utf-8 -*-
"""
06c_test_poi_collection.py — POI 기반 이미지 수집 테스트
─────────────────────────────────────────────────────────────────────────────
[목적]
    소상공인 상가(상권)정보 개별 점포 좌표(POI)를 기반으로
    GSV 이미지를 수집하는 새 방식 테스트.
    → 기존 방식(도로 엣지 수직 샘플링)의 도로변 사진 혼입 문제 해결

[테스트 상권]
    PC1 상위 / 중간 / 하위 각 1개씩 (총 3개)
    상권당 최대 N_SAMPLE_PER_SQ개 POI → GSV 수집

[핵심 변경사항]
    1. 샘플링 기준: 도로 엣지 → 실제 점포 위경도(POI)
    2. Heading 계산: 파노라마 위치 → 점포 방향 자동 계산
    3. 날짜 필터: 2018년 파노라마만 사용 (pre-COVID 역인과 차단)
    4. CLIP 필터 제거 (POI 기반이라 상업 전면 자연 확보)

실행: python scripts/06c_test_poi_collection.py
"""

import math
import os
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
SHP_PATH   = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
POI_CSV    = ROOT / "data/raw/store/소상공인시장진흥공단_상가(상권)정보_서울_201903.csv"
LIST_CSV   = ROOT / "data/processed/final_sangkwon_list.csv"
DINO_CSV   = ROOT / "data/processed/image_features_dino.csv"
CROSS_CSV  = ROOT / "data/processed/cross_sectional_data.csv"
VALID_CSV  = ROOT / "data/processed/valid_image_sangkwon.csv"
OUT_DIR    = ROOT / "data/images_poi_test"   # 테스트 전용 폴더

API_KEY  = os.getenv("GOOGLE_STREETVIEW_API_KEY")
SV_BASE  = "https://maps.googleapis.com/maps/api/streetview"
SV_META  = "https://maps.googleapis.com/maps/api/streetview/metadata"

# ── 설정 ──────────────────────────────────────────────────────────────────────
TARGET_YEAR_MAX  = 2019         # 이 연도 이하 파노라마만 사용 (pre-COVID)
N_SAMPLE_QUERY   = 30           # 상권당 POI 쿼리 수 (메타데이터)
N_TARGET_IMAGES  = 20           # 상권당 목표 다운로드 수
MIN_POI_DIST_M   = 20           # POI 간 최소 거리 (중복 점포 방지)
MAX_SNAP_M       = 30           # 파노라마-POI 거리 허용 한계
IMG_SIZE         = "640x640"
FOV              = 90
PITCH            = 0            # 수평 촬영 (segmentation 정확도 향상)

# 테스트 상권: PC1 상위/중간/하위 — 데이터에서 자동 선택
# (아래 main()에서 실제 PC1 분포 기반으로 결정)
N_TEST_SANGKWON = 3   # 상위·중간·하위 각 1개

# 상업 업종만 필터 (주거/부동산 제외)
COMMERCIAL_CATEGORIES = {
    "음식", "소매", "수리·개인", "예술·스포츠", "숙박",
    "생활서비스", "관광·여가·오락"
}


# ══════════════════════════════════════════════════════════════════════════════
def dist_m(lat1, lng1, lat2, lng2):
    return math.hypot(lat1 - lat2, lng1 - lng2) * 111_000


def compute_heading(from_lat, from_lng, to_lat, to_lng):
    """파노라마 위치 → 점포 방향 heading (0~360°)"""
    dlon = math.radians(to_lng - from_lng)
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def spread_sample(pois, n, min_dist_m):
    """공간 분산 샘플링 — 가까운 POI 중복 제거"""
    min_deg = min_dist_m / 111_000
    selected = []
    for lat, lng, meta in pois:
        if len(selected) >= n:
            break
        if not selected or min(
            math.hypot(lat - s[0], lng - s[1]) for s in selected
        ) >= min_deg:
            selected.append((lat, lng, meta))
    return selected


def query_metadata(lat, lng):
    """GSV 메타데이터 조회 → (pano_id, date, pano_lat, pano_lng) or None"""
    try:
        r = requests.get(
            SV_META,
            params={"location": f"{lat},{lng}", "radius": MAX_SNAP_M,
                    "source": "outdoor", "key": API_KEY},
            timeout=10,
        )
        d = r.json()
        if d.get("status") != "OK":
            return None
        loc = d["location"]
        return {
            "pano_id":   d["pano_id"],
            "date":      d.get("date", ""),
            "pano_lat":  float(loc["lat"]),
            "pano_lng":  float(loc["lng"]),
        }
    except Exception:
        return None


def download_image(pano_id, heading, save_path):
    """pano_id로 특정 파노라마 이미지 다운로드"""
    try:
        r = requests.get(
            SV_BASE,
            params={"pano": pano_id, "size": IMG_SIZE,
                    "heading": round(heading), "fov": FOV,
                    "pitch": PITCH, "key": API_KEY},
            timeout=20,
        )
        if r.status_code == 200 and len(r.content) > 5_000:
            save_path.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not API_KEY:
        raise ValueError(".env에 GOOGLE_STREETVIEW_API_KEY가 없습니다.")

    print("=" * 65)
    print("06c — POI 기반 이미지 수집 테스트")
    print(f"  대상 연도: {TARGET_YEAR_MAX}년 이하 파노라마만 사용")
    print(f"  상권당 최대 {N_SAMPLE_QUERY}개 POI 샘플")
    print("=" * 65)

    # ── 1. PC1 계산 후 테스트 상권 자동 선택 ────────────────────────────────
    print("\n[1] PC1 기반 테스트 상권 자동 선택 중...")
    import numpy as np
    dino  = pd.read_csv(DINO_CSV)
    valid = pd.read_csv(VALID_CSV)
    dino["상권_코드"]  = dino["상권_코드"].astype(str).str.strip()
    valid["상권_코드"] = valid["상권_코드"].astype(str).str.strip()

    valid_codes = set(valid[valid["flagged"] == False]["상권_코드"])
    dino_valid  = dino[dino["상권_코드"].isin(valid_codes)].copy().reset_index(drop=True)
    dino_cols   = [c for c in dino.columns if c.startswith("dino_")]

    X = dino_valid[dino_cols].values.astype(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = U[:, 0] * S[0]

    pca_df = dino_valid[["상권_코드"]].copy()
    pca_df["PC1"] = pc1

    cross = pd.read_csv(CROSS_CSV)
    cross["상권_코드"] = cross["상권_코드"].astype(str).str.strip()
    pca_df = pca_df.merge(cross[["상권_코드", "상권_코드_명"]], on="상권_코드")

    # 상위·중간·하위 각 1개 선택
    pca_sorted = pca_df.sort_values("PC1").reset_index(drop=True)
    n = len(pca_sorted)
    picks = {
        "PC1_LOW":  pca_sorted.iloc[int(n * 0.05)],   # 하위 5%
        "PC1_MID":  pca_sorted.iloc[int(n * 0.50)],   # 중앙값
        "PC1_HIGH": pca_sorted.iloc[int(n * 0.95)],   # 상위 5%
    }

    test_codes = {}
    for label, row in picks.items():
        test_codes[label] = {"code": row["상권_코드"], "name": row["상권_코드_명"]}
        print(f"  {label}: {row['상권_코드_명']} (PC1={row['PC1']:.2f})")

    # ── 상권 폴리곤 로드 ──────────────────────────────────────────────────────
    print("\n[2] 상권 경계 로드 중...")
    gdf = gpd.read_file(SHP_PATH, encoding="cp949")
    gdf = gdf.set_crs("EPSG:5181", allow_override=True).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={"TRDAR_CD": "상권_코드"})
    gdf["상권_코드"] = gdf["상권_코드"].astype(str).str.strip()

    # ── POI 데이터 로드 ──────────────────────────────────────────────────────
    print(f"\n[3] POI 데이터 로드 중: {POI_CSV.name}")
    poi_df = pd.read_csv(POI_CSV, encoding="utf-8-sig",
                         usecols=["상가업소번호", "상호명", "상권업종대분류명",
                                  "경도", "위도"])
    poi_df = poi_df.dropna(subset=["경도", "위도"])
    poi_df = poi_df[poi_df["상권업종대분류명"].isin(COMMERCIAL_CATEGORIES)]
    print(f"  상업 업종 점포: {len(poi_df):,}개")

    # GeoDataFrame 변환
    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df["경도"], poi_df["위도"]),
        crs="EPSG:4326"
    )

    # ── 공간 조인: POI → 상권 ────────────────────────────────────────────────
    print("\n[4] 공간 조인 중 (POI → 상권 폴리곤)...")
    test_gdf = gdf[gdf["상권_코드"].isin(
        [v["code"] for v in test_codes.values()]
    )][["상권_코드", "geometry"]].copy()

    joined = gpd.sjoin(poi_gdf, test_gdf, how="inner", predicate="within")
    print(f"  테스트 상권 내 POI: {len(joined)}개")
    for code in test_gdf["상권_코드"]:
        n = (joined["상권_코드"] == code).sum()
        name = next((v["name"] for v in test_codes.values() if v["code"] == code), code)
        print(f"    {name}: {n}개 점포")

    # ── 상권별 이미지 수집 ───────────────────────────────────────────────────
    print(f"\n[5] GSV 이미지 수집 ({TARGET_YEAR_MAX}년 이하 파노라마)")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for label, info in test_codes.items():
        code = info["code"]
        name = info["name"]

        sq_pois = joined[joined["상권_코드"] == code].copy()
        if len(sq_pois) == 0:
            print(f"\n  [{label}] {name}: POI 없음 → 스킵")
            continue

        print(f"\n  [{label}] {name}  (POI 후보: {len(sq_pois)}개)")

        # 공간 분산 샘플링
        poi_list = [(row["위도"], row["경도"], row["상호명"])
                    for _, row in sq_pois.iterrows()]
        sampled = spread_sample(poi_list, N_SAMPLE_QUERY, MIN_POI_DIST_M)
        print(f"  → 분산 샘플: {len(sampled)}개")

        img_dir = OUT_DIR / f"{code}_{label}"
        img_dir.mkdir(exist_ok=True)

        stats = defaultdict(int)
        saved_imgs = []

        for poi_lat, poi_lng, store_name in tqdm(sampled, desc=f"    {name[:8]}"):
            if stats["downloaded"] >= N_TARGET_IMAGES:
                break  # 목표 달성 시 중단

            meta = query_metadata(poi_lat, poi_lng)
            time.sleep(0.05)

            if meta is None:
                stats["no_gsv"] += 1
                continue

            # 날짜 확인 (2020년 이전만 허용)
            year_str = meta["date"][:4] if meta["date"] else "unknown"
            stats[f"year_{year_str}"] += 1

            try:
                year_int = int(year_str)
            except ValueError:
                year_int = 9999

            if year_int > TARGET_YEAR_MAX:
                stats["year_skip"] += 1
                continue

            # 파노라마 → 점포 heading 계산
            heading = compute_heading(
                meta["pano_lat"], meta["pano_lng"],
                poi_lat, poi_lng
            )

            # pano_id로 고정 다운로드
            fname = f"{poi_lat:.6f}_{poi_lng:.6f}_{round(heading)}.jpg"
            save_path = img_dir / fname

            if save_path.exists():
                stats["skipped"] += 1
                continue

            ok = download_image(meta["pano_id"], heading, save_path)
            if ok:
                stats["downloaded"] += 1
                saved_imgs.append(save_path)
            else:
                stats["fail"] += 1

            time.sleep(0.05)

        print(f"    다운로드: {stats['downloaded']}장  "
              f"연도 스킵: {stats['year_skip']}개  "
              f"GSV없음: {stats['no_gsv']}개")

        year_dist = {k: v for k, v in stats.items() if k.startswith("year_")}
        print(f"    연도 분포: {year_dist}")

        summary.append({
            "label":       label,
            "상권_코드_명": name,
            "poi_sampled":  len(sampled),
            "downloaded":   stats["downloaded"],
            "year_skip":    stats["year_skip"],
            "no_gsv":       stats["no_gsv"],
            "year_dist":    str(year_dist),
            "img_dir":      str(img_dir),
        })

    # ── 5. 결과 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("테스트 완료 요약")
    print(f"{'=' * 65}")
    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        print(df_sum[["label", "상권_코드_명", "poi_sampled",
                       "downloaded", "year_skip", "no_gsv"]].to_string(index=False))

    total = df_sum["downloaded"].sum() if not df_sum.empty else 0
    print(f"\n  총 수집: {total}장  →  {OUT_DIR}")
    print()
    print("  [확인 사항]")
    print("  1. 수집된 이미지가 상업 전면(점포 정면)을 향하는지 확인")
    print("  2. 도로변 빈 사진 비율이 기존 방식보다 낮은지 비교")
    print("  3. 2018년 이외 파노라마 비율이 높다면 TARGET_YEAR 조정 검토")
    print(f"\n  이미지 저장 위치: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
