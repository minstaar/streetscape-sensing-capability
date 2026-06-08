# -*- coding: utf-8 -*-
"""
06d_collect_images_poi.py — POI 기반 전체 상권 이미지 수집
─────────────────────────────────────────────────────────────────────────────
[방법론]
    소상공인시장진흥공단 상가(상권)정보의 개별 점포 좌표(POI)를 기반으로
    각 점포 전면 방향 GSV 이미지를 수집.

    기존 도로 엣지 수직 샘플링 대비 개선사항:
    ① 점포 좌표 → 상업 전면 자연 확보 (CLIP 필터 불필요)
    ② heading = 파노라마 → 점포 방향 자동 계산
    ③ 2020년 이전 파노라마만 수집 → pre-COVID 역인과 차단
    ④ pano_id 고정 다운로드 → 날짜 재현 가능

[수집 설정]
    상권당 POI 쿼리: 30개 (메타데이터 API, 무료)
    상권당 목표 이미지: 20장 (pre-COVID 파노라마)
    총 예상 이미지: ~16,800장 / 비용: ~$120

[실행]
    전체:  python scripts/06d_collect_images_poi.py
    분할:  python scripts/06d_collect_images_poi.py --part 1 --total 2
    재개:  동일 명령 재실행 (이미 수집된 상권 자동 스킵)
"""

import argparse
import math
import os
import time
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
SHP_PATH  = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
POI_CSV   = ROOT / "data/raw/store/소상공인시장진흥공단_상가(상권)정보_서울_201903.csv"
LIST_CSV  = ROOT / "data/processed/final_sangkwon_list.csv"
IMG_DIR   = ROOT / "data/images_poi"      # ★ 기존 data/images와 분리 — 덮어쓰기 없음
LOG_CSV   = ROOT / "data/processed/image_sampling_log_poi.csv"
REPORT_DIR = ROOT / "reports"

API_KEY  = os.getenv("GOOGLE_STREETVIEW_API_KEY")
SV_BASE  = "https://maps.googleapis.com/maps/api/streetview"
SV_META  = "https://maps.googleapis.com/maps/api/streetview/metadata"

# ── 수집 설정 ─────────────────────────────────────────────────────────────────
TARGET_YEAR_MAX  = 2019   # 이 연도 이하 파노라마만 허용 (pre-COVID)
N_SAMPLE_QUERY   = 30     # 상권당 POI 쿼리 수 (메타데이터)
N_TARGET_IMAGES  = 20     # 상권당 목표 다운로드 수
MIN_POI_DIST_M   = 20     # POI 간 최소 거리(m) — 중복 점포 방지
MAX_SNAP_M       = 30     # 파노라마-POI 허용 거리(m)
IMG_SIZE         = "640x640"
FOV              = 90
PITCH            = 0      # 수평 촬영

# 상업 업종 필터 (주거/부동산/시설관리 제외)
COMMERCIAL_CATEGORIES = {
    "음식", "소매", "수리·개인", "예술·스포츠", "숙박",
    "생활서비스", "관광·여가·오락", "교육",
}


# ══════════════════════════════════════════════════════════════════════════════
def dist_m(lat1, lng1, lat2, lng2):
    return math.hypot(lat1 - lat2, lng1 - lng2) * 111_000


def compute_heading(from_lat, from_lng, to_lat, to_lng):
    dlon = math.radians(to_lng - from_lng)
    lat1 = math.radians(from_lat)
    lat2 = math.radians(to_lat)
    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def spread_sample(pois, n, min_dist_m):
    min_deg = min_dist_m / 111_000
    selected = []
    for item in pois:
        if len(selected) >= n:
            break
        lat, lng = item[0], item[1]
        if not selected or min(
            math.hypot(lat - s[0], lng - s[1]) for s in selected
        ) >= min_deg:
            selected.append(item)
    return selected


def query_metadata(lat, lng):
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
            "pano_id":  d["pano_id"],
            "date":     d.get("date", ""),
            "pano_lat": float(loc["lat"]),
            "pano_lng": float(loc["lng"]),
        }
    except Exception:
        return None


def download_image(pano_id, heading, save_path):
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


def already_done(code, target=N_TARGET_IMAGES):
    """해당 상권에 이미 목표 장수 이상 수집됐으면 True"""
    d = IMG_DIR / code
    if not d.exists():
        return False
    return len(list(d.glob("*.jpg"))) >= target


# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part",  type=int, default=1)
    parser.add_argument("--total", type=int, default=1)
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError(".env에 GOOGLE_STREETVIEW_API_KEY가 없습니다.")

    print("=" * 68)
    print("06d — POI 기반 전체 상권 이미지 수집")
    print(f"  pre-COVID 기준: {TARGET_YEAR_MAX}년 이하 파노라마")
    print(f"  목표: 상권당 {N_TARGET_IMAGES}장  |  POI 쿼리: {N_SAMPLE_QUERY}개")
    if args.total > 1:
        print(f"  분할 실행: {args.part}/{args.total} 파트")
    print("=" * 68)

    # ── 1. 상권 목록 ──────────────────────────────────────────────────────────
    valid_df = pd.read_csv(LIST_CSV)
    valid_df["상권_코드"] = valid_df["상권_코드"].astype(str).str.strip()
    all_codes = valid_df["상권_코드"].tolist()

    # 파트 분할
    if args.total > 1:
        chunk = len(all_codes) // args.total
        start = (args.part - 1) * chunk
        end   = start + chunk if args.part < args.total else len(all_codes)
        all_codes = all_codes[start:end]
        print(f"  담당 상권: {start}~{end-1}번 ({len(all_codes)}개)")

    # 이미 완료된 상권 스킵
    todo = [c for c in all_codes if not already_done(c)]
    done = len(all_codes) - len(todo)
    est_imgs  = len(todo) * N_TARGET_IMAGES
    est_cost  = est_imgs * 0.007
    est_hours = len(todo) * N_SAMPLE_QUERY * 0.08 / 3600  # 초당 ~12.5 API호출
    print(f"  전체: {len(all_codes)}개  |  완료: {done}개  |  남은: {len(todo)}개")
    print(f"  예상 이미지: ~{est_imgs:,}장  |  예상 비용: ~${est_cost:.0f}")
    print(f"  예상 소요 시간: ~{est_hours:.1f}시간 (API 딜레이 포함)")
    print(f"  ★ 중단 후 재실행 시 완료 상권은 자동 스킵됩니다\n")

    input("  계속 진행하려면 Enter를 누르세요... (Ctrl+C로 중단)")
    print()

    # ── 2. 상권 폴리곤 ────────────────────────────────────────────────────────
    print("[1] 상권 경계 로드 중...")
    gdf = gpd.read_file(SHP_PATH, encoding="cp949")
    gdf = gdf.set_crs("EPSG:5181", allow_override=True).to_crs("EPSG:4326")
    gdf = gdf.rename(columns={"TRDAR_CD": "상권_코드"})
    gdf["상권_코드"] = gdf["상권_코드"].astype(str).str.strip()
    gdf = gdf[gdf["상권_코드"].isin(todo)][["상권_코드", "geometry"]]
    print(f"  로드 완료: {len(gdf)}개 상권 폴리곤")

    # ── 3. POI 데이터 로드 & 공간 조인 ───────────────────────────────────────
    print("\n[2] POI 데이터 로드 및 공간 조인 중...")
    poi_df = pd.read_csv(
        POI_CSV, encoding="utf-8-sig",
        usecols=["상가업소번호", "상호명", "상권업종대분류명", "경도", "위도"]
    ).dropna(subset=["경도", "위도"])
    poi_df = poi_df[poi_df["상권업종대분류명"].isin(COMMERCIAL_CATEGORIES)]

    poi_gdf = gpd.GeoDataFrame(
        poi_df,
        geometry=gpd.points_from_xy(poi_df["경도"], poi_df["위도"]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(poi_gdf, gdf, how="inner", predicate="within")
    print(f"  공간 조인 완료: {len(joined):,}개 점포 매핑")

    # ── 4. 상권별 수집 ────────────────────────────────────────────────────────
    print(f"\n[3] 이미지 수집 시작 ({len(todo)}개 상권)...")
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    all_logs  = []
    total_dl  = 0
    total_skip = 0

    for code in tqdm(todo, desc="상권"):
        sq_pois = joined[joined["상권_코드"] == code]
        if len(sq_pois) == 0:
            continue

        # 공간 분산 샘플링
        poi_list = [(r["위도"], r["경도"], r["상호명"])
                    for _, r in sq_pois.iterrows()]
        sampled = spread_sample(poi_list, N_SAMPLE_QUERY, MIN_POI_DIST_M)

        img_dir = IMG_DIR / code
        img_dir.mkdir(exist_ok=True)

        n_downloaded = len(list(img_dir.glob("*.jpg")))  # 기존 이미지 수
        year_counts  = defaultdict(int)

        for poi_lat, poi_lng, store_name in sampled:
            if n_downloaded >= N_TARGET_IMAGES:
                break

            meta = query_metadata(poi_lat, poi_lng)
            time.sleep(0.04)
            if meta is None:
                continue

            year_str = meta["date"][:4] if meta["date"] else "0000"
            year_counts[year_str] += 1

            try:
                year_int = int(year_str)
            except ValueError:
                continue

            if year_int > TARGET_YEAR_MAX:
                continue  # COVID 이후 파노라마 스킵

            heading  = compute_heading(
                meta["pano_lat"], meta["pano_lng"], poi_lat, poi_lng
            )
            fname    = f"{poi_lat:.6f}_{poi_lng:.6f}_{round(heading)}.jpg"
            save_path = img_dir / fname

            if save_path.exists():
                total_skip += 1
                n_downloaded += 1
                continue

            ok = download_image(meta["pano_id"], heading, save_path)
            time.sleep(0.04)

            all_logs.append({
                "상권_코드":     code,
                "filename":      fname,
                "poi_lat":       round(poi_lat, 6),
                "poi_lng":       round(poi_lng, 6),
                "pano_lat":      round(meta["pano_lat"], 6),
                "pano_lng":      round(meta["pano_lng"], 6),
                "heading":       round(heading),
                "capture_date":  meta["date"],
                "capture_year":  year_str,
                "store_name":    store_name,
                "success":       ok,
            })

            if ok:
                n_downloaded += 1
                total_dl += 1

    # ── 5. 로그 저장 ──────────────────────────────────────────────────────────
    if all_logs:
        new_df = pd.DataFrame(all_logs)
        if LOG_CSV.exists():
            old_df = pd.read_csv(LOG_CSV)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["상권_코드", "filename"])
        else:
            combined = new_df
        combined.to_csv(LOG_CSV, index=False, encoding="utf-8-sig")

    # ── 6. 요약 ───────────────────────────────────────────────────────────────
    total_files = sum(
        len(list((IMG_DIR / c).glob("*.jpg")))
        for c in all_codes if (IMG_DIR / c).exists()
    )
    completed = sum(1 for c in all_codes if already_done(c))

    print(f"\n{'=' * 68}")
    print("수집 완료 요약")
    print(f"  이번 실행 다운로드   : {total_dl:,}장")
    print(f"  스킵 (기존 파일)     : {total_skip:,}장")
    print(f"  목표 달성 상권       : {completed}/{len(all_codes)}개")
    print(f"  총 이미지 파일       : {total_files:,}장")
    print(f"  로그 저장            : {LOG_CSV}")
    print(f"{'=' * 68}")

    # 연도 분포 (로그에서)
    if all_logs:
        log_df = pd.DataFrame(all_logs)
        print("\n  [수집 이미지 연도 분포]")
        yr = log_df[log_df["success"]]["capture_year"].value_counts().sort_index()
        total_success = yr.sum()
        for y, cnt in yr.items():
            bar = "█" * int(cnt / total_success * 30)
            print(f"    {y}  {bar}  {cnt}장 ({cnt/total_success*100:.1f}%)")

    print("\n다음 단계: python scripts/08_extract_features.py")


if __name__ == "__main__":
    main()
