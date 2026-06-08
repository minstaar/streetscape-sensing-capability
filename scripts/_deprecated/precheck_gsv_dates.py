# -*- coding: utf-8 -*-
"""
precheck_gsv_dates.py — 이미지 수집 전 GSV 촬영 연도 사전 확인
─────────────────────────────────────────────────────────────────────────────
[목적]
    이미지 다운로드 없이 Metadata API만 호출해서
    서울 상권 지역의 GSV 촬영 연도 분포를 빠르게 확인.

    → 결과를 보고 05b_cross_sectional.py 의 CROSS_YEAR 를 확정한 뒤
      06_collect_images.py (본 수집) 을 실행할 것.

[비용]
    Metadata API 는 무료 (이미지 Static API 와 별개)
    100포인트 호출 = $0

[실행]
    python scripts/precheck_gsv_dates.py

[의존성]
    data/processed/final_sangkwon_list.csv  (01번 완료 후)
    data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp
    .env  →  GOOGLE_STREETVIEW_API_KEY
"""

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

API_KEY  = os.getenv("GOOGLE_STREETVIEW_API_KEY")
ROOT     = Path(__file__).resolve().parents[1]
LIST_CSV = ROOT / "data/processed/final_sangkwon_list.csv"
SHP_PATH = ROOT / "data/raw/boundaries/서울시 상권분석서비스(영역-상권).shp"
SV_META  = "https://maps.googleapis.com/maps/api/streetview/metadata"
SAMPLE_N = 100   # 확인할 포인트 수 (많을수록 정확, 시간은 ~2분)
RADIUS   = 50


def get_centroid_from_shp(codes):
    """상권 폴리곤 centroid 좌표 반환"""
    try:
        import geopandas as gpd
        from pyproj import Transformer
        gdf = gpd.read_file(SHP_PATH, encoding="cp949")
        gdf["상권_코드"] = gdf["TRDAR_CD"].astype(str)
        gdf = gdf[gdf["상권_코드"].isin(codes)]
        # TM → WGS84
        transformer = Transformer.from_crs("EPSG:5181", "EPSG:4326", always_xy=True)
        centroids = gdf.geometry.centroid
        lngs, lats = transformer.transform(centroids.x.values, centroids.y.values)
        return list(zip(lats, lngs, gdf["상권_코드"].values))
    except Exception as e:
        print(f"  ⚠ SHP 로드 실패: {e}")
        return []


def get_centroid_from_csv(codes):
    """SHP 없을 때: CSV 에서 좌표 컬럼 사용 (있으면)"""
    df = pd.read_csv(LIST_CSV)
    df["상권_코드"] = df["상권_코드"].astype(str)
    df = df[df["상권_코드"].isin(codes)]
    lat_col = next((c for c in df.columns if "위도" in c or "lat" in c.lower()), None)
    lng_col = next((c for c in df.columns if "경도" in c or "lng" in c.lower() or "lon" in c.lower()), None)
    if lat_col and lng_col:
        return list(zip(df[lat_col], df[lng_col], df["상권_코드"]))
    return []


def query_metadata(lat, lng):
    """GSV Metadata API → capture date 반환 ('YYYY-MM' or None)"""
    try:
        r = requests.get(
            SV_META,
            params={"location": f"{lat},{lng}", "radius": RADIUS, "key": API_KEY},
            timeout=10,
        )
        data = r.json()
        if data.get("status") == "OK":
            return data.get("date", "")   # 'YYYY-MM'
        return None   # 해당 위치에 GSV 없음
    except Exception:
        return None


def main():
    print("=" * 60)
    print("precheck_gsv_dates.py — GSV 촬영 연도 사전 확인")
    print("=" * 60)

    # ── 상권 코드 로드 ─────────────────────────────────────────
    df = pd.read_csv(LIST_CSV)
    df["상권_코드"] = df["상권_코드"].astype(str)
    all_codes = df["상권_코드"].tolist()
    sample_codes = pd.Series(all_codes).sample(
        min(SAMPLE_N, len(all_codes)), random_state=42
    ).tolist()
    print(f"  전체 상권: {len(all_codes)}개  →  샘플: {len(sample_codes)}개")

    # ── 좌표 확보 ──────────────────────────────────────────────
    print("\n  좌표 로드 중...")
    pts = get_centroid_from_shp(sample_codes)
    if not pts:
        pts = get_centroid_from_csv(sample_codes)
    if not pts:
        print("  ✗ 좌표를 가져올 수 없습니다.")
        print("    SHP 파일 또는 좌표 컬럼(위도/경도)이 필요합니다.")
        return
    print(f"  좌표 확보: {len(pts)}개")

    # ── Metadata API 호출 ─────────────────────────────────────
    print(f"\n  Metadata API 호출 중 (이미지 다운로드 없음, 무료)...")
    results = []
    no_sv   = 0
    for lat, lng, code in pts:
        date_str = query_metadata(lat, lng)
        if date_str is None:
            no_sv += 1
            continue
        results.append({
            "상권_코드"    : code,
            "capture_date" : date_str,
            "capture_year" : date_str[:4] if len(date_str) >= 4 else "unknown",
        })
        time.sleep(0.03)

    if not results:
        print("  ✗ 유효한 응답 없음. API 키 확인 필요.")
        return

    df_res = pd.DataFrame(results)
    valid  = df_res[df_res["capture_year"] != "unknown"]

    # ── 결과 출력 ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"결과 요약")
    print(f"  응답 성공 : {len(results)}개  |  GSV 없음: {no_sv}개")
    print()

    yr_counts = valid["capture_year"].value_counts().sort_index()
    total_v   = len(valid)
    print("  [연도별 분포]")
    for yr, cnt in yr_counts.items():
        bar = "█" * int(cnt / total_v * 30)
        print(f"    {yr}  {bar}  {cnt}개 ({cnt/total_v*100:.1f}%)")

    # 최빈 연도
    modal_year = yr_counts.idxmax()
    modal_pct  = yr_counts.max() / total_v * 100

    print(f"\n  → 최빈 촬영 연도: {modal_year} ({modal_pct:.1f}%)")
    print()

    # ── 권고 ──────────────────────────────────────────────────
    print("  [권고]")
    if modal_pct >= 70:
        print(f"  ✓ 이미지의 {modal_pct:.0f}%가 {modal_year}년 촬영본입니다.")
        print(f"    05b_cross_sectional.py 의 CROSS_YEAR = {modal_year} 으로 확정 권장")
    else:
        top2 = yr_counts.nlargest(2)
        print(f"  ⚠ 촬영 연도가 분산됩니다 (최빈 {modal_year}: {modal_pct:.0f}%).")
        print(f"    상위 2개 연도: {list(top2.index)}")
        print(f"    가중 평균 고려 또는 논문에 '이미지 수집 기간 YYYY~YYYY' 으로 기술 권장")

    print(f"\n  다음 단계: CROSS_YEAR 확정 → 06_collect_images.py 실행")
    print("=" * 60)


if __name__ == "__main__":
    main()
