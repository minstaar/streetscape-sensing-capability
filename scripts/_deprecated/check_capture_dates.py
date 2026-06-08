# -*- coding: utf-8 -*-
"""
check_capture_dates.py — GSV 촬영 연월 사후 확인
─────────────────────────────────────────────────────────────────────────────
[목적]
    image_sampling_log.csv 의 좌표(lat/lng)로 Google Street View
    Metadata API를 재호출해 실제 촬영 연월(capture_date)을 확인

[출력]
    reports/capture_date_summary.csv  — 상권별 촬영 날짜 분포
    reports/capture_date_report.txt   — 요약 (연/월별 비율)

실행:
    python scripts/check_capture_dates.py

의존성:
    data/processed/image_sampling_log.csv  (06번 실행 후)
    .env  →  GOOGLE_STREETVIEW_API_KEY
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("GOOGLE_STREETVIEW_API_KEY")

ROOT       = Path(__file__).resolve().parents[1]
LOG_CSV    = ROOT / "data/processed/image_sampling_log.csv"
REPORT_DIR = ROOT / "reports"
OUT_CSV    = REPORT_DIR / "capture_date_summary.csv"
OUT_TXT    = REPORT_DIR / "capture_date_report.txt"

SV_META    = "https://maps.googleapis.com/maps/api/streetview/metadata"
SV_RADIUS  = 50
SAMPLE_N   = 300   # 전체를 다 호출하면 비용↑ → 대표 샘플만


def get_capture_date(lat, lng):
    try:
        r = requests.get(
            SV_META,
            params={"location": f"{lat},{lng}", "radius": SV_RADIUS, "key": API_KEY},
            timeout=10,
        )
        data = r.json()
        if data.get("status") != "OK":
            return ""
        return data.get("date", "")   # 'YYYY-MM'
    except Exception:
        return ""


def main():
    print("=" * 60)
    print("check_capture_dates.py — GSV 촬영 연월 확인")
    print("=" * 60)

    if not LOG_CSV.exists():
        print(f"  ⚠ {LOG_CSV.name} 없음 → 06_collect_images.py 먼저 실행")
        return

    log = pd.read_csv(LOG_CSV)
    log = log[log["success"] == True].copy()
    print(f"  수집 완료 이미지: {len(log):,}장")

    # 포인트(lat/lng) 기준으로 샘플링 (동일 좌표 중복 호출 방지)
    pts = log[["lat", "lng"]].drop_duplicates()
    if len(pts) > SAMPLE_N:
        pts = pts.sample(SAMPLE_N, random_state=42)
    print(f"  메타데이터 호출 포인트: {len(pts)}개 (중복 제거 후 샘플)")

    results = []
    for _, row in tqdm(pts.iterrows(), total=len(pts), desc="  API 호출"):
        date_str = get_capture_date(row["lat"], row["lng"])
        results.append({
            "lat"          : row["lat"],
            "lng"          : row["lng"],
            "capture_date" : date_str,
            "capture_year" : date_str[:4] if len(date_str) >= 4 else "",
            "capture_month": date_str[5:7] if len(date_str) >= 7 else "",
        })
        time.sleep(0.05)

    df = pd.DataFrame(results)
    valid = df[df["capture_date"] != ""]

    print(f"\n  응답 성공: {len(valid)} / {len(df)}")
    print(f"\n  [연도별 분포]")
    yr = valid["capture_year"].value_counts().sort_index()
    for y, cnt in yr.items():
        pct = cnt / len(valid) * 100
        print(f"    {y}: {cnt}개 ({pct:.1f}%)")

    print(f"\n  [연월별 분포 (상위 10)]")
    ym = valid["capture_date"].value_counts().head(10)
    for ym_key, cnt in ym.items():
        print(f"    {ym_key}: {cnt}개")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("GSV 촬영 연월 확인 보고서\n" + "=" * 50 + "\n\n")
        f.write(f"수집 이미지 총 {len(log):,}장 중 {len(pts)}포인트 샘플링\n")
        f.write(f"메타데이터 응답: {len(valid)} / {len(df)}\n\n")
        f.write("[연도별 분포]\n")
        for y, cnt in yr.items():
            f.write(f"  {y}: {cnt}개 ({cnt/len(valid)*100:.1f}%)\n")
        f.write("\n[연월별 분포]\n")
        for ym_key, cnt in valid["capture_date"].value_counts().items():
            f.write(f"  {ym_key}: {cnt}개\n")

    print(f"\n  저장 완료 → {OUT_CSV.name}, {OUT_TXT.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
