# -*- coding: utf-8 -*-
"""
07b_filter_by_year.py — 연도 기준 이미지 필터링
─────────────────────────────────────────────────────────────────────────────
[목적]
    data/images_poi/ 의 이미지 중 KEEP_YEAR 이외 연도 파노라마를
    data/images_poi_rejected/ 로 이동 (삭제 아님 — 수동 복구 가능)

[결과]
    KEEP_YEAR 이미지만 남은 images_poi/ 폴더
    이동된 이미지는 images_poi_rejected/ 에 보관
    상권별 잔여 이미지 수 리포트 → reports/year_filter_report.csv

실행: python scripts/07b_filter_by_year.py
"""

import shutil
from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).resolve().parents[1]
IMG_DIR      = ROOT / "data/images_poi"
REJECTED_DIR = ROOT / "data/images_poi_rejected"
LOG_CSV      = ROOT / "data/processed/image_sampling_log_poi.csv"
REPORT_DIR   = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

KEEP_YEAR       = "2018"   # 이 연도만 유지
MIN_VALID_IMGS  = 10       # 이 미만이면 flagged 경고

# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"07b — 연도 필터링  (유지: {KEEP_YEAR}년)")
    print("=" * 60)

    if not IMG_DIR.exists():
        print(f"  ✗ {IMG_DIR} 없음")
        return

    # ── 로그에서 연도 정보 로드 ───────────────────────────────────────────────
    if not LOG_CSV.exists():
        print("  ✗ 수집 로그 없음. 파일명 기반으로는 연도 판별 불가.")
        print("  → image_sampling_log_poi.csv 가 필요합니다.")
        return

    log = pd.read_csv(LOG_CSV, encoding="utf-8-sig")
    log["상권_코드"] = log["상권_코드"].astype(str).str.strip()
    log = log[log["success"] == True].copy()

    # filename → (상권_코드, year) 매핑
    keep_set   = set()   # (code, filename) 유지할 것
    reject_set = set()   # (code, filename) 제거할 것

    for _, row in log.iterrows():
        code = row["상권_코드"]
        fname = row["filename"]
        year  = str(row.get("capture_year", "")).strip()
        if year == KEEP_YEAR:
            keep_set.add((code, fname))
        else:
            reject_set.add((code, fname))

    print(f"\n  유지 대상:  {len(keep_set):,}장 ({KEEP_YEAR}년)")
    print(f"  이동 대상:  {len(reject_set):,}장 (그 외 연도)")

    # ── 이동 실행 ──────────────────────────────────────────────────────────────
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    moved = 0
    not_found = 0

    for code, fname in reject_set:
        src = IMG_DIR / code / fname
        if not src.exists():
            not_found += 1
            continue
        dst_dir = REJECTED_DIR / code
        dst_dir.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst_dir / fname))
        moved += 1

    print(f"\n  이동 완료:  {moved:,}장 → {REJECTED_DIR}")
    if not_found:
        print(f"  파일 없음:  {not_found}건 (이미 처리됨)")

    # ── 상권별 잔여 이미지 수 확인 ─────────────────────────────────────────────
    records = []
    sq_dirs = [d for d in IMG_DIR.iterdir() if d.is_dir()]
    for sq_dir in sq_dirs:
        code   = sq_dir.name
        n_imgs = len(list(sq_dir.glob("*.jpg")))
        flagged = n_imgs < MIN_VALID_IMGS
        records.append({
            "상권_코드":    code,
            "이미지_수":    n_imgs,
            "flagged":    flagged,
        })

    df = pd.DataFrame(records).sort_values("이미지_수")
    df.to_csv(REPORT_DIR / "year_filter_report.csv",
              index=False, encoding="utf-8-sig")

    flagged = df[df["flagged"]]
    normal  = df[~df["flagged"]]

    print(f"\n{'=' * 60}")
    print("필터링 후 상권별 이미지 수 요약")
    print(f"  정상 ({MIN_VALID_IMGS}장 이상): {len(normal):,}개")
    print(f"  주의 ({MIN_VALID_IMGS}장 미만): {len(flagged):,}개")
    print(f"  평균 이미지 수: {df['이미지_수'].mean():.1f}장")
    print(f"  최소: {df['이미지_수'].min()}장  최대: {df['이미지_수'].max()}장")

    if len(flagged) > 0:
        print(f"\n  [주의 상권 — {MIN_VALID_IMGS}장 미만]")
        for _, r in flagged.head(20).iterrows():
            print(f"    상권_코드 {r['상권_코드']}  {r['이미지_수']}장")
        if len(flagged) > 20:
            print(f"    ... 외 {len(flagged)-20}개")

    print(f"\n  리포트 저장 → {REPORT_DIR}/year_filter_report.csv")
    print("=" * 60)
    print("\n다음 단계: python scripts/07c_restore_flagged.py (이미지 부족 상권 보완 + valid 리스트 생성)")


if __name__ == "__main__":
    main()
