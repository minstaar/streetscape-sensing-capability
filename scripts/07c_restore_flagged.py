# -*- coding: utf-8 -*-
"""
07c_restore_flagged.py — 이미지 부족 상권 보완
─────────────────────────────────────────────────────────────────────────────
[목적]
    07b 필터링 후 이미지가 MIN_VALID_IMGS 미만인 상권에 대해
    images_poi_rejected/ 에서 RESTORE_YEAR_MIN 이상 연도 이미지를 복구.

    복구 우선순위: 최신 연도 우선 (2019 → 2017 → 2016 → 2015)
    2014년 이하 이미지는 복구하지 않음 (분석 기간과 격차 큼)

실행: python scripts/07c_restore_flagged.py
"""

import shutil
from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).resolve().parents[1]
IMG_DIR      = ROOT / "data/images_poi"
REJECTED_DIR = ROOT / "data/images_poi_rejected"
LOG_CSV      = ROOT / "data/processed/image_sampling_log_poi.csv"
REPORT_DIR   = ROOT / "reports"

MIN_VALID_IMGS  = 10     # 이 미만 상권을 보완 대상으로
RESTORE_YEAR_MIN = 2015  # 이 연도 이상만 복구 (2014 이하는 너무 오래됨)
TARGET_IMGS     = 15     # 보완 후 목표 이미지 수

# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print(f"07c — 부족 상권 이미지 보완")
    print(f"  복구 기준: {RESTORE_YEAR_MIN}년 이상, COVID 이전(≤2019)")
    print(f"  보완 목표: 상권당 {TARGET_IMGS}장")
    print("=" * 60)

    if not LOG_CSV.exists():
        print("  ✗ image_sampling_log_poi.csv 없음")
        return

    log = pd.read_csv(LOG_CSV, encoding="utf-8-sig")
    log["상권_코드"]    = log["상권_코드"].astype(str).str.strip()
    log["capture_year"] = log["capture_year"].astype(str).str.strip()
    log = log[log["success"] == True].copy()

    # ── 현재 이미지 수 파악 ───────────────────────────────────────────────────
    current_counts = {}
    for sq_dir in IMG_DIR.iterdir():
        if sq_dir.is_dir():
            current_counts[sq_dir.name] = len(list(sq_dir.glob("*.jpg")))

    flagged_codes = [c for c, n in current_counts.items() if n < MIN_VALID_IMGS]
    print(f"\n  보완 대상: {len(flagged_codes)}개 상권\n")

    # ── 복구 가능한 이미지 목록 (rejected에서) ────────────────────────────────
    # 연도 기준: RESTORE_YEAR_MIN 이상 & 2019 이하 (pre-COVID)
    restorable = log[
        log["상권_코드"].isin(flagged_codes) &
        (log["capture_year"].apply(
            lambda y: y.isdigit() and RESTORE_YEAR_MIN <= int(y) <= 2019
        ))
    ].copy()

    # 최신 연도 우선 정렬
    restorable = restorable.sort_values("capture_year", ascending=False)

    # ── 복구 실행 ──────────────────────────────────────────────────────────────
    restored_total = 0
    still_flagged  = []
    report_rows    = []

    for code in flagged_codes:
        current_n = current_counts.get(code, 0)
        need_n    = TARGET_IMGS - current_n

        candidates = restorable[restorable["상권_코드"] == code]

        restored = 0
        years_used = []
        for _, row in candidates.iterrows():
            if restored >= need_n:
                break
            src = REJECTED_DIR / code / row["filename"]
            if not src.exists():
                continue
            dst_dir = IMG_DIR / code
            dst_dir.mkdir(exist_ok=True)
            shutil.move(str(src), str(dst_dir / row["filename"]))
            restored += 1
            years_used.append(row["capture_year"])

        final_n = current_n + restored
        restored_total += restored

        if final_n < MIN_VALID_IMGS:
            still_flagged.append(code)

        yr_summary = ", ".join(sorted(set(years_used), reverse=True)) if years_used else "없음"
        print(f"  {code}: {current_n}장 → {final_n}장  (복구 {restored}장, 연도: {yr_summary})")
        report_rows.append({
            "상권_코드":     code,
            "before":       current_n,
            "restored":     restored,
            "after":        final_n,
            "years_used":   yr_summary,
            "still_flagged": final_n < MIN_VALID_IMGS,
        })

    # ── 요약 ──────────────────────────────────────────────────────────────────
    pd.DataFrame(report_rows).to_csv(
        REPORT_DIR / "restore_report.csv", index=False, encoding="utf-8-sig"
    )

    print(f"\n{'=' * 60}")
    print(f"  총 복구: {restored_total}장")
    print(f"  보완 성공: {len(flagged_codes) - len(still_flagged)}개 상권")
    if still_flagged:
        print(f"  여전히 부족 (<{MIN_VALID_IMGS}장): {len(still_flagged)}개 상권")
        print(f"  → 이 상권들은 분석에서 제외하거나 추가 수집 검토 필요")
        for c in still_flagged:
            n = current_counts.get(c, 0)
            r = next((r["restored"] for r in report_rows if r["상권_코드"] == c), 0)
            print(f"    {c}: {n+r}장")

    print(f"\n  리포트 → {REPORT_DIR}/restore_report.csv")

    # ── 최종 valid 리스트 산출 (downstream 08/08b/08c/10 입력) ─────────────────
    valid_rows = []
    for sq_dir in sorted(IMG_DIR.iterdir()):
        if not sq_dir.is_dir():
            continue
        n = len(list(sq_dir.glob("*.jpg")))
        valid_rows.append({"상권_코드": sq_dir.name, "valid_count": n,
                           "flagged": n < MIN_VALID_IMGS})
    vdf = pd.DataFrame(valid_rows).sort_values("valid_count").reset_index(drop=True)
    vpath = ROOT / "data/processed/valid_image_sangkwon.csv"
    vdf.to_csv(vpath, index=False, encoding="utf-8-sig")
    print(f"  최종 valid 리스트 → {vpath}  "
          f"(유효 {(~vdf['flagged']).sum()}/{len(vdf)}개, flagged {vdf['flagged'].sum()}개)")

    print("=" * 60)
    print("\n다음 단계: python scripts/08_extract_features.py "
          "→ 08b_extract_segmentation.py → 08c_extract_handcrafted.py → 10_capability_map.py")


if __name__ == "__main__":
    main()
