# -*- coding: utf-8 -*-
"""
05b_cross_sectional.py — 2019년 기준 단면 데이터 구축
─────────────────────────────────────────────────────────────────────────────
[목적]
    이미지(2018년 빈티지)와 가장 가까운 매출패널 시작연도(2019)로 타뷸러 데이터 정렬.
    GSV 기반 선행연구 표준: "이미지 수집 연도 ≈ 타뷸러 데이터 연도".

[분석 연도]
    CROSS_YEAR = 2019  →  분기코드 20191 ~ 20194 (4개 분기 평균)

[Y 변수]
    log_sales = log1p( 분기별(추정매출_합계 / 점포_수) 의 연평균 )
                 ← 점포당 생산성(상권 경쟁력) 측정
                 ← 상권 규모(점포 수) 차이 제거

[X 변수 — 최종 피처셋 (9개)]
    수요       : 유동인구, 직장인구
    공급/규모  : 점포_수, 개업률, 폐업률
    상권 특성  : 상권_유형_더미, 면적_km2
    경쟁 구조  : 경쟁_집중도 (store/ CROSS_YEAR 데이터)
    업종 구성  : 식음료_비율 (store/ CROSS_YEAR 데이터)

[제외 이유]
    상주인구  : r=−0.04 (매출과 무관), 입지론 관점에서도 상업 매출의 직접
                수요와 거리가 멀어 제외 (Lee et al. 2020 참조)
    CPI·기준금리: 단면 분석에서 모든 상권이 동일 값 → 분산=0 → 설명력 없음

[출력]
    data/processed/cross_sectional_data.csv  ← 10_capability_map.py 입력

실행:
    python scripts/05b_cross_sectional.py

의존성:
    data/processed/panel_final.csv
    data/raw/store/서울시 상권분석서비스(점포-상권)_2019년.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
PANEL_CSV  = ROOT / "data/processed/panel_final.csv"
STORE_DIR  = ROOT / "data/raw/store"
OUT_CSV    = ROOT / "data/processed/cross_sectional_data.csv"

# ── 분석 연도 설정 ─────────────────────────────────────────────────────────────
CROSS_YEAR   = 2019
Q_START      = CROSS_YEAR * 10 + 1   # 20191
Q_END        = CROSS_YEAR * 10 + 4   # 20194

# 식음료 업종 키워드 (explore_features.py 와 동일)
FOOD_KEYWORDS = ["음식점", "분식", "호프", "주점", "커피", "음료",
                 "패스트푸드", "제과", "치킨", "도시락", "냉면"]


# ══════════════════════════════════════════════════════════════════════════════
# store/ 2024 데이터 → 경쟁_집중도 · 식음료_비율
# ══════════════════════════════════════════════════════════════════════════════
def load_store_features_2024() -> pd.DataFrame:
    """
    CROSS_YEAR 점포 데이터 → 상권별 경쟁_집중도 · 식음료_비율
    파일명 패턴: 서울시_상권분석서비스(점포-상권)_{CROSS_YEAR}년.csv
                 서울시 상권분석서비스(점포-상권)_{CROSS_YEAR}년.csv  (공백 버전)
    """
    # 언더스코어/공백 두 패턴 모두 시도
    candidates = [
        STORE_DIR / f"서울시_상권분석서비스(점포-상권)_{CROSS_YEAR}년.csv",
        STORE_DIR / f"서울시 상권분석서비스(점포-상권)_{CROSS_YEAR}년.csv",
    ]
    target_file = next((f for f in candidates if f.exists()), None)
    if target_file is None:
        print(f"  ⚠ {CROSS_YEAR}년 점포 파일 없음 (탐색 위치: {STORE_DIR})")
        return None

    df = pd.read_csv(target_file, encoding="cp949")
    df.columns = df.columns.str.strip()
    df["상권_코드"] = df["상권_코드"].astype(str)

    # 2024년 4개 분기만 사용
    df = df[
        (df["기준_년분기_코드"] >= Q_START) &
        (df["기준_년분기_코드"] <= Q_END)
    ].copy()

    if len(df) == 0:
        print("  ⚠ 2024년 점포 데이터 없음")
        return None

    # 상권별 연간 합산
    agg = (
        df.groupby("상권_코드")
        .agg(
            총_점포수      = ("점포_수",            "sum"),
            총_유사업종    = ("유사_업종_점포_수",   "sum"),
        )
        .reset_index()
    )

    # 경쟁_집중도 = 유사업종 점포 / 전체 점포
    agg["경쟁_집중도"] = agg["총_유사업종"] / (agg["총_점포수"] + 1)

    # 식음료_비율 — 업종명 기반
    df["is_food"] = df["서비스_업종_코드_명"].apply(
        lambda x: any(k in str(x) for k in FOOD_KEYWORDS)
    )
    food_agg = (
        df.groupby("상권_코드")
        .apply(
            lambda g: g.loc[g["is_food"], "점포_수"].sum()
                      / (g["점포_수"].sum() + 1)
        )
        .reset_index()
    )
    food_agg.columns = ["상권_코드", "식음료_비율"]
    agg = agg.merge(food_agg, on="상권_코드", how="left")

    print(f"  store 피처 생성 완료 ({CROSS_YEAR}년): {len(agg)}개 상권")
    return agg[["상권_코드", "경쟁_집중도", "식음료_비율"]]


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print(f"05b_cross_sectional.py — {CROSS_YEAR}년 단면 데이터 구축")
    print("=" * 65)
    print(f"\n  분석 분기: {Q_START} ~ {Q_END} (4개 분기 평균)")

    # ── 1. 패널 데이터 로드 및 2024년 필터링 ──────────────────────────────────
    print("\n[1] panel_final.csv 로드 중...")
    panel = pd.read_csv(PANEL_CSV)
    panel["상권_코드"] = panel["상권_코드"].astype(str)

    df24 = panel[
        (panel["기준_년분기_코드"] >= Q_START) &
        (panel["기준_년분기_코드"] <= Q_END)
    ].copy()

    n_dist = df24["상권_코드"].nunique()
    n_q    = df24["기준_년분기_코드"].nunique()
    print(f"  2024년 데이터: {n_dist}개 상권 × {n_q}개 분기 = {len(df24):,}행")

    if n_q < 4:
        print(f"  ⚠ 분기가 {n_q}개뿐입니다 (기대: 4개). 진행하지만 확인 필요.")

    # ── 2. Y 계산 ────────────────────────────────────────────────────────────
    print("\n[2] Y 변수 계산...")
    df24 = df24[df24["점포_수"] > 0].copy()

    # Y1: log(매출/점포_수) — 점포당 생산성 (기존)
    df24["sales_per_store"]     = df24["추정매출_합계"] / df24["점포_수"]
    df24["log_sales_per_store"] = np.log1p(df24["sales_per_store"])

    # Y2: log(추정매출_합계) — 상권 총매출 (신규)
    df24["log_sales_total"] = np.log1p(df24["추정매출_합계"])

    y_agg = (
        df24.groupby("상권_코드")[["log_sales_per_store", "log_sales_total"]]
        .mean()
        .rename(columns={"log_sales_per_store": "log_sales",
                          "log_sales_total":     "log_sales_total"})
        .reset_index()
    )
    print(f"  Y 범위: {y_agg['log_sales'].min():.3f} ~ {y_agg['log_sales'].max():.3f}")
    print(f"  Y 평균: {y_agg['log_sales'].mean():.3f}  std: {y_agg['log_sales'].std():.3f}")

    # ── 3. X 수치 변수 — 2024년 4개 분기 평균 ─────────────────────────────────
    print("\n[3] X 수치 변수 연평균 계산...")
    # 상주인구 제외 (r=−0.04, 설명력 없음)
    num_cols = ["유동인구", "직장인구", "점포_수", "개업률", "폐업률"]
    x_agg = (
        df24.groupby("상권_코드")[num_cols]
        .mean()
        .reset_index()
    )

    # ── 4. 메타 정보 (상권별 고정값) ───────────────────────────────────────────
    meta_cols = ["상권_코드", "상권_코드_명", "상권_구분_코드_명",
                 "자치구_코드", "자치구_코드_명", "면적_km2"]
    available_meta = [c for c in meta_cols if c in df24.columns]
    meta = df24.groupby("상권_코드")[available_meta[1:]].first().reset_index()

    # 상권_유형_더미 재생성 (발달상권=1, 골목상권=0)
    meta["상권_유형_더미"] = (meta["상권_구분_코드_명"] == "발달상권").astype(int)

    # 면적_km2 가 없으면 영역_면적에서 변환
    if "면적_km2" not in meta.columns and "영역_면적" in df24.columns:
        area = df24.groupby("상권_코드")["영역_면적"].first().reset_index()
        area["면적_km2"] = (area["영역_면적"] / 1_000_000).round(6)
        meta = meta.merge(area[["상권_코드", "면적_km2"]], on="상권_코드", how="left")

    # ── 5. store 피처 (경쟁_집중도, 식음료_비율) ──────────────────────────────
    print("\n[4] store 피처 로드 중...")
    store_df = load_store_features_2024()

    # ── 6. 전체 병합 ──────────────────────────────────────────────────────────
    print("\n[5] 병합 중...")
    cs = y_agg.merge(x_agg, on="상권_코드", how="inner")
    cs = cs.merge(meta,     on="상권_코드", how="left")

    if store_df is not None:
        cs = cs.merge(store_df, on="상권_코드", how="left")
        # 결측 상권은 중앙값으로 대체
        for col in ["경쟁_집중도", "식음료_비율"]:
            miss = cs[col].isna().sum()
            if miss > 0:
                cs[col] = cs[col].fillna(cs[col].median())
                print(f"    {col} 결측 {miss}개 → 중앙값 대체")

    # ── 7. 결측값 처리 및 최종 정리 ───────────────────────────────────────────
    before = len(cs)
    cs = cs.dropna(subset=["log_sales"] + num_cols)
    after  = len(cs)
    if before != after:
        print(f"  ⚠ 결측 제거: {before} → {after}개 상권")

    # 컬럼 순서 정리
    base_order = [
        "상권_코드", "상권_코드_명", "상권_구분_코드_명",
        "상권_유형_더미", "자치구_코드", "자치구_코드_명",
        "면적_km2",
        "log_sales",           # Y1: 점포당 매출 (기존)
        "log_sales_total",     # Y2: 총매출 (신규)
        "유동인구", "직장인구", # 수요
        "점포_수", "개업률", "폐업률",  # 공급·역동성
    ]
    store_order = []
    if store_df is not None:
        store_order = ["경쟁_집중도", "식음료_비율"]

    final_cols = [c for c in base_order + store_order if c in cs.columns]
    cs = cs[final_cols].reset_index(drop=True)

    # ── 8. 결과 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"최종 단면 데이터")
    print(f"  분석 연도  : {CROSS_YEAR}년 (4개 분기 평균)")
    print(f"  상권 수    : {len(cs):,}개")
    print(f"  피처 수    : {len(final_cols) - 7}개 (메타 제외)")  # 7 = 코드·명칭 컬럼
    print(f"  Y 컬럼     : log_sales = log1p(추정매출/점포_수)")
    print()
    print("  [제외된 변수]")
    print("    상주인구  : 매출과 무관 (r=−0.04), 상업 직접수요 아님")
    print("    CPI·기준금리: 단면에서 분산=0 (모든 상권 동일)")
    print()

    feat_cols = [c for c in final_cols if c not in
                 ["상권_코드","상권_코드_명","상권_구분_코드_명",
                  "상권_유형_더미","자치구_코드","자치구_코드_명","면적_km2","log_sales"]]

    print("  [포함된 피처 상관계수 with log_sales]")
    for col in ["상권_유형_더미", "면적_km2"] + feat_cols:
        if col in cs.columns:
            r = cs[col].corr(cs["log_sales"])
            print(f"    {col:<20} r = {r:+.4f}")
    print(f"{'=' * 65}")

    # ── 9. 저장 ───────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cs.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  저장 완료 → {OUT_CSV}")
    print("\n  다음 단계: python scripts/10_capability_map.py (구 09 계열은 폐기됨)")


if __name__ == "__main__":
    main()
