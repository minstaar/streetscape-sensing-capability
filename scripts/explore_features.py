# -*- coding: utf-8 -*-
"""
explore_features.py — 타뷸러 피처 탐색 (파이프라인 외 탐색용)
─────────────────────────────────────────────────────────────────────────────
[목적]
    메인 파이프라인(09번) 개선 전, 어떤 피처가 유의미한지 확인
    store/ 및 길단위인구 데이터에서 추출한 신규 피처의 기여도를 검증

[테스트 피처 그룹]
    BASE    : 유동인구, 직장인구, 점포_수, 개업률, 폐업률, 상권유형, 면적
             (상주인구 제외: r=−0.04, CPI·기준금리 제외: 단면분산=0)
    STORE   : 프랜차이즈_비율, 업종_다양성(HHI), 경쟁_집중도, 식음료_비율
    POP     : 청년_비율(20-30대), 야간_비율(21-24시), 주말_비율(토·일), 여성_비율
    COMBINED: BASE + STORE(best) + POP(best)

[출력]
    reports/feature_exploration.txt  ← 조합별 성능 비교
    reports/lasso_coefficients.csv   ← 피처별 LASSO 계수

실행:
    python scripts/explore_features.py
"""

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
CROSS_CSV  = ROOT / "data/processed/cross_sectional_data.csv"
STORE_DIR  = ROOT / "data/raw/store"
POP_DIR    = ROOT / "data/raw/population"
REPORT_DIR = ROOT / "reports"

Y_COL        = "log_sales"
RANDOM_STATE = 42
CV_FOLDS     = 5

BASE_COLS = [
    "유동인구", "직장인구", "점포_수",
    "개업률", "폐업률", "상권_유형_더미", "면적_km2",
]

# 식음료 업종 키워드
FOOD_KEYWORDS = ["음식점", "분식", "호프", "주점", "커피", "음료",
                 "패스트푸드", "제과", "치킨", "도시락", "냉면"]

# 길단위인구 CSV 파일명 패턴
POP_FILE_PATTERN = "서울시 상권분석서비스(길단위인구-상권).csv"


# ══════════════════════════════════════════════════════════════════════════════
# store 데이터 집계 → 상권별 신규 피처 생성
# ══════════════════════════════════════════════════════════════════════════════
def load_store_features():
    """store/ 데이터 2019~2024년 평균 집계 → 상권별 피처 DataFrame"""
    frames = []
    for f in sorted(STORE_DIR.glob("*.csv")):
        if "2025" in f.name:   # 2025년은 영문 컬럼 → 스킵
            continue
        try:
            df = pd.read_csv(f, encoding="cp949")
            # 컬럼명 통일
            df.columns = df.columns.str.strip()
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ {f.name} 로드 실패: {e}")

    if not frames:
        return None

    raw = pd.concat(frames, ignore_index=True)
    raw["상권_코드"] = raw["상권_코드"].astype(str)

    # ── 상권별 집계 ──────────────────────────────────────────────────────────
    agg = (
        raw.groupby("상권_코드")
        .agg(
            총_점포수      = ("점포_수",           "sum"),
            총_프랜차이즈   = ("프랜차이즈_점포_수", "sum"),
            총_유사업종     = ("유사_업종_점포_수",  "sum"),
        )
        .reset_index()
    )

    # 프랜차이즈 비율
    agg["프랜차이즈_비율"] = agg["총_프랜차이즈"] / (agg["총_점포수"] + 1)

    # 경쟁 집중도 (유사업종 점포 / 전체 점포)
    agg["경쟁_집중도"] = agg["총_유사업종"] / (agg["총_점포수"] + 1)

    # 업종 다양성 HHI (1 - Σ비중²): 높을수록 다양
    업종_pivot = (
        raw.groupby(["상권_코드", "서비스_업종_코드"])["점포_수"]
        .sum()
        .unstack(fill_value=0)
    )
    proportions  = 업종_pivot.div(업종_pivot.sum(axis=1), axis=0)
    hhi          = (proportions ** 2).sum(axis=1)
    업종_다양성   = (1 - hhi).reset_index()
    업종_다양성.columns = ["상권_코드", "업종_다양성_HHI"]
    agg = agg.merge(업종_다양성, on="상권_코드", how="left")

    # 식음료 비율
    raw["is_food"] = raw["서비스_업종_코드_명"].apply(
        lambda x: any(k in str(x) for k in FOOD_KEYWORDS)
    )
    food_agg = (
        raw.groupby("상권_코드")
        .apply(lambda g: g.loc[g["is_food"], "점포_수"].sum() / (g["점포_수"].sum() + 1))
        .reset_index()
    )
    food_agg.columns = ["상권_코드", "식음료_비율"]
    agg = agg.merge(food_agg, on="상권_코드", how="left")

    store_cols = ["상권_코드", "프랜차이즈_비율", "경쟁_집중도", "업종_다양성_HHI", "식음료_비율"]
    print(f"  store 피처 생성 완료: {len(agg)}개 상권")
    return agg[store_cols]


# ══════════════════════════════════════════════════════════════════════════════
# 길단위인구 데이터 집계 → 상권별 인구 피처 생성
# ══════════════════════════════════════════════════════════════════════════════
def load_pop_features():
    """길단위인구 데이터 → 청년/야간/주말/여성 비율 피처 DataFrame"""
    pop_file = POP_DIR / POP_FILE_PATTERN
    if not pop_file.exists():
        # 패턴 매칭 시도
        candidates = list(POP_DIR.glob("*길단위인구*.csv"))
        if not candidates:
            print("  ⚠ 길단위인구 CSV 없음 — 인구 피처 건너뜀")
            return None
        pop_file = candidates[0]

    try:
        df = pd.read_csv(pop_file, encoding="cp949")
    except Exception as e:
        print(f"  ⚠ 길단위인구 로드 실패: {e}")
        return None

    df["상권_코드"] = df["상권_코드"].astype(str)
    df = df[df["총_유동인구_수"] > 0].copy()

    # 2019~2024 평균 집계 (2025 제외 — 부분 연도)
    df = df[df["기준_년분기_코드"] < 20250]

    agg = (
        df.groupby("상권_코드")
        .agg(
            총_유동인구      = ("총_유동인구_수",        "mean"),
            여성_유동인구    = ("여성_유동인구_수",       "mean"),
            연령대_20       = ("연령대_20_유동인구_수",   "mean"),
            연령대_30       = ("연령대_30_유동인구_수",   "mean"),
            시간대_21_24    = ("시간대_21_24_유동인구_수","mean"),
            토요일          = ("토요일_유동인구_수",       "mean"),
            일요일          = ("일요일_유동인구_수",       "mean"),
        )
        .reset_index()
    )

    tot = agg["총_유동인구"] + 1  # 0 나눔 방지

    # 파생 비율 피처
    agg["청년_비율"]  = (agg["연령대_20"] + agg["연령대_30"]) / tot   # 20-30대 비율
    agg["야간_비율"]  = agg["시간대_21_24"] / tot                     # 21-24시 비율
    agg["주말_비율"]  = (agg["토요일"] + agg["일요일"]) / (tot * 2)   # 주말 유동 비율
    agg["여성_비율"]  = agg["여성_유동인구"] / tot                     # 여성 비율

    pop_cols = ["상권_코드", "청년_비율", "야간_비율", "주말_비율", "여성_비율"]
    print(f"  인구 피처 생성 완료: {len(agg)}개 상권")
    return agg[pop_cols]


# ══════════════════════════════════════════════════════════════════════════════
# CV 평가
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(X, y, label, reg_name="LASSO"):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    if reg_name == "LASSO":
        reg = LassoCV(cv=CV_FOLDS, max_iter=10000, random_state=RANDOM_STATE)
    else:
        reg = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=CV_FOLDS)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2_list, rmse_list, mae_list = [], [], []

    for tr, te in kf.split(X_sc):
        sc2 = StandardScaler()
        Xtr = sc2.fit_transform(X[tr])
        Xte = sc2.transform(X[te])
        reg_f = (LassoCV(cv=CV_FOLDS, max_iter=10000, random_state=RANDOM_STATE)
                 if reg_name == "LASSO"
                 else RidgeCV(alphas=np.logspace(-3, 4, 50), cv=CV_FOLDS))
        reg_f.fit(Xtr, y[tr])
        yp = reg_f.predict(Xte)
        r2_list.append(r2_score(y[te], yp))
        rmse_list.append(np.sqrt(mean_squared_error(y[te], yp)))
        mae_list.append(mean_absolute_error(y[te], yp))

    n, p  = len(y), X.shape[1]
    r2    = np.mean(r2_list)
    adj   = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    rmse  = np.mean(rmse_list)
    mae   = np.mean(mae_list)

    print(f"  {label:<45} R²={r2:.4f}  AdjR²={adj:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"label": label, "r2": r2, "adj_r2": adj, "rmse": rmse, "mae": mae,
            "r2_std": np.std(r2_list)}


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("explore_features.py — 타뷸러 피처 탐색")
    print("=" * 70)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    cross = pd.read_csv(CROSS_CSV)
    cross["상권_코드"] = cross["상권_코드"].astype(str)

    # 분산=0 컬럼 제거
    base_cols = [c for c in BASE_COLS if c in cross.columns and cross[c].std() > 0]
    dropped   = set(BASE_COLS) - set(base_cols)
    if dropped:
        print(f"\n  분산=0 제거: {dropped}")

    print(f"\n  분석 대상: {len(cross)}개 상권")
    y = cross[Y_COL].values

    # ── store 피처 생성 ───────────────────────────────────────────────────────
    print("\n[1] store 피처 생성 중...")
    store_df = load_store_features()

    # ── 길단위인구 피처 생성 ──────────────────────────────────────────────────
    print("\n[1b] 길단위인구 피처 생성 중...")
    pop_df = load_pop_features()

    if store_df is not None:
        # cross_sectional_data.csv 에 이미 있는 컬럼은 store_df 에서 제외 (중복 방지)
        already_in_cross = [c for c in store_df.columns
                            if c != "상권_코드" and c in cross.columns]
        if already_in_cross:
            print(f"  ℹ cross_sectional_data에 이미 있는 컬럼 제외: {already_in_cross}")
        store_df_merge = store_df.drop(columns=already_in_cross)
        merged = cross.merge(store_df_merge, on="상권_코드", how="left")
        store_cols = [c for c in ["프랜차이즈_비율", "경쟁_집중도", "업종_다양성_HHI", "식음료_비율"]
                      if c in merged.columns
                      and merged[c].notna().sum() > len(merged) * 0.8]
        if store_cols:
            merged[store_cols] = merged[store_cols].fillna(merged[store_cols].median())
    else:
        # store_df 없어도 cross에 경쟁_집중도·식음료_비율이 있으면 사용
        merged     = cross.copy()
        store_cols = [c for c in ["경쟁_집중도", "식음료_비율"] if c in merged.columns]
        if not store_cols:
            print("  ⚠ store 피처 없음 — base 피처만 테스트")

    if pop_df is not None:
        merged = merged.merge(pop_df, on="상권_코드", how="left")
        pop_cols_valid = [c for c in ["청년_비율", "야간_비율", "주말_비율", "여성_비율"]
                         if merged[c].notna().sum() > len(merged) * 0.8]
        merged[pop_cols_valid] = merged[pop_cols_valid].fillna(merged[pop_cols_valid].median())
    else:
        pop_cols_valid = []
        print("  ⚠ 인구 피처 없음 — base+store 피처만 테스트")

    y_merged = merged[Y_COL].values

    # ── 피처 조합 테스트 ──────────────────────────────────────────────────────
    print("\n[2] 피처 조합별 성능 (LASSO, 5-fold CV)\n")
    results = []

    # 1. 베이스라인
    results.append(evaluate(
        merged[base_cols].values, y_merged,
        "BASE (현재 8개 피처)"
    ))

    # 2. store 피처 각각 추가
    for sc in store_cols:
        cols = base_cols + [sc]
        results.append(evaluate(
            merged[cols].values, y_merged,
            f"BASE + {sc}"
        ))

    # 3. store 피처 전체 추가
    if store_cols:
        all_cols = base_cols + store_cols
        results.append(evaluate(
            merged[all_cols].values, y_merged,
            f"BASE + ALL store ({len(store_cols)}개)"
        ))

    # 4. store 피처 2개 조합
    if len(store_cols) >= 2:
        print()
        for combo in combinations(store_cols, 2):
            cols = base_cols + list(combo)
            results.append(evaluate(
                merged[cols].values, y_merged,
                f"BASE + {' + '.join(combo)}"
            ))

    # 5. 인구 피처 각각 추가 (BASE 기준)
    if pop_cols_valid:
        print()
        for pc in pop_cols_valid:
            cols = base_cols + [pc]
            results.append(evaluate(
                merged[cols].values, y_merged,
                f"BASE + {pc}"
            ))

    # 6. 인구 피처 전체 추가
    if pop_cols_valid:
        all_pop = base_cols + pop_cols_valid
        results.append(evaluate(
            merged[all_pop].values, y_merged,
            f"BASE + ALL 인구 ({len(pop_cols_valid)}개)"
        ))

    # 7. store best + 인구 피처 조합 탐색
    # store best = 경쟁_집중도 + 식음료_비율 (이전 실험 결과)
    best_store = [c for c in ["경쟁_집중도", "식음료_비율"] if c in store_cols]
    if best_store and pop_cols_valid:
        print()
        # best store + 인구 각각
        for pc in pop_cols_valid:
            cols = base_cols + best_store + [pc]
            results.append(evaluate(
                merged[cols].values, y_merged,
                f"BASE + store_best + {pc}"
            ))
        # best store + 인구 전체
        cols = base_cols + best_store + pop_cols_valid
        results.append(evaluate(
            merged[cols].values, y_merged,
            f"BASE + store_best + ALL 인구"
        ))

    # 8. 전체 조합 (BASE + ALL store + ALL 인구)
    if store_cols and pop_cols_valid:
        all_feats = base_cols + store_cols + pop_cols_valid
        results.append(evaluate(
            merged[all_feats].values, y_merged,
            f"BASE + ALL store + ALL 인구 (전체 {len(all_feats)}개)"
        ))

    # ── 개별 피처 중요도 (전체 피처 LASSO 계수) ──────────────────────────────
    if store_cols or pop_cols_valid:
        print("\n[3] 전체 피처 LASSO 계수 (절댓값 기준 정렬)\n")
        all_cols = base_cols + store_cols + pop_cols_valid
        X_all    = merged[all_cols].values
        scaler   = StandardScaler()
        X_sc     = scaler.fit_transform(X_all)
        lasso    = LassoCV(cv=CV_FOLDS, max_iter=10000, random_state=RANDOM_STATE)
        lasso.fit(X_sc, y_merged)
        coef_df  = pd.DataFrame({
            "feature": all_cols,
            "coef":    lasso.coef_,
            "abs_coef": np.abs(lasso.coef_),
        }).sort_values("abs_coef", ascending=False)
        print(coef_df[["feature", "coef"]].to_string(index=False))
        coef_df.to_csv(REPORT_DIR / "lasso_coefficients.csv",
                       index=False, encoding="utf-8-sig")

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)
    res_df = pd.DataFrame(results).sort_values("r2", ascending=False)

    print(f"\n{'=' * 70}")
    print("결과 요약 (R² 내림차순)")
    print(f"{'=' * 70}")
    print(res_df[["label", "r2", "r2_std", "adj_r2", "rmse", "mae"]].to_string(index=False))

    with open(REPORT_DIR / "feature_exploration.txt", "w", encoding="utf-8") as f:
        f.write("피처 탐색 결과 (LASSO 5-fold CV)\n")
        f.write("=" * 70 + "\n")
        f.write(res_df[["label", "r2", "r2_std", "adj_r2", "rmse", "mae"]]
                .to_string(index=False))

    print(f"\n  저장 → {REPORT_DIR}/feature_exploration.txt")
    print(f"  저장 → {REPORT_DIR}/lasso_coefficients.csv")


if __name__ == "__main__":
    main()
