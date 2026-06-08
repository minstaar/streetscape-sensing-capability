# -*- coding: utf-8 -*-
"""
09d_stratified_regression.py — 상권 유형별 분리 회귀
─────────────────────────────────────────────────────────────────────────────
[목적]
    발달상권 / 골목상권을 분리하여 각각 독립적으로 분석
    → 상권유형더미 문제 완전 해소
    → "골목상권 내 이미지 기여" / "발달상권 내 이미지 기여" 각각 검증

[설정]
    Y = log_매출_per_유동  (유동인구 대비 매출 전환율)
    골목상권: 10-fold CV  (n≈624, fold당 ~62개)
    발달상권:  5-fold CV  (n≈214, fold당 ~43개)

실행:
    python scripts/09d_stratified_regression.py
"""

import warnings, copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
CROSS_CSV = ROOT / "data/processed/cross_sectional_data.csv"
PANEL_CSV = ROOT / "data/processed/panel_final.csv"
VALID_CSV = ROOT / "data/processed/valid_image_sangkwon.csv"
DINO_CSV  = ROOT / "data/processed/image_features_dino_poi.csv"  # POI 2018 이미지
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

Y_COL        = "log_매출"
CROSS_YEAR   = "2019"
RANDOM_STATE = 42
MAX_PCA      = 100
CORR_THRESH  = 0.10

TABULAR_COLS = ["개업률", "폐업률", "면적_km2", "경쟁_집중도", "식음료_비율",
                "log_직장인구", "log_상주인구"]
# 자치구FE: 골목상권은 포함(n=598, 충분), 발달상권은 제외(n=194, 과적합 위험)
GU_FE_CONFIG = {
    "골목상권": True,
    "발달상권": False,
}

CV_CONFIG = {
    "골목상권": 10,
    "발달상권":  5,
}


# ══════════════════════════════════════════════════════════════════════════════
def run_cv(X_tab, X_img, y, cv_folds, use_img=False):
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    r2_list = []

    for train_idx, test_idx in kf.split(X_tab):
        X_tab_tr = X_tab[train_idx]
        X_tab_te = X_tab[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        sc = StandardScaler()
        X_tab_tr = sc.fit_transform(X_tab_tr)
        X_tab_te = sc.transform(X_tab_te)

        if use_img:
            img_sc = StandardScaler()
            X_img_sc_tr = img_sc.fit_transform(X_img[train_idx])
            X_img_sc_te = img_sc.transform(X_img[test_idx])

            n_comp = min(MAX_PCA, X_img_sc_tr.shape[1], X_img_sc_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
            pcs_tr = pca.fit_transform(X_img_sc_tr)
            pcs_te = pca.transform(X_img_sc_te)

            corrs = np.array([
                abs(np.corrcoef(pcs_tr[:, i], y_tr)[0, 1])
                for i in range(pcs_tr.shape[1])
            ])
            idx = np.where(corrs >= CORR_THRESH)[0]
            if len(idx) == 0:
                idx = np.argsort(corrs)[::-1][:5]

            X_tr = np.hstack([X_tab_tr, pcs_tr[:, idx]])
            X_te = np.hstack([X_tab_te, pcs_te[:, idx]])
        else:
            X_tr, X_te = X_tab_tr, X_tab_te

        reg = LassoCV(cv=min(5, cv_folds), max_iter=10000, random_state=RANDOM_STATE)
        reg.fit(X_tr, y_tr)
        r2_list.append(r2_score(y_te, reg.predict(X_te)))

    return np.array(r2_list)


def analyze_type(label, sub_df, X_img_full_df, cv_folds, all_rows):
    print(f"\n{'═'*65}")
    print(f"  {label}  (n={len(sub_df)}, {cv_folds}-fold CV)")
    print(f"{'═'*65}")

    y = sub_df[Y_COL].values

    # 자치구 FE 포함 여부 (상권유형별로 다르게 설정)
    use_gu = GU_FE_CONFIG.get(label, True)
    if use_gu and "자치구_코드_명" in sub_df.columns:
        gu = pd.get_dummies(sub_df["자치구_코드_명"], prefix="gu", drop_first=True)
        tab_cols_used = TABULAR_COLS + list(gu.columns)
        X_tab = np.hstack([sub_df[TABULAR_COLS].values, gu.values])
    else:
        tab_cols_used = TABULAR_COLS
        X_tab = sub_df[TABULAR_COLS].values

    # DINOv2 피처: sub_df와 동일한 순서로 align
    X_img = X_img_full_df.loc[sub_df.index].values

    print(f"  Y: mean={y.mean():.3f}  std={y.std():.3f}  "
          f"min={y.min():.3f}  max={y.max():.3f}")

    print(f"  tabular 변수: {len(tab_cols_used)}개 (자치구FE={'포함' if use_gu else '미포함'})")

    # tabular 변수 × Y 상관 (전체)
    print(f"\n  [Tabular-Y 상관]")
    for col in TABULAR_COLS:
        if col in sub_df.columns:
            r = np.corrcoef(sub_df[col].values, y)[0, 1]
            print(f"    {col:15s}  r={r:+.3f}")

    print(f"\n  [CV 결과]")
    r2_A = run_cv(X_tab, X_img, y, cv_folds, use_img=False)
    r2_B = run_cv(X_tab, X_img, y, cv_folds, use_img=True)

    t, p = ttest_rel(r2_B, r2_A)
    delta = r2_B.mean() - r2_A.mean()
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    print(f"    타뷸러만  R²={r2_A.mean():.4f}(±{r2_A.std():.4f})")
    print(f"    +DINOv2  R²={r2_B.mean():.4f}(±{r2_B.std():.4f})  "
          f"ΔR²={delta:+.4f}  t={t:+.2f}  p={p:.3f} {sig}")

    print(f"\n  [Fold별 ΔR²]")
    print(f"  fold  R²(tab)  R²(+img)   ΔR²")
    print(f"  {'─'*35}")
    for i, (ra, rb) in enumerate(zip(r2_A, r2_B), 1):
        flag = " ←" if ra < 0.05 else ""
        print(f"    {i:2d}   {ra:.4f}   {rb:.4f}   {rb-ra:+.4f}{flag}")
    print(f"  {'─'*35}")
    print(f"  avg  {r2_A.mean():.4f}   {r2_B.mean():.4f}   {delta:+.4f}")

    all_rows.append({
        "상권유형": label,
        "n": len(sub_df),
        "CV_folds": cv_folds,
        "R2_tabular": round(r2_A.mean(), 4),
        "R2_tabular_std": round(r2_A.std(), 4),
        "R2_image": round(r2_B.mean(), 4),
        "R2_image_std": round(r2_B.std(), 4),
        "delta_R2": round(delta, 4),
        "t_stat": round(t, 3),
        "p_value": round(p, 4),
        "significance": sig,
    })


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("09d_stratified_regression.py — 상권 유형별 분리 회귀")
    print("=" * 65)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    cross = pd.read_csv(CROSS_CSV)
    panel = pd.read_csv(PANEL_CSV)
    valid = pd.read_csv(VALID_CSV)
    dino  = pd.read_csv(DINO_CSV)

    for df in [cross, panel, valid, dino]:
        df["상권_코드"] = df["상권_코드"].astype(str).str.strip()

    p19 = (panel[panel["기준_년분기_코드"].astype(str).str.startswith(CROSS_YEAR)]
           .groupby("상권_코드")[["추정매출_합계", "유동인구", "직장인구", "상주인구"]].mean()
           .reset_index())
    p19[Y_COL]          = np.log1p(p19["추정매출_합계"] / p19["유동인구"])
    p19["log_직장인구"]  = np.log1p(p19["직장인구"])
    p19["log_상주인구"]  = np.log1p(p19["상주인구"])

    cross = cross.merge(
        p19[["상권_코드", Y_COL, "log_직장인구", "log_상주인구"]],
        on="상권_코드", how="left"
    )
    valid_codes = valid[valid["flagged"] == False]["상권_코드"]
    base = (cross[cross["상권_코드"].isin(valid_codes)]
            .dropna(subset=[Y_COL, "log_직장인구", "log_상주인구"])
            .reset_index(drop=True))

    # DINOv2 merge
    img_cols = [c for c in dino.columns if c.startswith("dino_")]
    merged = base.merge(dino[["상권_코드"] + img_cols], on="상권_코드", how="inner")
    merged = merged.reset_index(drop=True)

    # 이미지 피처 DataFrame (인덱스 정렬 보장)
    X_img_df = merged[img_cols].copy()

    print(f"\n  전체 표본: {len(merged)}개")
    print(f"  Y = {Y_COL}")
    print(f"  Tabular: {TABULAR_COLS}")

    all_rows = []

    # ── 유형별 분석 ───────────────────────────────────────────────────────────
    for type_name, cv_folds in CV_CONFIG.items():
        sub = merged[merged["상권_구분_코드_명"] == type_name].copy()
        analyze_type(type_name, sub, X_img_df, cv_folds, all_rows)

    # ── 요약 ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  종합 요약")
    print(f"{'═'*65}")
    print(f"\n  {'유형':8s}  {'n':>5s}  {'R²(tab)':>10s}  {'R²(+img)':>10s}  "
          f"{'ΔR²':>8s}  {'p':>8s}  {'sig':>5s}")
    print(f"  {'─'*65}")
    for r in all_rows:
        print(f"  {r['상권유형']:8s}  {r['n']:>5d}  "
              f"{r['R2_tabular']:>10.4f}  {r['R2_image']:>10.4f}  "
              f"{r['delta_R2']:>+8.4f}  {r['p_value']:>8.3f}  {r['significance']:>5s}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(
        REPORT_DIR / "stratified_results.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n  결과 저장 → {REPORT_DIR}/stratified_results.csv")
    print(f"{'='*65}")
    print("\n완료.")


if __name__ == "__main__":
    main()
