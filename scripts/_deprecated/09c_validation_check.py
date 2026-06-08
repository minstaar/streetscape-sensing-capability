# -*- coding: utf-8 -*-
"""
09c_validation_check.py — 결과 검증
─────────────────────────────────────────────────────────────────────────────
[검증 항목]
    1. 상권유형더미 포함/제외 × 이미지 유무 → 4가지 모델 비교
       - A:  더미❌ 이미지❌  (현재 baseline)
       - Ad: 더미✅ 이미지❌  (더미 포함 baseline)
       - B:  더미❌ 이미지✅  (현재 모델)
       - Bd: 더미✅ 이미지✅  (더미 포함 + 이미지)

       핵심 질문:
         ① 더미 제외해도 이미지가 유의한가? → B vs A
         ② 더미 포함해도 이미지가 유의한가? → Bd vs Ad  ← 리뷰어 반박 대응

    2. Y 분포 이상치 확인 (상·하위 1% 상권 목록)
    3. Fold별 R² 분포 (이상한 fold 있는지)

실행:
    python scripts/09c_validation_check.py
"""

import warnings
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
CROSS_CSV = ROOT / "data/processed/cross_sectional_data.csv"
PANEL_CSV = ROOT / "data/processed/panel_final.csv"
VALID_CSV = ROOT / "data/processed/valid_image_sangkwon.csv"
DINO_CSV  = ROOT / "data/processed/image_features_dino.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

Y_COL       = "log_매출_per_유동"
CROSS_YEAR  = "2019"
CV          = 10
RANDOM_STATE = 42
MAX_PCA     = 100
CORR_THRESH = 0.10


def run_cv(X_tab, X_img, y, use_img=False):
    kf = KFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
    r2_list = []
    for train_idx, test_idx in kf.split(X_tab):
        X_tab_tr = X_tab[train_idx]
        X_tab_te = X_tab[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        sc = StandardScaler()
        X_tab_tr = sc.fit_transform(X_tab_tr)
        X_tab_te = sc.transform(X_tab_te)

        if use_img and X_img is not None:
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

        reg = LassoCV(cv=5, max_iter=10000, random_state=RANDOM_STATE)
        reg.fit(X_tr, y_tr)
        r2_list.append(r2_score(y_te, reg.predict(X_te)))
    return np.array(r2_list)


def main():
    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    cross = pd.read_csv(CROSS_CSV)
    panel = pd.read_csv(PANEL_CSV)
    valid = pd.read_csv(VALID_CSV)
    dino  = pd.read_csv(DINO_CSV)

    for df in [cross, panel, valid, dino]:
        df["상권_코드"] = df["상권_코드"].astype(str).str.strip()

    p19 = (panel[panel["기준_년분기_코드"].astype(str).str.startswith(CROSS_YEAR)]
           .groupby("상권_코드")[["추정매출_합계", "유동인구"]].mean().reset_index())
    p19[Y_COL] = np.log1p(p19["추정매출_합계"] / p19["유동인구"])

    cross = cross.merge(p19[["상권_코드", Y_COL]], on="상권_코드", how="left")
    valid_codes = valid[valid["flagged"] == False]["상권_코드"]
    base = (cross[cross["상권_코드"].isin(valid_codes)]
            .dropna(subset=[Y_COL])
            .reset_index(drop=True))

    merged = base.merge(dino, on="상권_코드", how="inner")
    img_cols = [c for c in dino.columns if c.startswith("dino_")]
    X_img = merged[img_cols].values
    y     = merged[Y_COL].values

    print("=" * 65)
    print("09c_validation_check.py — 결과 검증")
    print("=" * 65)
    print(f"\n  표본: {len(merged)}개 상권")
    print(f"  Y = {Y_COL}  mean={y.mean():.3f}  std={y.std():.3f}")

    # ── [1] 이상치 확인 ───────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("[1] Y 분포 이상치 확인")
    print(f"{'─'*65}")
    df_s = merged[["상권_코드_명", "상권_구분_코드_명", Y_COL]].copy()
    print(f"\n  하위 5개 상권:")
    print(df_s.nsmallest(5, Y_COL).to_string(index=False))
    print(f"\n  상위 5개 상권:")
    print(df_s.nlargest(5, Y_COL).to_string(index=False))
    print(f"\n  상위/하위 1% 기준: {np.percentile(y,1):.3f} ~ {np.percentile(y,99):.3f}")
    print(f"  1% 밖 상권 수: {(y < np.percentile(y,1)).sum() + (y > np.percentile(y,99)).sum()}개")

    # ── [2] 핵심 검증: 더미 포함/제외 × 이미지 유무 ─────────────────────────
    print(f"\n{'─'*65}")
    print("[2] 상권유형더미 포함/제외 × 이미지 유무  (10-fold CV, LASSO)")
    print(f"{'─'*65}")

    TAB_NO_DUMMY   = ["개업률", "폐업률", "면적_km2", "경쟁_집중도", "식음료_비율"]
    TAB_WITH_DUMMY = ["개업률", "폐업률", "면적_km2", "경쟁_집중도", "식음료_비율", "상권_유형_더미"]

    X_A  = merged[TAB_NO_DUMMY].values
    X_Ad = merged[TAB_WITH_DUMMY].values

    print("\n  계산 중...")
    r2_A  = run_cv(X_A,  X_img, y, use_img=False)
    r2_Ad = run_cv(X_Ad, X_img, y, use_img=False)
    r2_B  = run_cv(X_A,  X_img, y, use_img=True)
    r2_Bd = run_cv(X_Ad, X_img, y, use_img=True)

    def show(label, r2_arr, r2_base=None):
        m, s = r2_arr.mean(), r2_arr.std()
        if r2_base is not None:
            t, p = ttest_rel(r2_arr, r2_base)
            delta = m - r2_base.mean()
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {label:<38} R²={m:.4f}(±{s:.4f})  ΔR²={delta:+.4f}  t={t:+.2f}  p={p:.3f} {sig}")
        else:
            print(f"  {label:<38} R²={m:.4f}(±{s:.4f})")

    print()
    show("A:  더미❌ 이미지❌  [현재 baseline]", r2_A)
    show("Ad: 더미✅ 이미지❌  [강화 baseline]", r2_Ad)
    show("B:  더미❌ 이미지✅  [현재 모델]",     r2_B,  r2_A)
    show("Bd: 더미✅ 이미지✅  [리뷰어 대응]",   r2_Bd, r2_Ad)

    print(f"\n  해석 기준:")
    print(f"    B vs A  → 이미지 기여 (더미 제외, 현재 주장)")
    print(f"    Bd vs Ad → 더미 포함해도 이미지 추가 기여? (리뷰어 반박 대응)")

    # ── [3] Fold별 R² 분포 확인 ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("[3] Fold별 R² 분포 (현재 모델 B: 더미❌ 이미지✅)")
    print(f"{'─'*65}")
    print(f"\n  fold  R²(tabular)  R²(+image)  ΔR²")
    print(f"  {'─'*40}")
    for i, (ra, rb) in enumerate(zip(r2_A, r2_B), 1):
        flag = " ← 주의" if rb < 0 or abs(rb - ra) > 0.4 else ""
        print(f"    {i:2d}    {ra:.4f}       {rb:.4f}    {rb-ra:+.4f}{flag}")
    print(f"  {'─'*40}")
    print(f"  avg   {r2_A.mean():.4f}       {r2_B.mean():.4f}    {r2_B.mean()-r2_A.mean():+.4f}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    results = {
        "model": ["A(더미X이미지X)", "Ad(더미O이미지X)", "B(더미X이미지O)", "Bd(더미O이미지O)"],
        "R2_mean": [r2_A.mean(), r2_Ad.mean(), r2_B.mean(), r2_Bd.mean()],
        "R2_std":  [r2_A.std(),  r2_Ad.std(),  r2_B.std(),  r2_Bd.std()],
    }
    pd.DataFrame(results).to_csv(REPORT_DIR / "validation_check.csv", index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장 → {REPORT_DIR}/validation_check.csv")
    print(f"\n{'='*65}")


if __name__ == "__main__":
    main()
