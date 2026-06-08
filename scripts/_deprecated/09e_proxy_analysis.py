# -*- coding: utf-8 -*-
"""
09e_proxy_analysis.py — 이미지의 인구통계 데이터 대체 능력 분석
─────────────────────────────────────────────────────────────────────────────
[연구 질문]
    GSV 이미지(DINOv2)는 어떤 인구통계 변수를 대리할 수 있는가?
    데이터 가용성 시나리오별로 이미지의 실용적 대체 가치는 얼마인가?

[분석 구조]
    PART 1. 역방향 예측: 이미지 → 타뷸러/인구 변수 예측
             DINOv2 PC가 직장인구, 상주인구, 상권유형을 얼마나 예측하나?

    PART 2. 데이터 시나리오별 Y 예측력 비교
             A: 타뷸러 전체                  (benchmark)
             B: 타뷸러 전체 + 이미지          (upper bound)
             C: 이미지만                      (완전 데이터 빈곤)
             D: 공간정보만 + 이미지            (쉽게 구할 수 있는 것)
             E: 타뷸러(인구제외) + 이미지      (인구통계 없을 때)
             F: 타뷸러 전체(인구포함) + 이미지 (모든 데이터)

    PART 3. 분산 분해: 이미지-타뷸러 정보 중첩 구조

실행: python scripts/09e_proxy_analysis.py
"""

import warnings, copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
CROSS_CSV  = ROOT / "data/processed/cross_sectional_data.csv"
PANEL_CSV  = ROOT / "data/processed/panel_final.csv"
VALID_CSV  = ROOT / "data/processed/valid_image_sangkwon.csv"
DINO_CSV   = ROOT / "data/processed/image_features_dino.csv"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────────────────────
Y_COL        = "log_매출_per_유동"
CROSS_YEAR   = "2019"
CV           = 10
RANDOM_STATE = 42
MAX_PCA      = 100
CORR_THRESH  = 0.10

# 타뷸러 변수 그룹 정의
TAB_BUSINESS = ["개업률", "폐업률", "면적_km2", "경쟁_집중도", "식음료_비율"]  # 쉽게 구할 수 있는 상권 구조 변수
TAB_DEMO     = ["log_직장인구", "log_상주인구"]                                # 인구통계 (비용 높음)
TAB_GU_FE    = None  # 자치구 FE: 아래서 동적 생성


# ══════════════════════════════════════════════════════════════════════════════
def make_gu_dummies(df):
    gu = pd.get_dummies(df["자치구_코드_명"], prefix="gu", drop_first=True)
    return gu, list(gu.columns)


def cv_r2(X, y, cv=CV, rs=RANDOM_STATE):
    """단순 Ridge CV R² (변수 선택 없음 — 역방향 예측용)"""
    kf = KFold(n_splits=cv, shuffle=True, random_state=rs)
    r2s = []
    for tr, te in kf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        reg = Ridge(alpha=1.0)
        reg.fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], reg.predict(Xte)))
    return float(np.mean(r2s)), float(np.std(r2s))


def cv_r2_with_img(X_tab, X_img, y, top_k=10, cv=CV, rs=RANDOM_STATE):
    """
    타뷸러 + 이미지 PC (leakage-free) → CV R²
    ★ PC 선택(|r|>thresh 또는 Top-k)은 train fold 내에서만 수행
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=rs)
    r2s = []
    for tr, te in kf.split(X_img):
        # 타뷸러 전처리
        sc_tab = StandardScaler()
        if X_tab is not None and X_tab.shape[1] > 0:
            Xt_tr = sc_tab.fit_transform(X_tab[tr])
            Xt_te = sc_tab.transform(X_tab[te])
        else:
            Xt_tr = np.zeros((len(tr), 0))
            Xt_te = np.zeros((len(te), 0))

        # 이미지 PCA (train only)
        sc_img = StandardScaler()
        Xi_tr  = sc_img.fit_transform(X_img[tr])
        Xi_te  = sc_img.transform(X_img[te])
        nc     = min(MAX_PCA, Xi_tr.shape[1], Xi_tr.shape[0] - 1)
        pca    = PCA(n_components=nc, random_state=rs)
        pc_tr  = pca.fit_transform(Xi_tr)
        pc_te  = pca.transform(Xi_te)

        # PC 선택 (train y 기준 — leakage-free)
        corrs = np.array([abs(np.corrcoef(pc_tr[:, i], y[tr])[0, 1])
                          for i in range(pc_tr.shape[1])])
        if top_k is not None:
            idx = np.argsort(corrs)[::-1][:top_k]
        else:
            idx = np.where(corrs >= CORR_THRESH)[0]
            if len(idx) == 0:
                idx = np.argsort(corrs)[::-1][:5]

        # 결합
        Xtr = np.hstack([Xt_tr, pc_tr[:, idx]]) if Xt_tr.shape[1] > 0 else pc_tr[:, idx]
        Xte = np.hstack([Xt_te, pc_te[:, idx]]) if Xt_te.shape[1] > 0 else pc_te[:, idx]

        reg = LassoCV(cv=5, max_iter=10000, random_state=rs)
        reg.fit(Xtr, y[tr])
        r2s.append(r2_score(y[te], reg.predict(Xte)))

    return float(np.mean(r2s)), float(np.std(r2s))


def print_row(label, r2, std, r2_bench=None):
    delta = ""
    if r2_bench is not None:
        d = r2 - r2_bench
        delta = f"  ΔR²={d:+.4f}"
    print(f"  {label:<52}  R²={r2:.4f}(±{std:.4f}){delta}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("09e — 이미지의 인구통계 데이터 대체 능력 분석")
    print("=" * 72)

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
    p19[Y_COL]           = np.log1p(p19["추정매출_합계"] / p19["유동인구"])
    p19["log_직장인구"]   = np.log1p(p19["직장인구"])
    p19["log_상주인구"]   = np.log1p(p19["상주인구"])

    cross = cross.merge(p19[["상권_코드", Y_COL, "log_직장인구", "log_상주인구"]],
                        on="상권_코드", how="left")
    valid_codes = valid[valid["flagged"] == False]["상권_코드"]
    base = (cross[cross["상권_코드"].isin(valid_codes)]
            .dropna(subset=[Y_COL, "log_직장인구", "log_상주인구"])
            .reset_index(drop=True))

    # 자치구 FE
    gu_df, gu_cols = make_gu_dummies(base)
    base = pd.concat([base, gu_df], axis=1)

    # DINOv2 merge
    dino_cols = [c for c in dino.columns if c.startswith("dino_")]
    merged = base.merge(dino[["상권_코드"] + dino_cols], on="상권_코드", how="inner").reset_index(drop=True)

    y      = merged[Y_COL].values
    X_img  = merged[dino_cols].values
    print(f"  표본: {len(merged)}개  |  DINOv2: {len(dino_cols)}dim")
    print(f"  Y mean={y.mean():.3f}  std={y.std():.3f}\n")

    # ── PART 1. 역방향 예측: 이미지 → 인구/유형 변수 ────────────────────────
    print("=" * 72)
    print("PART 1. 역방향 예측 — 이미지(DINOv2 Top-20 PC)가 각 변수를 얼마나 설명하나")
    print("  (10-fold CV Ridge  |  ★ leakage-free)")
    print("=" * 72)

    # 이미지 PCA (전체 기준 — 역방향 예측은 PC를 고정해서 사용)
    sc_full = StandardScaler()
    Xi_full = sc_full.fit_transform(X_img)
    pca_full = PCA(n_components=min(MAX_PCA, X_img.shape[1]), random_state=RANDOM_STATE)
    PC_full  = pca_full.fit_transform(Xi_full)[:, :20]

    proxy_targets = {
        "log_직장인구":  merged["log_직장인구"].values,
        "log_상주인구":  merged["log_상주인구"].values,
        "상권유형(골목=0/발달=1)": merged["상권_유형_더미"].values.astype(float),
        Y_COL + " (참고용)": y,
    }

    part1_rows = []
    for name, target in proxy_targets.items():
        r2m, r2s = cv_r2(PC_full, target)
        print(f"  이미지 → {name:<28}  R²={r2m:.4f}(±{r2s:.4f})")
        part1_rows.append({"target": name, "R2_mean": round(r2m, 4), "R2_std": round(r2s, 4)})

    # ── PART 2. 데이터 시나리오별 Y 예측력 ──────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PART 2. 데이터 가용성 시나리오별 Y 예측력")
    print(f"  Y = {Y_COL}  |  10-fold CV LASSO  |  ★ leakage-free")
    print("=" * 72)

    scenarios = {
        "A. 타뷸러 전체(인구 포함)":
            (merged[TAB_BUSINESS + TAB_DEMO + gu_cols].values, None),
        "B. 타뷸러 전체 + 이미지 [상한선]":
            (merged[TAB_BUSINESS + TAB_DEMO + gu_cols].values, X_img),
        "C. 이미지만 [완전 데이터 빈곤]":
            (None, X_img),
        "D. 공간정보(자치구FE+면적) + 이미지":
            (merged[["면적_km2"] + gu_cols].values, X_img),
        "E. 상권구조변수 + 이미지 (인구 없음)":
            (merged[TAB_BUSINESS + gu_cols].values, X_img),
        "F. 상권구조 + 인구통계 + 이미지 [풀셋]":
            (merged[TAB_BUSINESS + TAB_DEMO + gu_cols].values, X_img),
    }

    bench_r2 = None
    part2_rows = []
    for label, (X_tab, X_im) in scenarios.items():
        if X_im is None:
            # 타뷸러만
            r2m, r2s = cv_r2(X_tab, y)
        else:
            r2m, r2s = cv_r2_with_img(
                X_tab if X_tab is not None else np.zeros((len(y), 0)),
                X_im, y, top_k=10
            )
        if label.startswith("A"):
            bench_r2 = r2m
        print_row(label, r2m, r2s, bench_r2 if not label.startswith("A") else None)
        part2_rows.append({"scenario": label, "R2_mean": round(r2m, 4),
                           "R2_std": round(r2s, 4),
                           "delta_vs_A": round(r2m - bench_r2, 4) if bench_r2 else 0})

    # ── PART 3. 분산 분해 ────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PART 3. 분산 분해 — 이미지-타뷸러 정보 중첩 구조")
    print("=" * 72)

    # 이미지만: C
    r2_img_only, _ = cv_r2_with_img(
        np.zeros((len(y), 0)), X_img, y, top_k=10)

    # 타뷸러만 (인구 포함): A
    r2_tab_only, _ = cv_r2(
        merged[TAB_BUSINESS + TAB_DEMO + gu_cols].values, y)

    # 풀셋: F
    r2_full, _ = cv_r2_with_img(
        merged[TAB_BUSINESS + TAB_DEMO + gu_cols].values, X_img, y, top_k=10)

    # 분산 분해 (근사)
    shared     = r2_img_only + r2_tab_only - r2_full   # 중첩 분산
    img_unique = r2_img_only - shared                   # 이미지 고유 기여
    tab_unique = r2_tab_only - shared                   # 타뷸러 고유 기여
    unexplained = 1.0 - r2_full

    print(f"\n  타뷸러만 R²       = {r2_tab_only:.4f}")
    print(f"  이미지만 R²       = {r2_img_only:.4f}  "
          f"→ 타뷸러 정보의 {r2_img_only/r2_tab_only*100:.1f}% 포착")
    print(f"  타뷸러+이미지 R²  = {r2_full:.4f}")
    print(f"\n  분산 분해 (근사):")
    print(f"    이미지 고유 기여     : {max(img_unique,0):.4f}  "
          f"({max(img_unique,0)/r2_full*100:.1f}%)")
    print(f"    타뷸러 고유 기여     : {max(tab_unique,0):.4f}  "
          f"({max(tab_unique,0)/r2_full*100:.1f}%)")
    print(f"    이미지-타뷸러 중첩   : {max(shared,0):.4f}  "
          f"({max(shared,0)/r2_full*100:.1f}%)")
    print(f"    미설명 분산          : {unexplained:.4f}")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    pd.DataFrame(part1_rows).to_csv(
        REPORT_DIR / "proxy_part1_reverse.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(part2_rows).to_csv(
        REPORT_DIR / "proxy_part2_scenarios.csv", index=False, encoding="utf-8-sig")

    decomp = {
        "tab_only": round(r2_tab_only, 4),
        "img_only": round(r2_img_only, 4),
        "full":     round(r2_full, 4),
        "img_unique":  round(max(img_unique, 0), 4),
        "tab_unique":  round(max(tab_unique, 0), 4),
        "shared":      round(max(shared, 0), 4),
        "unexplained": round(unexplained, 4),
    }
    pd.DataFrame([decomp]).to_csv(
        REPORT_DIR / "proxy_part3_decomp.csv", index=False, encoding="utf-8-sig")

    print(f"\n  결과 저장 → {REPORT_DIR}/proxy_part*.csv")
    print("=" * 72)
    print("완료.")


if __name__ == "__main__":
    main()
