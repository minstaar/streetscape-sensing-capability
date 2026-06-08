# -*- coding: utf-8 -*-
"""
09f_covid_panel_interaction.py
─────────────────────────────────────────────────────────────────────────────
[연구 질문]
    코로나19 기간 소비 패턴 변화(재난지원금·배달 전환·선택적 방문) 속에서,
    가로환경의 시각적 품질이 높은 상권이 매출 혜택을 더 많이 받았는가?

[식별 전략]
    이미지(Image_i) × COVID_t 상호작용 + 상권 FE + 분기 FE
    ─────────────────────────────────────────────────────
    ① 상권 FE   → 직장인구·상권유형·입지 등 시간불변 교란 전부 흡수
    ② 분기 FE   → 공통 경기·계절 추세 흡수
    ③ COVID 외생성 → 이미지는 COVID 이전 측정 → 역인과 차단
    ④ 이미지 × COVID → 시변 상호작용 → 상권 FE 이후에도 식별 가능

[모델]
    log_매출_{it} = α_i + γ_t
                  + β  · (PC1_i    × COVID_t)         ← 핵심 계수
                  + δ  · (식음료비율_i × COVID_t)       ← 배달 효과 통제
                  + ε_{it}

    추정: Within 추정량(상권 내 시계열 평균 차분) + 분기 더미 + 클러스터 SE(상권 수준)

[COVID 기간]
    2020Q1 ~ 2021Q4  (사회적 거리두기 본격화~해제 직전)

[이미지 품질 지표]
    DINOv2 768차원 → PCA → PC1 (분산 최대화, Y 미사용 → 순환 없음)
    Robustness: PC1~PC3, PC1~PC5 상호작용 집합

실행: python scripts/09f_covid_panel_interaction.py
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
PANEL_CSV  = ROOT / "data/processed/panel_final.csv"
CROSS_CSV  = ROOT / "data/processed/cross_sectional_data.csv"
VALID_CSV  = ROOT / "data/processed/valid_image_sangkwon.csv"
DINO_CSV   = ROOT / "data/processed/image_features_dino_poi.csv"  # POI 2018 이미지
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── 설정 ──────────────────────────────────────────────────────────────────────
COVID_QUARTERS = {
    "20201","20202","20203","20204",
    "20211","20212","20213","20214",
}
PRE_QUARTERS = {"20191","20192","20193","20194"}
Y_COL = "log_매출"
SANGKWON_TYPE = "골목상권"  # None이면 전체, "골목상권" / "발달상권" 지정 가능


# ══════════════════════════════════════════════════════════════════════════════
# Within 추정량 + 클러스터 SE
# ══════════════════════════════════════════════════════════════════════════════
def within_demean(df, entity_col, cols):
    """상권별 시계열 평균 차분 (Within 변환)"""
    means = df.groupby(entity_col)[cols].transform("mean")
    return df[cols] - means


def ols_cluster_se(X, y, cluster_ids):
    """
    OLS β + 클러스터 강건 표준오차 (Sandwich estimator)
    cluster_ids: 1D array of cluster labels (상권_코드)
    Returns: β, se, t, p, n, k
    """
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    resid   = y - X @ beta

    # Meat of sandwich (cluster-level)
    clusters = np.unique(cluster_ids)
    meat = np.zeros((k, k))
    for c in clusters:
        mask = cluster_ids == c
        Xc   = X[mask]
        ec   = resid[mask]
        sc   = Xc.T @ ec          # k-vector
        meat += np.outer(sc, sc)

    # Small-sample correction: G/(G-1) * (N-1)/(N-k)
    G  = len(clusters)
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    V  = adj * XtX_inv @ meat @ XtX_inv

    se = np.sqrt(np.diag(V))

    from scipy import stats
    t  = beta / se
    p  = 2 * stats.t.sf(np.abs(t), df=G - 1)

    return beta, se, t, p, n, k


def run_panel_fe(panel_df, entity_col, time_col, y_col, x_cols, cluster_col=None):
    """
    이원 고정효과 추정 (상권 FE + 분기 FE)
    ─────────────────────────────────────
    1. 상권 내 시계열 평균 차분 → 상권 FE 제거
    2. 분기 더미 포함 → 분기 FE 처리
    3. OLS + 클러스터 SE
    """
    df = panel_df.copy().dropna(subset=[y_col] + x_cols)
    if cluster_col is None:
        cluster_col = entity_col

    all_cols = [y_col] + x_cols
    # Step 1: Within 변환
    demeaned = within_demean(df, entity_col, all_cols)
    demeaned[cluster_col]  = df[cluster_col].values
    demeaned[time_col]     = df[time_col].values

    # Step 2: 분기 더미 (within 변환 적용)
    time_dummies = pd.get_dummies(df[time_col], prefix="q", drop_first=True)
    time_dummies_demeaned = within_demean(
        pd.concat([df[[entity_col]], time_dummies], axis=1),
        entity_col, list(time_dummies.columns)
    )

    # Step 3: 설계 행렬
    X_mat = np.column_stack([
        demeaned[x_cols].values,
        time_dummies_demeaned.values
    ])
    y_vec   = demeaned[y_col].values
    cl_ids  = df[cluster_col].values

    beta, se, t, p, n, k = ols_cluster_se(X_mat, y_vec, cl_ids)

    # x_cols 부분만 반환 (분기 더미 계수는 표시하지 않음)
    n_x = len(x_cols)
    results = pd.DataFrame({
        "variable": x_cols,
        "coef":     beta[:n_x],
        "se":       se[:n_x],
        "t":        t[:n_x],
        "p":        p[:n_x],
    })
    results["sig"] = results["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01
                  else "*" if p < 0.05 else "†" if p < 0.10 else "n.s."
    )
    # R² (within)
    y_dm   = demeaned[y_col].values
    ss_res = np.sum((y_dm - X_mat @ beta) ** 2)
    ss_tot = np.sum((y_dm - y_dm.mean()) ** 2)
    r2_within = 1 - ss_res / ss_tot

    return results, r2_within, n


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("09f — 이미지 × COVID 패널 상호작용 분석")
    print("=" * 72)
    print(f"  Y = {Y_COL}  |  상권FE + 분기FE + 클러스터SE")
    print(f"  COVID 기간: 2020Q1 ~ 2021Q4\n")

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    panel = pd.read_csv(PANEL_CSV)
    cross = pd.read_csv(CROSS_CSV)
    valid = pd.read_csv(VALID_CSV)
    dino  = pd.read_csv(DINO_CSV)

    for df in [panel, cross, valid, dino]:
        df["상권_코드"] = df["상권_코드"].astype(str).str.strip()
    panel["기준_년분기_코드"] = panel["기준_년분기_코드"].astype(str)

    # ── 유효 이미지 상권 필터 ────────────────────────────────────────────────
    valid_codes = set(valid[valid["flagged"] == False]["상권_코드"])

    # ── 이미지 품질 지표: DINOv2 PCA ─────────────────────────────────────────
    dino_cols = [c for c in dino.columns if c.startswith("dino_")]
    dino_valid = dino[dino["상권_코드"].isin(valid_codes)].copy()

    sc  = StandardScaler()
    Xi  = sc.fit_transform(dino_valid[dino_cols].values)
    pca = PCA(n_components=10, random_state=42)
    PCs = pca.fit_transform(Xi)

    pca_df = pd.DataFrame(
        PCs[:, :5],
        columns=[f"PC{i+1}" for i in range(5)]
    )
    pca_df["상권_코드"] = dino_valid["상권_코드"].values

    print(f"  이미지 상권: {len(pca_df)}개")
    print(f"  PC1 설명 분산: {pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC1~3: {pca.explained_variance_ratio_[:3].sum()*100:.1f}%\n")

    # ── 식음료_비율 (cross_sectional에서) ────────────────────────────────────
    food_df = cross[["상권_코드", "식음료_비율"]].drop_duplicates()

    # ── 패널 구성 ─────────────────────────────────────────────────────────────
    # 분석 기간: 2019 ~ 2021 (Pre + COVID)
    target_q = PRE_QUARTERS | COVID_QUARTERS
    pnl = panel[
        panel["기준_년분기_코드"].isin(target_q) &
        panel["상권_코드"].isin(valid_codes)
    ].copy()

    pnl[Y_COL] = np.log1p(pnl["추정매출_합계"])
    pnl["COVID"] = pnl["기준_년분기_코드"].isin(COVID_QUARTERS).astype(float)

    # 이미지 PC, 식음료비율 merge
    pnl = pnl.merge(pca_df, on="상권_코드", how="inner")
    pnl = pnl.merge(food_df, on="상권_코드", how="left")

    # 상권 유형 필터 (골목상권만 / 전체)
    if SANGKWON_TYPE:
        type_col = "상권_구분_코드_명"
        if type_col in pnl.columns:
            pnl = pnl[pnl[type_col] == SANGKWON_TYPE].reset_index(drop=True)
        else:
            # cross_sectional에서 유형 정보 가져오기
            type_df = pd.read_csv(CROSS_CSV)[["상권_코드", "상권_구분_코드_명"]]
            type_df["상권_코드"] = type_df["상권_코드"].astype(str).str.strip()
            pnl = pnl.merge(type_df, on="상권_코드", how="left")
            pnl = pnl[pnl["상권_구분_코드_명"] == SANGKWON_TYPE].reset_index(drop=True)
        print(f"  ★ 상권 유형 필터: {SANGKWON_TYPE}")

    print(f"  분석 대상: {pnl['상권_코드'].nunique()}개 상권  "
          f"× {pnl['기준_년분기_코드'].nunique()}개 분기  "
          f"= {len(pnl):,}행\n")

    # ── 기술통계: COVID 전후 매출 변화 ───────────────────────────────────────
    print("=" * 72)
    print("기술통계 — COVID 전후 log_매출 변화")
    print("=" * 72)
    pre_mean = pnl[pnl["COVID"] == 0][Y_COL].mean()
    cov_mean = pnl[pnl["COVID"] == 1][Y_COL].mean()
    change   = pnl.groupby("상권_코드").apply(
        lambda g: g[g["COVID"]==1][Y_COL].mean() - g[g["COVID"]==0][Y_COL].mean()
    ).dropna()
    print(f"  Pre-COVID 평균: {pre_mean:.3f}")
    print(f"  COVID 평균:     {cov_mean:.3f}  (Δ={cov_mean-pre_mean:+.3f})")
    print(f"  상권별 ΔY: mean={change.mean():+.3f}  std={change.std():.3f}  "
          f"min={change.min():.3f}  max={change.max():.3f}")
    print(f"  매출 감소 상권: {(change<0).sum()}개 / {len(change)}개 "
          f"({(change<0).mean()*100:.1f}%)\n")

    # ── 상호작용 변수 생성 ────────────────────────────────────────────────────
    for k in range(1, 6):
        pnl[f"PC{k}_x_COVID"] = pnl[f"PC{k}"] * pnl["COVID"]
    pnl["food_x_COVID"] = pnl["식음료_비율"] * pnl["COVID"]

    # 자치구 × COVID 상호작용 — 지역별 COVID 효과 통제
    # (강남/강북 등 지역 차이로 인한 COVID 기간 소비 패턴 차이 흡수)
    gu_dummies = pd.get_dummies(pnl["자치구_코드_명"], prefix="gu", drop_first=True)
    gu_covid_cols = []
    for gu_col in gu_dummies.columns:
        col_name = f"{gu_col}_x_COVID"
        pnl[col_name] = gu_dummies[gu_col].values * pnl["COVID"]
        gu_covid_cols.append(col_name)

    # ── 모델 실행 ─────────────────────────────────────────────────────────────
    print("=" * 72)
    print("Panel FE 추정 결과  (상권FE + 분기FE + 클러스터SE)")
    print("=" * 72)

    models = {
        "Model 1: PC1×COVID (기본)":
            ["PC1_x_COVID"],
        "Model 2: PC1×COVID + 식음료×COVID":
            ["PC1_x_COVID", "food_x_COVID"],
        "Model 3: PC1~3×COVID + 식음료×COVID (Robustness)":
            ["PC1_x_COVID", "PC2_x_COVID", "PC3_x_COVID", "food_x_COVID"],
        "Model 4: PC1~5×COVID + 식음료×COVID (Robustness)":
            ["PC1_x_COVID", "PC2_x_COVID", "PC3_x_COVID",
             "PC4_x_COVID", "PC5_x_COVID", "food_x_COVID"],
        "Model 5: PC1×COVID + 식음료×COVID + 자치구×COVID [지역교란 통제]":
            ["PC1_x_COVID", "food_x_COVID"] + gu_covid_cols,
    }

    summary_rows = []
    for mname, x_cols in models.items():
        res, r2w, n = run_panel_fe(
            pnl, "상권_코드", "기준_년분기_코드", Y_COL, x_cols
        )
        print(f"\n  [{mname}]")
        print(f"  Within R²={r2w:.4f}  N={n:,}")
        print(f"  {'변수':<42}  {'β':>8}  {'SE':>8}  {'t':>7}  {'p':>7}  sig")
        print(f"  {'─'*78}")
        for _, row in res.iterrows():
            # 자치구×COVID 계수는 개별 출력 생략 (모델 5 가독성)
            if row["variable"].startswith("gu_") and "COVID" in row["variable"]:
                continue
            print(f"  {row['variable']:<42}  {row['coef']:>8.4f}  "
                  f"{row['se']:>8.4f}  {row['t']:>7.3f}  "
                  f"{row['p']:>7.3f}  {row['sig']}")
            if "PC1" in row["variable"] and "COVID" in row["variable"]:
                summary_rows.append({
                    "model":           mname,
                    "coef_PC1_COVID":  round(row["coef"], 4),
                    "se":              round(row["se"], 4),
                    "t":               round(row["t"], 3),
                    "p":               round(row["p"], 4),
                    "sig":             row["sig"],
                    "R2_within":       round(r2w, 4),
                    "N":               n,
                })
        # Model 5: 자치구×COVID 요약만 출력
        if "자치구×COVID" in mname:
            gu_res = res[res["variable"].str.startswith("gu_")]
            sig_gu = gu_res[gu_res["p"] < 0.05]
            print(f"  (자치구×COVID {len(gu_res)}개 계수 중 p<0.05: {len(sig_gu)}개 — 지역별 COVID 효과 흡수 확인)")

    # ── 요약 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("요약: PC1×COVID 계수 (핵심 관심 변수)")
    print(f"{'='*72}")
    print(f"  {'모델':<45}  {'β':>8}  {'p':>7}  sig")
    print(f"  {'─'*65}")
    for r in summary_rows:
        name = r["model"].split(":")[0]
        print(f"  {name:<45}  {r['coef_PC1_COVID']:>8.4f}  "
              f"{r['p']:>7.3f}  {r['sig']}")

    # ── 해석 가이드 ───────────────────────────────────────────────────────────
    print(f"\n  [해석]")
    b = summary_rows[1]["coef_PC1_COVID"] if len(summary_rows) > 1 else summary_rows[0]["coef_PC1_COVID"]
    if b > 0:
        print(f"  β(PC1×COVID) = {b:+.4f} > 0")
        print("  → PC1이 높은 상권(DINOv2가 포착한 시각적 특성이 강한 상권)이")
        print("    COVID 기간 상대적으로 더 높은 매출을 기록했음")
    else:
        print(f"  β(PC1×COVID) = {b:+.4f} < 0")
        print("  → PC1이 높은 상권이 COVID 기간 상대적으로 낮은 매출을 기록했음")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    pd.DataFrame(summary_rows).to_csv(
        REPORT_DIR / "covid_panel_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    print(f"\n  결과 저장 → {REPORT_DIR}/covid_panel_summary.csv")
    print("=" * 72)
    print("완료.")


if __name__ == "__main__":
    main()
