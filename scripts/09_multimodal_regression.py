# -*- coding: utf-8 -*-
"""
09_multimodal_regression.py — LASSO / Ridge 멀티모달 회귀 분석
─────────────────────────────────────────────────────────────────────────────
[역할]
    동일한 분석 표본(이미지 유효 상권 N개)에서 세 모델 × 두 회귀를 비교:
      Model A : 타뷸러만
      Model B : 타뷸러 + ResNet-50 이미지 피처
      Model C : 타뷸러 + DINOv2-base 이미지 피처

    회귀 방법: LASSO (α 자동선택) & Ridge (α 자동선택) 동시 실행
    평가:      외부 5-fold CV → R² 평균 ± 표준편차
    주요 결과: ΔR²_B = R²_B − R²_A  (ResNet 기여)
               ΔR²_C = R²_C − R²_A  (DINOv2 기여)

[표본 선정]
    valid_image_sangkwon.csv 에서 flagged=False 상권만 사용
    → 모든 모델이 동일한 N개 상권 위에서 실행 (공정한 비교)

[처리 흐름]
    1. cross_sectional_data.csv + valid_image_sangkwon.csv 로드
    2. 이미지 유효 상권으로 타뷸러 데이터 필터링
    3. 이미지 피처(resnet/dino CSV) 로드 및 타뷸러와 병합
    4. Pipeline 구성:
         타뷸러: StandardScaler
         이미지: StandardScaler → PCA(n_components=PCA_N)
         회귀:   LassoCV(cv=5)
    5. 5-fold CV → R² 집계
    6. 결과 저장

[주의]
    - PCA는 Pipeline 내부에서 fit → data leakage 없음
    - CPI, 기준금리 등 분산이 0인 컬럼 자동 제거
    - 08_extract_features.py 완료 후 실행할 것

[출력]
    reports/multimodal_results.txt  ← 요약 텍스트
    reports/multimodal_results.csv  ← 모델별 fold R² 전체

실행:
    python scripts/09_multimodal_regression.py

의존성:
    pip install scikit-learn pandas numpy --break-system-packages
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import FunctionTransformer

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost 미설치 → pip install xgboost --break-system-packages")
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
CROSS_CSV    = ROOT / "data/processed/cross_sectional_data.csv"
VALID_CSV    = ROOT / "data/processed/valid_image_sangkwon.csv"
RESNET_CSV   = ROOT / "data/processed/image_features_resnet.csv"
DINO_CSV     = ROOT / "data/processed/image_features_dino.csv"
REPORT_DIR   = ROOT / "reports"

# ── 분석 설정 ──────────────────────────────────────────────────────────────────
Y_COL        = "log_sales"    # 점포당 매출 생산성 (log(매출/점포_수))
TABULAR_COLS = [
    "유동인구", "직장인구",
    # 점포_수 제외: Y=매출/점포_수 분모와 동일 → 내생성
    "개업률", "폐업률", "면적_km2",
    "경쟁_집중도", "식음료_비율",
    # 상권_유형_더미 제외 — 이미지가 포착하는 시각 환경과 거의 동일한 정보
]
PCA_N        = 50       # 이미지 피처 PCA 차원
CV_OUTER     = 5        # 외부 CV fold 수
CV_INNER     = 5        # LassoCV 내부 fold 수
RANDOM_STATE = 42

# skewness > 2 → log1p 변환 (CV 파이프라인 내부 적용, leakage 없음)
LOG_TRANSFORM_COLS = ["유동인구", "직장인구"]


# ══════════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def drop_zero_variance(df, cols):
    """분산이 0인 컬럼 제거 (cross-sectional에서 CPI 등)"""
    keep = [c for c in cols if df[c].std() > 0]
    dropped = set(cols) - set(keep)
    if dropped:
        print(f"  ⚠ 분산=0 컬럼 제거: {dropped}")
    return keep


REGRESSORS = {
    "LASSO": LassoCV(cv=CV_INNER, max_iter=10000, random_state=RANDOM_STATE),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 4, 50), cv=CV_INNER),
}

# PCA 차원 그리드 (Model B/C에서 최적 탐색)
PCA_GRID = [50, 100, 150, 200, 300]


def make_tab_preprocessor(tab_cols):
    """타뷸러 전처리: skew 큰 컬럼은 log1p → StandardScaler, 나머지는 StandardScaler"""
    log_cols    = [c for c in LOG_TRANSFORM_COLS if c in tab_cols]
    linear_cols = [c for c in tab_cols if c not in log_cols]
    transformers = []
    if log_cols:
        transformers.append((
            "log_tab",
            Pipeline([
                ("log",    FunctionTransformer(np.log1p, validate=False)),
                ("scaler", StandardScaler()),
            ]),
            log_cols,
        ))
    if linear_cols:
        transformers.append(("lin_tab", StandardScaler(), linear_cols))
    return ColumnTransformer(transformers)


def build_pipeline_tabular(tab_cols, regressor):
    """Model A: 타뷸러만 (skew 보정 포함)"""
    return Pipeline([
        ("pre", make_tab_preprocessor(tab_cols)),
        ("reg", regressor),
    ])


def build_pipeline_multimodal(tab_cols, img_cols, regressor):
    """Model B/C: 타뷸러(skew 보정) + 이미지(PCA)"""
    n_components = min(PCA_N, len(img_cols), 200)
    tab_pre  = make_tab_preprocessor(tab_cols)
    img_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=RANDOM_STATE)),
    ])
    preprocessor = ColumnTransformer([
        ("tab", tab_pre,  tab_cols),
        ("img", img_pipe, img_cols),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("reg", regressor),
    ])


def adjusted_r2(r2, n, p):
    """Adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1)"""
    if n - p - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def run_cv(pipeline, X, y, label, n_features):
    """5-fold CV → R², RMSE, MAE, Adj-R² 반환"""
    kf = KFold(n_splits=CV_OUTER, shuffle=True, random_state=RANDOM_STATE)
    # DataFrame 유지 (ColumnTransformer가 컬럼명 필요)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    n = len(y)

    r2_list, rmse_list, mae_list = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr = X.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        r2_list.append(r2_score(y_te, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae_list.append(mean_absolute_error(y_te, y_pred))

    r2   = np.mean(r2_list)
    rmse = np.mean(rmse_list)
    mae  = np.mean(mae_list)
    adj  = adjusted_r2(r2, n, n_features)

    print(f"  {label:<38} "
          f"R²={r2:.4f}(±{np.std(r2_list):.4f})  "
          f"AdjR²={adj:.4f}  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}")

    return {
        "r2":      np.array(r2_list),
        "rmse":    np.array(rmse_list),
        "mae":     np.array(mae_list),
        "adj_r2":  adj,
    }


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("09_multimodal_regression.py — LASSO 멀티모달 회귀")
    print("=" * 65)

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    cross_df = pd.read_csv(CROSS_CSV)
    valid_df = pd.read_csv(VALID_CSV)

    # 이미지 유효 상권 필터링 → 최종 분석 표본
    valid_codes = valid_df[valid_df["flagged"] == False]["상권_코드"].astype(str)
    cross_df["상권_코드"] = cross_df["상권_코드"].astype(str)
    base_df = cross_df[cross_df["상권_코드"].isin(valid_codes)].copy()
    base_df = base_df.reset_index(drop=True)
    print(f"\n  최종 분석 표본: {len(base_df)}개 상권 (flagged=False 기준)")

    # 분산=0 컬럼 제거
    tab_cols = drop_zero_variance(base_df, [c for c in TABULAR_COLS if c in base_df.columns])

    y = base_df[Y_COL].values
    X_tab = base_df[tab_cols]

    # ── 이미지 피처 사전 로드 ─────────────────────────────────────────────────
    image_data = {}
    for model_name, csv_path, prefix in [
        ("B_resnet", RESNET_CSV, "resnet_"),
        ("C_dino",   DINO_CSV,   "dino_"),
    ]:
        if not csv_path.exists():
            print(f"  ⚠ {csv_path.name} 없음 → Model {model_name[-1]} 건너뜀")
            continue
        img_df = pd.read_csv(csv_path)
        img_df["상권_코드"] = img_df["상권_코드"].astype(str)
        merged   = base_df.merge(img_df, on="상권_코드", how="inner")
        img_cols = [c for c in img_df.columns if c.startswith(prefix)]
        if len(merged) < len(base_df):
            print(f"  ⚠ 피처 미추출 상권 {len(base_df)-len(merged)}개 제외")
        image_data[model_name] = (merged, img_cols)

    # ── 회귀 방법별 실행 ──────────────────────────────────────────────────────
    import copy
    all_results = {}   # key: (reg_name, model_key)

    for reg_name, reg_template in REGRESSORS.items():
        print(f"\n[{reg_name}]  Y = {Y_COL}  |  {CV_OUTER}-fold CV\n")

        # Model A
        reg    = copy.deepcopy(reg_template)
        pipe_A = build_pipeline_tabular(tab_cols, reg)
        res_A  = run_cv(pipe_A, X_tab, y, "Model A │ 타뷸러만", len(tab_cols))
        all_results[(reg_name, "A")] = res_A

        # Model B / C
        for model_key, label_suffix in [("B_resnet", "ResNet-50"), ("C_dino", "DINOv2-base")]:
            if model_key not in image_data:
                continue
            merged, img_cols = image_data[model_key]
            X_merged  = merged[tab_cols + img_cols]
            y_merged   = merged[Y_COL].values
            n_feat     = len(tab_cols) + min(PCA_N, len(img_cols))
            reg        = copy.deepcopy(reg_template)
            pipe       = build_pipeline_multimodal(tab_cols, img_cols, reg)
            lbl        = f"Model {model_key[0]} │ 타뷸러 + {label_suffix}"
            res        = run_cv(pipe, X_merged, y_merged, lbl, n_feat)
            all_results[(reg_name, model_key[0])] = res

        # ΔR² 요약 + paired t-test
        r2_A    = all_results[(reg_name, "A")]["r2"]
        rmse_A  = all_results[(reg_name, "A")]["rmse"]
        print(f"  {'─'*60}")
        for (rn, mk), res in all_results.items():
            if rn != reg_name or mk == "A":
                continue
            delta_r2   = res["r2"].mean()   - r2_A.mean()
            delta_rmse = res["rmse"].mean() - rmse_A.mean()
            # fold별 R² 차이에 대한 paired t-test
            t_stat, p_val = stats.ttest_rel(res["r2"], r2_A)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "(n.s.)"
            print(f"  Model {mk}  ΔR²={delta_r2:+.4f}  ΔRMSE={delta_rmse:+.4f}"
                  f"  t={t_stat:+.3f}  p={p_val:.3f} {sig}")

    # ── XGBoost 타뷸러 비교 ──────────────────────────────────────────────────
    if HAS_XGB:
        print(f"\n[XGBoost]  Y = {Y_COL}  |  {CV_OUTER}-fold CV\n")
        xgb = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        )
        # XGBoost는 스케일 불필요, log 변환만 적용
        log_cols_xgb = [c for c in LOG_TRANSFORM_COLS if c in tab_cols]
        X_xgb = X_tab.copy()
        for c in log_cols_xgb:
            X_xgb[c] = np.log1p(X_xgb[c])
        res_xgb = run_cv(xgb, X_xgb, y, "XGBoost │ 타뷸러만", len(tab_cols))
        all_results[("XGBoost", "A")] = res_xgb

    # ── PCA 그리드서치 (DINOv2 × LASSO) ─────────────────────────────────────
    print(f"\n[PCA 그리드서치]  DINOv2 × LASSO  |  {CV_OUTER}-fold CV\n")
    if "C_dino" in image_data:
        merged_dino, img_cols_dino = image_data["C_dino"]
        X_m = merged_dino[tab_cols + img_cols_dino]
        y_m = merged_dino[Y_COL].values
        best_pca, best_r2 = PCA_N, -np.inf
        for pca_n in PCA_GRID:
            import copy
            reg_tmp = copy.deepcopy(
                LassoCV(cv=CV_INNER, max_iter=10000, random_state=RANDOM_STATE)
            )
            # 임시로 전역 PCA_N 교체
            orig_pca = globals().get("PCA_N", 50)
            globals()["PCA_N"] = pca_n
            pipe_tmp = build_pipeline_multimodal(tab_cols, img_cols_dino, reg_tmp)
            res_tmp  = run_cv(pipe_tmp, X_m, y_m,
                              f"  DINOv2 PCA={pca_n:3d}", len(tab_cols) + pca_n)
            globals()["PCA_N"] = orig_pca
            if res_tmp["r2"].mean() > best_r2:
                best_r2, best_pca = res_tmp["r2"].mean(), pca_n
        print(f"\n  → 최적 PCA 차원: {best_pca}  (R²={best_r2:.4f})")

    # ── 저장 ─────────────────────────────────────────────────────────────────
    REPORT_DIR.mkdir(exist_ok=True)

    # CSV: fold별 전체 지표
    rows = []
    for (reg_name, model_key), res in all_results.items():
        for fold_i, (r2, rmse, mae) in enumerate(
                zip(res["r2"], res["rmse"], res["mae"]), 1):
            rows.append({
                "regressor": reg_name, "model": model_key, "fold": fold_i,
                "r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6),
            })
    pd.DataFrame(rows).to_csv(REPORT_DIR / "multimodal_results.csv",
                               index=False, encoding="utf-8-sig")

    # TXT 요약
    with open(REPORT_DIR / "multimodal_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Multimodal Regression Results (LASSO & Ridge, 5-fold CV)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"분석 표본 : {len(base_df)}개 상권\n")
        f.write(f"Y 변수    : {Y_COL}\n")
        f.write(f"타뷸러    : {tab_cols}\n")
        f.write(f"PCA 차원  : {PCA_N}\n\n")
        for reg_name in REGRESSORS:
            f.write(f"[{reg_name}]\n")
            r2_a   = all_results.get((reg_name, "A"), {}).get("r2",   np.array([np.nan])).mean()
            rmse_a = all_results.get((reg_name, "A"), {}).get("rmse", np.array([np.nan])).mean()
            for (rn, mk), res in all_results.items():
                if rn != reg_name:
                    continue
                r2m   = res["r2"].mean()
                rmsem = res["rmse"].mean()
                maem  = res["mae"].mean()
                adjr2 = res["adj_r2"]
                if mk != "A":
                    r2_base = all_results.get((reg_name, "A"), {}).get("r2", np.array([np.nan]))
                    t_stat, p_val = stats.ttest_rel(res["r2"], r2_base)
                    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "(n.s.)"
                    delta = (f"  ΔR²={r2m-r2_a:+.4f}  ΔRMSE={rmsem-rmse_a:+.4f}"
                             f"  t={t_stat:+.3f}  p={p_val:.3f}{sig}")
                else:
                    delta = ""
                f.write(f"  Model {mk}: R²={r2m:.4f}(±{res['r2'].std():.4f})  "
                        f"AdjR²={adjr2:.4f}  RMSE={rmsem:.4f}  MAE={maem:.4f}"
                        f"{delta}\n")
            f.write("\n")

    print(f"\n  결과 저장 → {REPORT_DIR}/multimodal_results.txt")
    print(f"{'=' * 65}")
    print("\n완료. 이미지 피처가 없으면 08_extract_features.py 먼저 실행.")


if __name__ == "__main__":
    main()
