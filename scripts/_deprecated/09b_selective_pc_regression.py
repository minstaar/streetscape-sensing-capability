# -*- coding: utf-8 -*-
"""
09b_selective_pc_regression.py
Y = log_매출_per_유동  (유동인구 대비 매출 전환율)
X = 개업률, 폐업률, 면적, 경쟁집중도, 식음료비율, log_직장인구 + 자치구FE(24개)
Image = DINOv2 / ResNet-50  (selective PC, leakage-free)

실행: python scripts/09b_selective_pc_regression.py
"""

import warnings, copy
from pathlib import Path
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

try:
    from scipy.stats import ttest_rel as _scipy_ttest_rel
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── 경로 ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
CROSS_CSV  = ROOT / "data/processed/cross_sectional_data.csv"
PANEL_CSV  = ROOT / "data/processed/panel_final.csv"
VALID_CSV  = ROOT / "data/processed/valid_image_sangkwon.csv"
RESNET_CSV = ROOT / "data/processed/image_features_resnet_poi.csv"
DINO_CSV   = ROOT / "data/processed/image_features_dino_poi.csv"
REPORT_DIR = ROOT / "reports"

# ── 설정 ──────────────────────────────────────────────────────────────────────
Y_COL      = "log_매출_per_유동"
CROSS_YEAR = "2019"
USE_GU_FE  = True   # 자치구 고정효과 포함

TABULAR_COLS = [
    "개업률", "폐업률", "면적_km2", "경쟁_집중도", "식음료_비율",
    "log_직장인구",  # 상권 소비 잠재력 통제
]
LOG_TRANSFORM_COLS = []

CV_OUTER     = 10
CV_INNER     = 5
RANDOM_STATE = 42
MAX_PCA      = 300
CORR_THRESH  = 0.10


# ── 전처리 ────────────────────────────────────────────────────────────────────
def make_preprocessor(tab_cols):
    log_cols    = [c for c in LOG_TRANSFORM_COLS if c in tab_cols]
    linear_cols = [c for c in tab_cols if c not in log_cols]
    transformers = []
    if log_cols:
        transformers.append(("log_t", Pipeline([
            ("log",    FunctionTransformer(np.log1p, validate=False)),
            ("scaler", StandardScaler()),
        ]), log_cols))
    if linear_cols:
        transformers.append(("lin_t", StandardScaler(), linear_cols))
    return ColumnTransformer(transformers)


def drop_zero_var(df, cols):
    keep = [c for c in cols if df[c].std() > 0]
    if len(keep) < len(cols):
        print(f"  분산=0 제거: {set(cols)-set(keep)}")
    return keep


def ttest_paired(a, b):
    if _HAS_SCIPY:
        res = _scipy_ttest_rel(np.array(a), np.array(b))
        return float(res.statistic), float(res.pvalue)
    d = np.array(a) - np.array(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(1.0)


def adj_r2(r2, n, p):
    if n - p - 1 <= 0:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


# ── PC 선택 ───────────────────────────────────────────────────────────────────
def select_pcs(pcs_tr, y_tr, thresh=CORR_THRESH, top_k=None):
    corrs = np.array([abs(np.corrcoef(pcs_tr[:, i], y_tr)[0, 1])
                      for i in range(pcs_tr.shape[1])])
    if top_k is not None:
        return np.argsort(corrs)[::-1][:top_k], corrs
    return np.where(corrs >= thresh)[0], corrs


# ── CV 함수 ───────────────────────────────────────────────────────────────────
def run_tab_cv(X_tab, y, tab_cols, label, reg_tmpl):
    kf  = KFold(n_splits=CV_OUTER, shuffle=True, random_state=RANDOM_STATE)
    pre = make_preprocessor(tab_cols)
    r2s, rmses, maes = [], [], []
    for tr, te in kf.split(X_tab):
        p = copy.deepcopy(pre)
        Xtr = p.fit_transform(X_tab.iloc[tr])
        Xte = p.transform(X_tab.iloc[te])
        reg = copy.deepcopy(reg_tmpl)
        reg.fit(Xtr, y[tr])
        yp  = reg.predict(Xte)
        r2s.append(r2_score(y[te], yp))
        rmses.append(np.sqrt(mean_squared_error(y[te], yp)))
        maes.append(mean_absolute_error(y[te], yp))
    r2m = np.mean(r2s)
    print(f"  {label:<50} R2={r2m:.4f}(+-{np.std(r2s):.4f})  "
          f"AdjR2={adj_r2(r2m,len(y),len(tab_cols)):.4f}  RMSE={np.mean(rmses):.4f}")
    return {"r2": np.array(r2s), "rmse": np.array(rmses), "mae": np.array(maes)}


def run_img_cv(X_tab, X_img, y, tab_cols, label, reg_tmpl, thresh=CORR_THRESH, top_k=None):
    kf  = KFold(n_splits=CV_OUTER, shuffle=True, random_state=RANDOM_STATE)
    pre = make_preprocessor(tab_cols)
    r2s, rmses, maes, npcs = [], [], [], []
    for tr, te in kf.split(X_img):
        p = copy.deepcopy(pre)
        Xt_tr = p.fit_transform(X_tab.iloc[tr])
        Xt_te = p.transform(X_tab.iloc[te])
        sc    = StandardScaler()
        Xi_tr = sc.fit_transform(X_img[tr])
        Xi_te = sc.transform(X_img[te])
        nc    = min(MAX_PCA, Xi_tr.shape[1], Xi_tr.shape[0] - 1)
        pca   = PCA(n_components=nc, random_state=RANDOM_STATE)
        pc_tr = pca.fit_transform(Xi_tr)
        pc_te = pca.transform(Xi_te)
        idx, _ = select_pcs(pc_tr, y[tr], thresh=thresh, top_k=top_k)
        if len(idx) == 0:
            idx = np.argsort([abs(np.corrcoef(pc_tr[:, i], y[tr])[0, 1])
                              for i in range(pc_tr.shape[1])])[::-1][:5]
        Xtr = np.hstack([Xt_tr, pc_tr[:, idx]])
        Xte = np.hstack([Xt_te, pc_te[:, idx]])
        reg = copy.deepcopy(reg_tmpl)
        reg.fit(Xtr, y[tr])
        yp  = reg.predict(Xte)
        r2s.append(r2_score(y[te], yp))
        rmses.append(np.sqrt(mean_squared_error(y[te], yp)))
        maes.append(mean_absolute_error(y[te], yp))
        npcs.append(len(idx))
    r2m = np.mean(r2s)
    print(f"  {label:<50} R2={r2m:.4f}(+-{np.std(r2s):.4f})  "
          f"AdjR2={adj_r2(r2m,len(y),len(tab_cols)+np.mean(npcs)):.4f}  "
          f"RMSE={np.mean(rmses):.4f}  PC={np.mean(npcs):.1f}")
    return {"r2": np.array(r2s), "rmse": np.array(rmses),
            "mae": np.array(maes), "n_pc": npcs}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("09b — 선택적 PC 멀티모달 회귀  (log_직장인구 + 자치구FE 포함)")
    print("=" * 72)

    # 데이터 로드
    cross = pd.read_csv(CROSS_CSV)
    panel = pd.read_csv(PANEL_CSV)
    valid = pd.read_csv(VALID_CSV)
    for df in [cross, panel, valid]:
        df["상권_코드"] = df["상권_코드"].astype(str).str.strip()

    p19 = (panel[panel["기준_년분기_코드"].astype(str).str.startswith(CROSS_YEAR)]
           .groupby("상권_코드")[["추정매출_합계", "유동인구", "직장인구"]].mean()
           .reset_index())
    p19[Y_COL]          = np.log1p(p19["추정매출_합계"] / p19["유동인구"])
    p19["log_직장인구"]  = np.log1p(p19["직장인구"])

    cross = cross.merge(p19[["상권_코드", Y_COL, "log_직장인구"]], on="상권_코드", how="left")
    vc    = valid[valid["flagged"] == False]["상권_코드"]
    base  = (cross[cross["상권_코드"].isin(vc)]
             .dropna(subset=[Y_COL, "log_직장인구"])
             .reset_index(drop=True))

    tab_cols = drop_zero_var(base, [c for c in TABULAR_COLS if c in base.columns])

    if USE_GU_FE and "자치구_코드_명" in base.columns:
        gu = pd.get_dummies(base["자치구_코드_명"], prefix="gu", drop_first=True)
        gu_cols = list(gu.columns)
        base = pd.concat([base.reset_index(drop=True), gu.reset_index(drop=True)], axis=1)
        tab_cols = tab_cols + gu_cols
        print(f"  자치구FE: {len(gu_cols)}개  |  tabular 총 {len(tab_cols)}변수")

    y     = base[Y_COL].values
    X_tab = base[tab_cols]
    print(f"  표본: {len(base)}개  Y mean={y.mean():.3f} std={y.std():.3f}\n")

    # 이미지 피처
    img_data = {}
    for name, path, pfx in [("ResNet-50", RESNET_CSV, "resnet_"),
                             ("DINOv2",   DINO_CSV,   "dino_")]:
        if not path.exists():
            continue
        idf = pd.read_csv(path)
        idf["상권_코드"] = idf["상권_코드"].astype(str)
        mg  = base.merge(idf, on="상권_코드", how="inner")
        ic  = [c for c in idf.columns if c.startswith(pfx)]
        img_data[name] = {"mg": mg, "ic": ic,
                          "Xi": mg[ic].values,
                          "Xt": mg[tab_cols],
                          "y":  mg[Y_COL].values}

    REGS = {
        "LASSO":      LassoCV(cv=CV_INNER, max_iter=10000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNetCV(cv=CV_INNER, max_iter=10000, random_state=RANDOM_STATE,
                                   l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]),
        "Ridge":      RidgeCV(alphas=np.logspace(-3, 4, 50), cv=CV_INNER),
    }

    all_res = {}

    for rname, rtmpl in REGS.items():
        print(f"\n{'='*72}\n[{rname}]  Y={Y_COL}  {CV_OUTER}-fold\n{'='*72}\n")

        rA = run_tab_cv(X_tab, y, tab_cols, "Model A | tabular only", rtmpl)
        all_res[(rname, "A")] = rA

        for mname, d in img_data.items():
            r_th = run_img_cv(d["Xt"], d["Xi"], d["y"], tab_cols,
                              f"tabular + {mname} [|r|>{CORR_THRESH}]", rtmpl)
            all_res[(rname, mname, "thresh")] = r_th
            for k in [5, 10, 20]:
                rk = run_img_cv(d["Xt"], d["Xi"], d["y"], tab_cols,
                                f"tabular + {mname} [Top-{k}PC]", rtmpl, top_k=k)
                all_res[(rname, mname, f"top{k}")] = rk

        r2A = all_res[(rname, "A")]["r2"]
        print(f"\n  delta-R2 요약")
        print(f"  {'─'*66}")
        for key, res in all_res.items():
            if not (isinstance(key, tuple) and len(key) == 3 and key[0] == rname):
                continue
            delta   = res["r2"].mean() - r2A.mean()
            t, p    = ttest_paired(res["r2"], r2A)
            sig     = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            pc_str  = f"  PC={np.mean(res['n_pc']):.1f}" if "n_pc" in res else ""
            print(f"  {str(key[1:]):<44}  dR2={delta:+.4f}  t={t:+.3f}  p={p:.3f} {sig}{pc_str}")

    # 저장
    REPORT_DIR.mkdir(exist_ok=True)
    rows = []
    for key, res in all_res.items():
        for fi, (r2, rmse, mae) in enumerate(zip(res["r2"], res["rmse"], res["mae"]), 1):
            rows.append({"reg": key[0], "model": str(key[1:]), "fold": fi,
                         "r2": round(r2, 6), "rmse": round(rmse, 6), "mae": round(mae, 6)})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "selective_pc_results.csv",
                              index=False, encoding="utf-8-sig")

    out_lines = ["=" * 72 + "\n",
                 f"Y={Y_COL}  log_직장인구 포함  자치구FE={USE_GU_FE}\n",
                 "=" * 72 + "\n\n"]
    for rname in REGS:
        out_lines.append(f"[{rname}]\n")
        r2A = all_res.get((rname, "A"), {}).get("r2", np.array([np.nan]))
        for key, res in all_res.items():
            if key[0] != rname:
                continue
            r2m = res["r2"].mean()
            if len(key) == 3:
                t, p = ttest_paired(res["r2"], r2A)
                sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ex   = f"  dR2={r2m-r2A.mean():+.4f}  t={t:+.3f}  p={p:.3f} {sig}"
                ex  += f"  PC={np.mean(res.get('n_pc',[0])):.1f}" if "n_pc" in res else ""
            else:
                ex = ""
            out_lines.append(f"  {str(key[1:]):<44}  R2={r2m:.4f}(+-{res['r2'].std():.4f}){ex}\n")
        out_lines.append("\n")

    txt = REPORT_DIR / "selective_pc_results.txt"
    with open(txt, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    print(f"\n  저장 -> {txt}")
    print("=" * 72)
    print("완료.")


if __name__ == "__main__":
    main()
