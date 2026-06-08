# -*- coding: utf-8 -*-
"""
feature_importance.py — 타뷸러 피처별 영향도 분석
─────────────────────────────────────────────────
실행:
    python scripts/feature_importance.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/cross_sectional_data.csv")

Y_COL    = "log_sales"
TAB_COLS = ["유동인구","직장인구","점포_수","개업률","폐업률",
            "상권_유형_더미","면적_km2","경쟁_집중도","식음료_비율"]
LOG_COLS = ["유동인구","직장인구","점포_수"]

df = df.dropna(subset=TAB_COLS + [Y_COL]).reset_index(drop=True)
y  = df[Y_COL].values
X  = df[TAB_COLS].copy()
for c in LOG_COLS:
    X[c] = np.log1p(X[c])

# ── 1. 단순 상관계수 ──────────────────────────────────────
print("=" * 55)
print("1. 단순 상관계수 with log_sales")
print("=" * 55)
corr = X.corrwith(pd.Series(y, index=X.index))
for col, r in corr.sort_values(key=abs, ascending=False).items():
    bar = "█" * int(abs(r) * 30)
    sign = "+" if r > 0 else "-"
    print(f"  {col:<20} r={r:+.4f}  {bar}")

# ── 2. 단변량 R² ─────────────────────────────────────────
print()
print("=" * 55)
print("2. 단변량 R² (피처 1개만 사용)")
print("=" * 55)
univ = []
for col in TAB_COLS:
    x_sc = StandardScaler().fit_transform(X[[col]])
    r2   = cross_val_score(LinearRegression(), x_sc, y, cv=5, scoring="r2").mean()
    univ.append((col, r2))
for col, r2 in sorted(univ, key=lambda x: -x[1]):
    bar = "█" * max(0, int(r2 * 40))
    print(f"  {col:<20} R²={r2:.4f}  {bar}")

# ── 3. Drop-one-out ───────────────────────────────────────
print()
print("=" * 55)
print("3. 피처 제거 시 R² 변화 (drop-one-out, Ridge)")
print("=" * 55)
X_sc    = StandardScaler().fit_transform(X)
base_r2 = cross_val_score(Ridge(), X_sc, y, cv=5, scoring="r2").mean()
print(f"  풀 모델 R² = {base_r2:.4f}\n")
doo = []
for col in TAB_COLS:
    rest  = [c for c in TAB_COLS if c != col]
    X_tmp = StandardScaler().fit_transform(X[rest])
    r2_d  = cross_val_score(Ridge(), X_tmp, y, cv=5, scoring="r2").mean()
    doo.append((col, r2_d - base_r2))
for col, delta in sorted(doo, key=lambda x: x[1]):
    bar = "█" * max(0, int(abs(delta) * 200))
    print(f"  -{col:<20} Δ={delta:+.4f}  {bar}")

# ── 4. LASSO 표준화 계수 ─────────────────────────────────
print()
print("=" * 55)
print("4. LASSO 표준화 계수 (전체 모델)")
print("=" * 55)
lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
lasso.fit(X_sc, y)
coefs = sorted(zip(TAB_COLS, lasso.coef_), key=lambda x: abs(x[1]), reverse=True)
for col, coef in coefs:
    bar = "█" * max(0, int(abs(coef) * 30))
    print(f"  {col:<20} β={coef:+.4f}  {bar}")

print()
print(f"  선택 최적 α = {lasso.alpha_:.4f}")
print(f"  0으로 수축된 피처: {sum(1 for _,c in coefs if c==0)}개")
