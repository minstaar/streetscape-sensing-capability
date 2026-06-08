# -*- coding: utf-8 -*-
"""
diagnose_image_features.py — 이미지 피처 다양성 진단
─────────────────────────────────────────────────────
실행: python scripts/diagnose_image_features.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[1]
CROSS_CSV = ROOT / "data/processed/cross_sectional_data.csv"
RESNET    = ROOT / "data/processed/image_features_resnet.csv"
DINO      = ROOT / "data/processed/image_features_dino.csv"
LOG_CSV   = ROOT / "data/processed/image_sampling_log.csv"

cross = pd.read_csv(CROSS_CSV)
cross["상권_코드"] = cross["상권_코드"].astype(str)

for model_name, csv_path, prefix in [("ResNet-50", RESNET, "resnet_"),
                                       ("DINOv2",   DINO,   "dino_")]:
    if not csv_path.exists():
        print(f"  ⚠ {csv_path.name} 없음")
        continue

    print("=" * 60)
    print(f"[{model_name}]")
    print("=" * 60)

    df  = pd.read_csv(csv_path)
    df["상권_코드"] = df["상권_코드"].astype(str)
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    X = df[feat_cols].values
    print(f"  상권 수        : {len(df)}개")
    print(f"  피처 차원      : {len(feat_cols)}개")

    # ── 1. 피처 분산 분포 ────────────────────────────────
    feat_std = X.std(axis=0)
    print(f"\n  [피처별 표준편차 분포]")
    print(f"    mean  = {feat_std.mean():.4f}")
    print(f"    min   = {feat_std.min():.4f}")
    print(f"    p25   = {np.percentile(feat_std, 25):.4f}")
    print(f"    p75   = {np.percentile(feat_std, 75):.4f}")
    print(f"    max   = {feat_std.max():.4f}")
    near_zero = (feat_std < 0.01).sum()
    print(f"    std<0.01 (사실상 상수): {near_zero}개 ({near_zero/len(feat_cols)*100:.1f}%)")

    # ── 2. PCA 누적 설명 분산 ───────────────────────────
    X_sc = StandardScaler().fit_transform(X)
    pca  = PCA(n_components=min(300, len(df)-1, len(feat_cols)))
    pca.fit(X_sc)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"\n  [PCA 누적 설명 분산]")
    for n in [10, 20, 50, 100, 150, 200]:
        if n <= len(cumvar):
            print(f"    PC{n:3d}: {cumvar[n-1]*100:.1f}%")

    # ── 3. 상권 간 vs 상권 내 분산 ──────────────────────
    if LOG_CSV.exists():
        log = pd.read_csv(LOG_CSV)
        log["상권_코드"] = log["상권_코드"].astype(str)
        n_imgs = log[log["success"]==True].groupby("상권_코드").size()
        merged = df.merge(n_imgs.rename("n_images"), on="상권_코드", how="left")
        print(f"\n  [상권당 이미지 수]")
        print(f"    mean = {merged['n_images'].mean():.1f}")
        print(f"    min  = {merged['n_images'].min()}")
        print(f"    max  = {merged['n_images'].max()}")
        print(f"    1장만 있는 상권: {(merged['n_images']==1).sum()}개")

    # ── 4. 상위 PC와 Y의 상관 ────────────────────────────
    merged_y = df.merge(cross[["상권_코드","log_sales","log_sales_total"]]
                        if "log_sales_total" in cross.columns
                        else cross[["상권_코드","log_sales"]],
                        on="상권_코드", how="inner")
    y_vals = merged_y["log_sales_total"].values \
             if "log_sales_total" in merged_y.columns \
             else merged_y["log_sales"].values

    X_merged = merged_y[feat_cols].values
    X_sc2    = StandardScaler().fit_transform(X_merged)
    pca2     = PCA(n_components=min(20, len(X_merged)-1, len(feat_cols)))
    pcs      = pca2.fit_transform(X_sc2)

    print(f"\n  [상위 PC와 log_sales 상관계수]")
    corrs = [np.corrcoef(pcs[:, i], y_vals)[0, 1] for i in range(pcs.shape[1])]
    for i, r in enumerate(corrs[:10], 1):
        bar = "█" * int(abs(r) * 20)
        print(f"    PC{i:2d} (var={pca2.explained_variance_ratio_[i-1]*100:.1f}%)"
              f"  r={r:+.4f}  {bar}")

    max_corr = max(abs(r) for r in corrs)
    print(f"\n  → 최대 |r| = {max_corr:.4f}  "
          + ("⚠ 이미지-매출 상관 매우 낮음" if max_corr < 0.1 else
             "△ 약한 상관" if max_corr < 0.2 else "✓ 일정 수준 상관 존재"))
    print()
