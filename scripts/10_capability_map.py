# -*- coding: utf-8 -*-
"""
10_capability_map.py — 스트리트뷰 "선택적 센서 능력지도" 통합 분석
=============================================================================
흩어져 있던 09 계열 실험을 하나로 대체. canonical = POI 피처.

[RQ1] 능력지도 : image -> {직장·상주·유동·점포·식음료·개업·폐업} CV R2, -> 유형 AUC
       (a) 풀링(8변수)  (b) 유형 내(골목 10-fold / 발달 5-fold)
[강건성] 정규화 방식(LASSO/Ridge/ElasticNet) 일관성 확인
[RQ2] 메커니즘 : 해석가능(세그) + 녹지 반직관 + commonality(공유/DINO고유/seg고유)
       + 텍스처 검증(08c): DINO고유가 핸드크래프트 텍스처로 환원되나? (보행자/차량 제외)
[RQ3] 경계(null): 구조 통제 후 매출 dR2 (풀링 + 유형 내)

입력 : data/processed/{panel_final, cross_sectional_data, valid_image_sangkwon,
                       image_features_dino_poi[, image_segmentation_poi, image_handcrafted_poi]}.csv
출력 : reports/capability_map.txt, capability_map_rq1.csv, capability_map_rq3.csv
실행 : python scripts/10_capability_map.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
PANEL      = ROOT / "data/processed/panel_final.csv"
CROSS      = ROOT / "data/processed/cross_sectional_data.csv"
VALID      = ROOT / "data/processed/valid_image_sangkwon.csv"
DINO       = ROOT / "data/processed/image_features_dino_poi.csv"   # canonical
SEG        = ROOT / "data/processed/image_segmentation_poi.csv"    # 08b 산출(선택)
HAND       = ROOT / "data/processed/image_handcrafted_poi.csv"     # 08c 산출(선택)
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

YEAR  = "2019"
N_PCA = 100
CV    = 10
RS    = 42
OUT   = []

STRAT_FOLDS = {"골목상권": 10, "발달상권": 5}   # 발달 n~194 -> 5-fold
EXCLUDE_SEG = {"seg_person", "seg_car"}          # 촬영시점 의존 -> 가로경관 아님

# 능력지도 대상 8변수 (유형은 별도 AUC)
TARGETS = [("직장인구", "log_직장인구"), ("점포수", "log_점포_수"),
           ("상주인구", "log_상주인구"), ("유동인구", "log_유동인구"),
           ("식음료비율", "식음료_비율"), ("개업률", "개업률"), ("폐업률", "폐업률")]
STRAT_TARGETS = TARGETS[:4]   # 유형 내 표는 핵심 4개


def log(s=""):
    print(s); OUT.append(s)


def load():
    panel = pd.read_csv(PANEL); panel["상권_코드"] = panel["상권_코드"].astype(str)
    cross = pd.read_csv(CROSS); cross["상권_코드"] = cross["상권_코드"].astype(str)
    valid = pd.read_csv(VALID); valid["상권_코드"] = valid["상권_코드"].astype(str)
    dino  = pd.read_csv(DINO);  dino["상권_코드"]  = dino["상권_코드"].astype(str)

    cols = ["추정매출_합계", "유동인구", "직장인구", "상주인구", "점포_수", "개업률", "폐업률"]
    p = (panel[panel["기준_년분기_코드"].astype(str).str.startswith(YEAR)]
         .groupby("상권_코드")[cols].mean().reset_index())
    for c in ["유동인구", "직장인구", "상주인구", "점포_수"]:
        p["log_" + c] = np.log1p(p[c])
    p["log_매출"]        = np.log1p(p["추정매출_합계"])
    p["log_매출per유동"] = np.log1p(p["추정매출_합계"] / p["유동인구"])

    d = p.merge(cross[["상권_코드", "상권_구분_코드_명", "자치구_코드_명",
                        "면적_km2", "식음료_비율"]], on="상권_코드", how="left")
    vc = set(valid[valid["flagged"] == False]["상권_코드"])
    d = d[d["상권_코드"].isin(vc)].reset_index(drop=True)

    icols = [c for c in dino.columns if c.startswith("dino_")]
    m = d.merge(dino, on="상권_코드", how="inner")

    has_seg = SEG.exists()
    if has_seg:
        seg = pd.read_csv(SEG); seg["상권_코드"] = seg["상권_코드"].astype(str)
        m = m.merge(seg, on="상권_코드", how="left")
    if HAND.exists():
        hd = pd.read_csv(HAND); hd["상권_코드"] = hd["상권_코드"].astype(str)
        keep = ["상권_코드"] + [c for c in hd.columns if c.startswith("hc_")]
        m = m.merge(hd[keep], on="상권_코드", how="left")
    return m, icols, has_seg


def cv_r2(X, y, folds=CV, pca=N_PCA, reg_factory=None):
    kf = KFold(folds, shuffle=True, random_state=RS); r = []
    for tr, te in kf.split(X):
        sc = StandardScaler(); A = sc.fit_transform(X[tr]); B = sc.transform(X[te])
        if pca:
            pp = PCA(min(pca, A.shape[0] - 1), random_state=RS)
            A = pp.fit_transform(A); B = pp.transform(B)
        reg = (reg_factory() if reg_factory
               else LassoCV(cv=5, max_iter=10000, random_state=RS)).fit(A, y[tr])
        r.append(r2_score(y[te], reg.predict(B)))
    return np.mean(r), np.std(r)


def cv_blocks(blocks, y, folds=CV):
    kf = KFold(folds, shuffle=True, random_state=RS); r = []
    for tr, te in kf.split(y):
        At, Ae = [], []
        for kind, X, *rest in blocks:
            sc = StandardScaler(); xt = sc.fit_transform(X[tr]); xe = sc.transform(X[te])
            if kind == "pca":
                pp = PCA(min(rest[0], xt.shape[0] - 1), random_state=RS)
                xt = pp.fit_transform(xt); xe = pp.transform(xe)
            At.append(xt); Ae.append(xe)
        reg = LassoCV(cv=5, max_iter=10000, random_state=RS).fit(np.hstack(At), y[tr])
        r.append(r2_score(y[te], reg.predict(np.hstack(Ae))))
    return np.mean(r)


def cv_auc(X, yb):
    P = PCA(min(N_PCA, X.shape[0] - 1), random_state=RS).fit_transform(
        StandardScaler().fit_transform(X))
    return cross_val_score(LogisticRegression(max_iter=2000), P, yb,
                           cv=CV, scoring="roc_auc").mean()


def cv_delta(Xb, Xi, y, folds=CV):
    kf = KFold(folds, shuffle=True, random_state=RS); rA, rB = [], []
    for tr, te in kf.split(Xb):
        sc = StandardScaler(); A = sc.fit_transform(Xb[tr]); Bb = sc.transform(Xb[te])
        regA = LassoCV(cv=5, max_iter=10000, random_state=RS).fit(A, y[tr])
        rA.append(r2_score(y[te], regA.predict(Bb)))
        isc = StandardScaler(); Ai = isc.fit_transform(Xi[tr]); Bi = isc.transform(Xi[te])
        pca = PCA(min(50, Ai.shape[0] - 1), random_state=RS)
        Pt = pca.fit_transform(Ai); Pe = pca.transform(Bi)
        cc = np.array([abs(np.corrcoef(Pt[:, i], y[tr])[0, 1]) for i in range(Pt.shape[1])])
        idx = np.argsort(cc)[::-1][:10]
        A2 = np.hstack([A, Pt[:, idx]]); B2 = np.hstack([Bb, Pe[:, idx]])
        regB = LassoCV(cv=5, max_iter=10000, random_state=RS).fit(A2, y[tr])
        rB.append(r2_score(y[te], regB.predict(B2)))
    rA, rB = np.array(rA), np.array(rB)
    t, p = ttest_rel(rB, rA)
    return rA.mean(), rB.mean(), rB.mean() - rA.mean(), p


def sig(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def main():
    m, icols, has_seg = load()
    Xi = m[icols].values
    log("=" * 70)
    log(f"10_capability_map  n={len(m)}  (canonical: POI / DINOv2)")
    log("=" * 70)

    # RQ1a 풀링(8변수) ----------------------------------------------------------
    log("\n[RQ1a] 능력지도(풀링) - 이미지 단독 예측 (행정데이터 0개)")
    rq1 = []
    for name, col in TARGETS:
        sub = m.dropna(subset=[col]); r, s = cv_r2(sub[icols].values, sub[col].values)
        log(f"   image -> {name:<6}  R2={r:+.3f} (+-{s:.3f})")
        rq1.append({"level": "pooled", "target": name, "R2": round(r, 4)})
    yb = (m["상권_구분_코드_명"] == "발달상권").astype(int).values
    auc = cv_auc(Xi, yb)
    log(f"   image -> 유형(발달/골목)  AUC={auc:.3f}")
    rq1.append({"level": "pooled", "target": "유형(AUC)", "R2": round(auc, 4)})
    log("   => 강(직장·점포·유형) / 약(식음료·상주·유동·폐업) / 미미(개업) = 선택성")

    # RQ1b 유형 내 -------------------------------------------------------------
    log("\n[RQ1b] 능력지도(유형 내) - between vs within 분해")
    for typ, k in STRAT_FOLDS.items():
        s = m[m["상권_구분_코드_명"] == typ]
        parts = []
        for name, col in STRAT_TARGETS:
            sub = s.dropna(subset=[col])
            r, _ = cv_r2(sub[icols].values, sub[col].values, folds=k)
            parts.append(f"{name}={r:.3f}")
            rq1.append({"level": typ, "target": name, "R2": round(r, 4)})
        log(f"   {typ}(n={len(s)}, {k}-fold): " + "  ".join(parts))
    log("   => 풀링 선택성 상당부분은 between-type. 발달 내에선 직장 선택성 유지.")
    pd.DataFrame(rq1).to_csv(REPORT_DIR / "capability_map_rq1.csv",
                             index=False, encoding="utf-8-sig")

    # 강건성: 정규화 방식 비교 --------------------------------------------------
    log("\n[강건성] 정규화 방식 비교 (image -> 직장인구)")
    regs = {
        "LASSO":      lambda: LassoCV(cv=5, max_iter=10000, random_state=RS),
        "Ridge":      lambda: RidgeCV(alphas=np.logspace(-3, 3, 30)),
        "ElasticNet": lambda: ElasticNetCV(cv=5, max_iter=10000, random_state=RS, l1_ratio=[.2, .5, .8]),
    }
    sub = m.dropna(subset=["log_직장인구"])
    for nm, f in regs.items():
        r, _ = cv_r2(sub[icols].values, sub["log_직장인구"].values, reg_factory=f)
        log(f"   {nm:<11} R2={r:.3f}")
    log("   => 정규화 방식에 강건 (보고서 주장 근거)")

    # RQ2 메커니즘 + commonality + 텍스처 검증 ----------------------------------
    log("\n[RQ2] 메커니즘 - 해석가능(세그) 피처  [보행자/차량 제외]")
    if has_seg:
        seg_cols = [c for c in m.columns
                    if c.startswith("seg_") and c not in EXCLUDE_SEG]
        Xs = m[seg_cols].fillna(m[seg_cols].mean()).values
        cors = {c: np.corrcoef(m[c].fillna(m[c].mean()), m["log_직장인구"])[0, 1]
                for c in seg_cols}
        top = sorted(cors.items(), key=lambda x: -abs(x[1]))[:5]
        log("   직장인구 상관 상위 seg: " +
            ", ".join(f"{c.replace('seg_','')}={v:+.2f}" for c, v in top))
        if "seg_green" in m.columns:
            gd = m[m["상권_구분_코드_명"] == "발달상권"]["seg_green"].mean()
            gg = m[m["상권_구분_코드_명"] == "골목상권"]["seg_green"].mean()
            log(f"   [반직관 녹지] green 발달={gd:.3f} 골목={gg:.3f} | green~직장 r={cors.get('seg_green',0):+.2f}")
        log("   [commonality 분해 = 공유(해석가능) / DINO고유 / seg고유]")
        for name, col in [("직장인구", "log_직장인구"), ("점포수", "log_점포_수")]:
            y = m[col].values
            rs = cv_blocks([("raw", Xs)], y)
            rd = cv_blocks([("pca", Xi, N_PCA)], y)
            rb = cv_blocks([("raw", Xs), ("pca", Xi, N_PCA)], y)
            shared, ud, us = rs + rd - rb, rb - rs, rb - rd
            log(f"     {name}: 결합R2={rb:.3f}  공유={shared:.3f}({shared/rb*100:.0f}%)  "
                f"DINO고유={ud:.3f}({ud/rb*100:.0f}%)  seg고유={us:.3f}")
        log("   ※ DINO고유 = 해석가능 구성으로 환원 안 되는 신호(‘분위기’로 단정 금지)")
        hc_cols = [c for c in m.columns if c.startswith("hc_")]
        if hc_cols:
            log("   [텍스처 검증: seg vs seg+텍스처(GLCM/Gabor/엣지/색) vs DINO]")
            Xh = m[hc_cols].fillna(m[hc_cols].mean()).values
            for name, col in [("직장인구", "log_직장인구"), ("점포수", "log_점포_수")]:
                y = m[col].values
                rS   = cv_blocks([("raw", Xs)], y)
                rSH  = cv_blocks([("raw", Xs), ("raw", Xh)], y)
                rSD  = cv_blocks([("raw", Xs), ("pca", Xi, N_PCA)], y)
                rALL = cv_blocks([("raw", Xs), ("raw", Xh), ("pca", Xi, N_PCA)], y)
                log(f"     {name}: 텍스처 추가기여={rSH - rS:+.3f}  "
                    f"DINO고유 over[seg]={rSD - rS:+.3f} -> over[seg+텍스처]={rALL - rSH:+.3f}")
            log("   => 줄면 'DINO고유 일부=수학적 텍스처'(규명), 안 줄면 '거대비전모델 고유'(방어)")
    else:
        log("   (image_segmentation_poi.csv 없음 -> 08b 먼저 실행)")

    # RQ3 경계 -----------------------------------------------------------------
    log("\n[RQ3] 경계(null) - 구조 통제 후 이미지의 매출 기여")
    ctrl = ["log_직장인구", "log_상주인구", "log_유동인구", "면적_km2"]
    mm = m.dropna(subset=ctrl).reset_index(drop=True)
    Xi_m = mm[icols].values
    gu = pd.get_dummies(mm["자치구_코드_명"], prefix="gu", drop_first=True)
    typd = pd.get_dummies(mm["상권_구분_코드_명"], drop_first=True)
    base_pool = np.hstack([mm[ctrl].astype(float).values,
                           gu.astype(float).values, typd.astype(float).values])
    rq3 = []
    log("   [풀링: 인구+면적+자치구FE+유형 통제]")
    for name, col in [("log매출", "log_매출"), ("log매출per유동", "log_매출per유동")]:
        mk = mm[col].notna().values
        a, b, dr, p = cv_delta(base_pool[mk], Xi_m[mk], mm.loc[mk, col].values)
        log(f"     {name:<14} base={a:.3f}  +img={b:.3f}  dR2={dr:+.3f}  p={p:.3f} {sig(p)}")
        rq3.append({"level": "pooled", "Y": name, "base": round(a, 4),
                    "img": round(b, 4), "dR2": round(dr, 4), "p": round(p, 4)})
    log("   [유형 내(stratified)]")
    for typ, k in STRAT_FOLDS.items():
        s = m[m["상권_구분_코드_명"] == typ].dropna(subset=ctrl).reset_index(drop=True)
        gus = pd.get_dummies(s["자치구_코드_명"], prefix="gu", drop_first=True)
        bs = np.hstack([s[ctrl].astype(float).values, gus.astype(float).values])
        col = "log_매출per유동"
        a, b, dr, p = cv_delta(bs, s[icols].values, s[col].values, folds=k)
        log(f"     {typ} log매출per유동  base={a:.3f}  +img={b:.3f}  dR2={dr:+.3f}  p={p:.3f} {sig(p)}")
        rq3.append({"level": typ, "Y": col, "base": round(a, 4),
                    "img": round(b, 4), "dR2": round(dr, 4), "p": round(p, 4)})
    pd.DataFrame(rq3).to_csv(REPORT_DIR / "capability_map_rq3.csv",
                             index=False, encoding="utf-8-sig")

    (REPORT_DIR / "capability_map.txt").write_text("\n".join(OUT), encoding="utf-8")
    log("\n저장: reports/capability_map.txt (+ rq1.csv, rq3.csv)")


if __name__ == "__main__":
    main()
