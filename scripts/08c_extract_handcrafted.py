# -*- coding: utf-8 -*-
"""
08c_extract_handcrafted.py — 전통적 이미지 처리 피처 (텍스처/색/엣지)  [RQ2 보강]
=============================================================================
[역할]
    commonality 분해에서 "DINO 고유 55%"의 실체를 검증하기 위한 *수학적으로
    정의되는* 핸드크래프트 피처를 산출한다.
      - DINO고유가 줄면  -> 그 일부는 '시각적 텍스처/색'으로 환원 가능(규명)
      - 안 줄면          -> 전통 피처로도 못 잡는 거대비전모델 고유 신호(방어)

[피처(상권별 평균)]
    색   hc_h_mean,hc_h_std,hc_s_mean,hc_s_std,hc_v_mean,hc_v_std,hc_colorful
    엣지 hc_edge_mean,hc_edge_std,hc_edge_density
    GLCM hc_glcm_contrast,_dissimilarity,_homogeneity,_energy,_correlation,_ASM
    Gabor hc_gabor_0,_45,_90,_135   (4 방향 에너지)
    엔트로피 hc_entropy

[입력]  data/images_poi/{상권_코드}/*.jpg, valid_image_sangkwon.csv
[출력]  data/processed/image_handcrafted_poi.csv

[실행]  pip install scikit-image pillow numpy pandas tqdm
        python scripts/08c_extract_handcrafted.py
        # GPU 불필요(CPU). 딥모델 없음.
"""

import glob
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[1]
IMG_DIR   = ROOT / "data/images_poi"
VALID_CSV = ROOT / "data/processed/valid_image_sangkwon.csv"
OUT_CSV   = ROOT / "data/processed/image_handcrafted_poi.csv"
RESIZE    = 256
GLCM_LV   = 32           # GLCM gray level 양자화
RESUME    = True


def feats_one(rgb):
    from skimage.color import rgb2hsv, rgb2gray
    from skimage.filters import sobel, gabor
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import shannon_entropy

    f = {}
    hsv = rgb2hsv(rgb)
    for i, nm in enumerate(["h", "s", "v"]):
        f[f"hc_{nm}_mean"] = float(hsv[..., i].mean())
        f[f"hc_{nm}_std"]  = float(hsv[..., i].std())
    # colorfulness (Hasler-Süsstrunk 간이)
    R, G, B = [rgb[..., i].astype(float) for i in range(3)]
    rg = R - G; yb = 0.5 * (R + G) - B
    f["hc_colorful"] = float(np.sqrt(rg.std()**2 + yb.std()**2)
                             + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))
    gray = rgb2gray(rgb)
    edge = sobel(gray)
    f["hc_edge_mean"]    = float(edge.mean())
    f["hc_edge_std"]     = float(edge.std())
    f["hc_edge_density"] = float((edge > 0.1).mean())
    g8 = (gray * (GLCM_LV - 1)).astype(np.uint8)
    glcm = graycomatrix(g8, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=GLCM_LV, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        f[f"hc_glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    for ang, nm in [(0, "0"), (np.pi/4, "45"), (np.pi/2, "90"), (3*np.pi/4, "135")]:
        real, imag = gabor(gray, frequency=0.3, theta=ang)
        f[f"hc_gabor_{nm}"] = float(np.sqrt(real**2 + imag**2).mean())
    f["hc_entropy"] = float(shannon_entropy(gray))
    return f


def _flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        old = pd.read_csv(OUT_CSV); old["상권_코드"] = old["상권_코드"].astype(str)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates("상권_코드", keep="last")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


def main():
    from PIL import Image
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k):
            return x

    valid = pd.read_csv(VALID_CSV); valid["상권_코드"] = valid["상권_코드"].astype(str).str.strip()
    codes = valid[valid["flagged"] == False]["상권_코드"].tolist()
    done = set()
    if RESUME and OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV); prev["상권_코드"] = prev["상권_코드"].astype(str)
        done = set(prev["상권_코드"]); print(f"  재개: {len(done)}개 건너뜀")

    rows = []
    for code in tqdm(codes, desc="상권"):
        if code in done:
            continue
        imgs = glob.glob(str(IMG_DIR / code / "*.jpg"))
        if not imgs:
            continue
        acc = defaultdict(list)
        for ip in imgs:
            try:
                rgb = np.asarray(Image.open(ip).convert("RGB").resize((RESIZE, RESIZE))) / 255.0
            except Exception:
                continue
            for k, v in feats_one(rgb).items():
                acc[k].append(v)
        if not acc:
            continue
        row = {"상권_코드": code, "n_images": len(next(iter(acc.values())))}
        for k, vs in acc.items():
            row[k] = float(np.mean(vs))
        rows.append(row)
        if len(rows) % 25 == 0:
            _flush(rows); done |= {r["상권_코드"] for r in rows}; rows = []
    _flush(rows)
    print(f"[08c] 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
