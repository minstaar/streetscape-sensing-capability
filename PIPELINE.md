# 파이프라인 (정리본 — 능력지도 보고서 기준)

연구 방향: **스트리트뷰의 "선택적 센서 능력지도"**. 산출물은 `상권_스트리트뷰_능력지도_연구보고서.docx`
(요약은 `README.md`). 이미지 canonical = **POI 수집**(`data/images_poi/`, `image_features_*_poi.csv`).

> **스크립트 번호 규칙**: 번호의 빈칸(04·06·07·09 등)은 *의도된 흔적*이다. 폐기·교체된 단계는
> `_deprecated/`로 보내되 번호는 재배열하지 않는다 — git 이력·상호참조·과거 재현성을 지키기 위함이며,
> 실제 실행 순서는 본 문서가 단일 기준이다. (연구 코드의 일반 관례: 번호 안정 유지 + 문서로 순서 명시.)

## Live 파이프라인 (`scripts/`)

| 단계 | 스크립트 | 역할 | 주요 출력 |
|---|---|---|---|
| 1 | `01_filter_sangkwon.py` | 분석 대상 상권 선별 | `final_sangkwon_list.csv` |
| 2 | `02_aggregate_sales.py` | 분기 추정매출 집계 (※부차 변수, RQ3 전용) | `sales_panel.csv` |
| 3 | `03_process_features.py` | 구조변수(유동·직장·상주·점포·개·폐업률) | `features_panel.csv` |
| 5 | `05_merge_panel.py` | 분기 패널 병합 (**macro 제거됨**) | `panel_final.csv` |
| 5b | `05b_cross_sectional.py` | 2019 기준 단면(유형·면적·식음료·자치구) | `cross_sectional_data.csv` |
| 6 | `06d_collect_images_poi.py` | POI 앵커 GSV 수집 | `data/images_poi/` |
| 7 | `07b_filter_by_year.py` · `07c_restore_flagged.py` | 촬영연도 필터·보완 | `valid_image_sangkwon.csv` |
| 8 | `08_extract_features.py` | DINOv2 피처(블랙박스) | `image_features_dino_poi.csv` |
| 8b | `08b_extract_segmentation.py` | 세그멘테이션 해석가능 피처 (RQ2) | `image_segmentation_poi.csv` |
| 8c | `08c_extract_handcrafted.py` | 핸드크래프트 텍스처/색/엣지 (RQ2 검증) | `image_handcrafted_poi.csv` |
| 10 | `10_capability_map.py` | **능력지도 RQ1+RQ2+RQ3 통합 분석** | `reports/capability_map*` |

실행 순서: 데이터 `01→02→03→05→05b` · 이미지 `06d→07b→07c→08→08b→08c` · 분석 `10`.

### 결과 변수의 위치 (질문 1 관련)
- **매출은 부차 변수**다: `02`가 만든 추정매출은 `panel_final`에 포함되어 **RQ3(경계)에서만** Y로 쓰인다.
  주 분석(RQ1·RQ2)은 구조변수 복원·메커니즘이므로 매출이 중심이 아니다. (02를 01에 합칠 필요는 없음 —
  01=상권 선별, 02=매출 집계로 관심사가 다름.)

## 정리 내역

**삭제(deprecate) — `04_process_macro.py`** (질문 2):
거시변수(CPI·기준금리)는 폐기된 LSTM 예측용. `04`를 `_deprecated/`로 이동하고 `05_merge_panel.py`에서
macro 병합부를 제거했다. 기존 `panel_final.csv`에 남은 CPI/기준금리 컬럼은 `10`이 무시하므로 무해하며,
파이프라인 재실행 시 깨끗하게 재생성된다.

**아카이브 — `data/processed/_archive/`** (구버전·미사용):
`baseline_predictions`(LSTM), `image_features_dino`·`image_features_resnet`·`image_sampling_log`(구 ±90°),
`image_features_resnet_poi`(보고서는 DINO만 사용), `macro_panel`, `test_comparison_report`.

**아카이브 — `reports/_archive/`** (구 09 계열·탐색):
`multimodal_results`·`selective_pc_results`·`stratified_results`·`validation_check`·`proxy_part1~3`·
`covid_panel_summary`·`baseline_metrics`·`cross_sectional_metrics`·`feature_exploration`·`lasso_coefficients`.

**유지 — `reports/`**: `capability_map.txt` + `capability_map_rq1.csv` + `capability_map_rq3.csv` + `year_filter_report.csv`.

**`scripts/_deprecated/`** (provenance 보존): 구 수집(06·06b·06c·07), 구 분석(09·09b·09c·09d·09e·09f),
04_macro, 수집 유틸·탐색. **삭제 아님 — 복구 가능.**

## 검증·수정 내역 (DAG 무결성)

- **[수정] valid 리스트 생성자 누락 해결**: `valid_image_sangkwon.csv`(08·08b·08c·10의 핵심 입력)를
  live 스크립트가 아무도 만들지 않던 끊김을 발견 → **`07c_restore_flagged.py`가 보완 후 최종 valid 리스트를
  산출**하도록 추가(상권_코드·valid_count·flagged). 이제 `06d→07b→07c→08…`가 처음부터 무오류로 연결됨.
- **[정리] 미사용 폴더 아카이브**: `models/`(LSTM .pt)·`cache/`(API 응답 캐시)·`results/`(빈 figures·tables)는
  live 스크립트가 참조하지 않아 `_archive_unused/`로 이동(복구 가능).
- **[최신화] 낡은 주석**: 02→03, 03→05(04 폐기), 05b→10, 07b→07c, 08(07c 이후·_poi 출력·08b/08c/10) 등
  "다음 단계"·출력경로·docstring을 현 구조에 맞게 갱신.
- **[추가완료] Ridge·ElasticNet 강건성**: `10`에 `[강건성]` 섹션(LASSO/Ridge/ElasticNet) 정식 추가 → 보고서 주장 live 재현.

## 미해결 / 권장

1. **연도 정렬 (확인 필요)**: `10`은 구조변수를 **2019년 기준**(panel_final)으로, `05b`의 `cross_sectional_data`는
   **2024년 기준**으로 만든다. 유형·자치구·면적은 시간불변이라 무방하나, **식음료비율(RQ1 약-tier 타깃)만 2024년 값**이
   2019 이미지와 섞인다. 식음료를 2019로 맞출지(점포 2019 데이터에서 재계산) 현 상태로 둘지 결정 필요.
2. **ResNet 제거됨**: `08`은 DINOv2 전용(구 ResNet 추출 경로는 삭제). 과거 ResNet 산출물은 `data/processed/_archive/`에 보존.
3. **용량 회수(선택)**: 미사용 구 이미지 폴더 `data/images/`(±90° 원본), `data/images_test*`·`images_*test*`는
   수동 삭제 가능. `data/images_poi_rejected/`는 `07c` 복구에 쓰이므로 **유지**.
