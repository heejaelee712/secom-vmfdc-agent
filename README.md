# SECOM 반도체 불량 예측 및 VM-FDC 가이던스 에이전트

> UCI SECOM 센서 데이터를 활용해 웨이퍼 불량을 사전 예측하고, 비용 기준에 따라 선별검사 의사결정을 지원하는 ML 파이프라인과 XAI/LLM 기반 진단 에이전트를 구현한 개인 포트폴리오 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-green)
![SHAP](https://img.shields.io/badge/SHAP-XAI-purple)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-LLM%20Report-lightgrey)

---

## 1. 문제 정의 및 배경

반도체 Etch 공정에서는 모든 웨이퍼를 정밀검사(VM, Virtual Metrology)하기엔 비용·시간 부담이 크고, 검사를 생략하면 불량을 그대로 다음 공정으로 흘려보낼 위험이 있습니다.

이 프로젝트는 다음 질문에서 출발했습니다.

> **센서 데이터만으로 불량 가능성이 높은 웨이퍼를 미리 골라내, 검사 자원을 효율적으로 배분할 수 있을까?**

단순히 모델 성능 지표(Accuracy, AUC)를 올리는 것을 목표로 하지 않고, **데이터 증강 기법이 만들어내는 인공적 신호(Artifact)를 검증하고, 비용 구조를 고려한 의사결정 지원까지 이어지는 전체 파이프라인**을 구축하는 데 중점을 두었습니다.

---

## 2. 데이터셋

[UCI SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom)

| 항목 | 내용 |
|---|---|
| 샘플 수 | 1,567개 (웨이퍼 단위) |
| 센서(피처) 수 | 590개 → 전처리 후 약 440개 |
| 불량 비율 | 6.6% (심한 클래스 불균형) |
| 라벨 | Pass(정상) / Fail(불량) |

전처리 단계에서 결측치 50% 이상 컬럼과 상수 컬럼을 제거했습니다.

---

## 3. 분석 파이프라인

```
전처리 (결측치/상수 컬럼 제거)
   ↓
Feature Selection (SelectKBest, 440 → 100개)
   ↓
데이터 증강 비교 (None vs SMOTE vs GMM+PCA)
   ↓
Cost-sensitive Threshold 최적화
   ↓
통계적 검증 (5×5 Repeated Nested CV + Wilcoxon Test)
   ↓
SHAP 기반 원인 진단 + Artifact 필터링
   ↓
LLM(GPT-4o-mini) 자연어 리포트
```

모든 전처리/증강/Feature Selection은 **train fold 내부에서만 fit**하여 data leakage를 방지하는 구조로 설계했습니다.

---

## 4. 핵심 발견: 가설 → 검증 → 결론

이 프로젝트에서 가장 비중 있게 다룬 부분은 "정교한 기법이 항상 더 좋은 결과를 보장하는가"를 직접 검증한 과정입니다.

### 4-1. Data Leakage 발견 및 수정

초기 버전에서는 cost-sensitive threshold를 **test fold의 정답값**으로 탐색하고 있었습니다. 이로 인해 성능이 낙관적으로 추정되는 leakage가 발생했습니다(검사량감소 91.8%, Recall 25.0%처럼 비현실적인 결과가 나옴).

→ Threshold 탐색을 **train fold 내부 OOF(Out-of-Fold) 확률**로만 수행하도록 수정하여 leakage를 제거했습니다.

### 4-2. GMM+PCA 증강이 만든 Artifact 발견

스케일링 없이 PCA를 적용한 GMM+PCA 증강 버전에서, 특정 센서(`sensor_67`)가 Feature Importance **1위(0.220)**를 차지했고, 2위와 importance 격차가 약 8배에 달했습니다.

**가설**: PCA가 측정 단위/분산이 큰 센서에 의해 왜곡되었을 것이다.

**검증**: PCA 적용 전 RobustScaler를 추가한 결과, `sensor_67`의 순위가 1위 → 9위로 완화되었습니다.

**추가 검증**: 그러나 random seed를 10회 바꿔 반복한 결과, 평균 6.0위(표준편차 2.4)로 **일관되게 상위권**을 유지했습니다. 분포를 확인한 결과 Skewness 20.77로 극단적으로 치우친 분포였습니다.

**결론**: 스케일링으로 왜곡을 완화했지만 완전히 해소되지 않은 **구조적 잔존 artifact**로 최종 판정하고 분석 대상에서 제외했습니다.

### 4-3. 더 근본적인 원인 — 표본 부족과 차원의 문제

`sensor_67`을 제거한 뒤에도 같은 방식(440차원 전체로 GMM+PCA 적용)에서 `sensor_74`가 importance 1위(0.360)로 새롭게 부각되는 현상을 확인했습니다. 분포를 보니 Skewness 39.5로 더욱 극단적이었습니다.

**검증**: Feature Selection(SelectKBest)을 GMM+PCA보다 먼저 적용한 결과, `sensor_74`는 10회 반복 모두 selection 단계에서 통계적으로 유의하지 않다는 이유로 자동 제외되었습니다.

**결론**: 소수클래스 표본(약 100개)으로 440차원의 분포를 추정하는 것 자체가 통계적으로 불안정하며, **Feature Selection을 GMM+PCA보다 먼저 적용**해야 이런 종류의 artifact를 줄일 수 있다는 것을 확인했습니다.

### 4-4. 증강 기법 간 통계적 성능 비교

위 발견이 실제 모델 성능에도 영향을 주는지, **5×5 Repeated Nested CV**(outer 5-fold × 5회 반복 = 25개 성능 점수)로 None / SMOTE / GMM+PCA를 비교했습니다.

| 증강 기법 | Recall | Precision | AUC | 검사량감소 |
|---|---|---|---|---|
| None | 0.698 ± 0.110 | 0.123 ± 0.017 | 0.723 ± 0.038 | 61.3% ± 8.5% |
| SMOTE | **0.711 ± 0.156** | 0.116 ± 0.018 | 0.721 ± 0.044 | 58.4% ± 10.5% |
| GMM+PCA | 0.627 ± 0.130 | 0.131 ± 0.023 | 0.732 ± 0.045 | 67.4% ± 7.7% |

**Wilcoxon signed-rank test 결과 (p < 0.05 기준):**

- Recall: GMM+PCA가 None, SMOTE 대비 **통계적으로 유의하게 낮음** (p=0.048, p=0.008)
- Precision: GMM+PCA가 SMOTE 대비 유의하게 높음 (p=0.0066)
- AUC: 세 기법 간 유의한 차이 없음 (모두 p > 0.05)
- None vs SMOTE: 모든 지표에서 유의한 차이 없음

**결론**: GMM+PCA는 모델의 근본적 판별력(AUC)을 개선하지 못했고, VM 스크리닝의 핵심 지표인 Recall에서는 오히려 통계적으로 더 낮은 성능을 보였습니다. 이는 4-2~4-3에서 발견한 구조적 불안정성이 실제 일반화 성능 저하로 이어진다는 것을 뒷받침합니다. **"더 정교한 증강 기법이 항상 더 나은 결과를 보장하지 않는다"**는 것을 통계적으로 확인하고, 최종적으로 SMOTE 기반 파이프라인을 채택했습니다.

### 증강 기법별 성능 분포 (5×5 Repeated Nested CV, 25회)
<img width="4400" height="1476" alt="Augmentation Method Comparision2_REV" src="https://github.com/user-attachments/assets/711bb05d-da3a-48b2-b703-831bc643c176" />




GMM+PCA의 Recall 분포가 다른 두 기법보다 낮고 더 넓게 퍼져 있어, 통계 검정 결과(Wilcoxon p<0.05)와 일치하는 시각적 근거를 보여줍니다.

---

## 5. Cost-sensitive Threshold 최적화

불량을 놓치는 비용(FN)이 헛검사 비용(FP)보다 몇 배 더 큰지를 의미하는 **Cost Ratio**를 파라미터화하고, 이를 최소화하는 threshold를 train fold OOF 확률로 탐색하는 알고리즘을 직접 구현했습니다.

Cost Ratio를 5~500까지 바꿔가며 시뮬레이션한 결과:

- Cost Ratio가 100 이상으로 올라가면 Recall이 **0.981에서 정체**되어 더 이상 오르지 않습니다.
- 이는 현재 센서 정보만으로는 약 1.9%의 불량이 구조적으로 예측 불가능한 한계로 해석됩니다.

최종 채택 기준인 **Cost Ratio 15:1**(SMOTE 기준)에서 **Recall 71.1%, 검사량감소 58.4%**를 달성했습니다.


### Cost Ratio별 Trade-off
<img width="650" alt="VM Screening Trade-off_REV" src="https://github.com/user-attachments/assets/e9e89899-4530-4eaf-a13e-d47335e84ccb" />






위 그래프에서 보듯, Cost Ratio가 100 이상으로 올라가면 Recall이 0.981에서 정체되어 더 이상 오르지 않습니다. 이는 현재 보유한 센서 정보만으로는 예측이 불가능한 불량 유형이 일부 존재함을 시사합니다.

> ⚠️ Cost Ratio 15:1은 임의로 가정한 값입니다. 실제 현장에 적용하려면 공정 단계별 실측 비용(검사 비용, 불량 유출 손실)으로 보정이 필요합니다. 본 프로젝트는 이 비율을 고정값이 아닌 **조정 가능한 파라미터**로 설계하여, 비용 데이터가 주어지면 그에 맞는 최적 threshold를 자동으로 찾는 프레임워크를 제공하는 데 초점을 맞췄습니다.

---

## 6. XAI 및 LLM 가이던스 에이전트

`EtchProcessAgent` 클래스를 3단계로 구성했습니다.

| Stage | 역할 | 의사결정 주체 |
|---|---|---|
| 1. VM 스크리닝 | threshold 기준으로 INSPECT/PASS 판정 | 모델 확률 + threshold |
| 2. XAI 가이던스 | SHAP으로 원인 센서 추출, artifact 센서(`sensor_67` 등) 자동 필터링 | SHAP + 화이트리스트 |
| 3. Action Control | 위험도에 따라 Interlock/Recipe 점검 등 조치 결정 | 룰 기반 로직 |


### 실행 예시

위험 웨이퍼가 감지되면 다음과 같은 흐름으로 가이던스가 생성됩니다.

----
[⚠️ 위험 감지] Wafer 2008-07-29 08:23:00 불량 확률: 0.0173 (기준 0.0080, Cost Ratio 15:1) → 정밀 VM 대상 선별

[🤖 Agent 가이던스 리포트]
 - 주요 유발 인자(위험 증가 기준): sensor_15
 - 추천 액션: sensor_15 관련 Recipe 파라미터 및 Chamber 상태 점검

[LLM 상세 리포트]
1. 종합 판단: 점검 대상 센서인 sensor_15는 위험 증가 방향으로 작용하고 있습니다.
2. 핵심 원인 추정 센서 및 해석: sensor_15의 측정값은 5.03으로, 위험 증가에 기여하고 있습니다.
   반면, sensor_24와 sensor_0은 각각 위험을 감소시키는 방향으로 작용하고 있습니다.
3. 권고 조치: sensor_15의 상태를 면밀히 점검하고, 필요시 조치를 취하여 위험 요소를 최소화하는 것이 필요합니다.

[🔧 Interlock/Recipe 시스템 모킹] action_type=RECIPE_ADJUSTMENT_SUGGESTION, signal=RECIPE_REVIEW_REQUEST
  → 불량 확률 1.7% - 중위험. sensor_15 관련 Recipe 파라미터 점검 권장 (예: 해당 공정 Step의 가스 유량/RF Power ±5% 조정 검토).

-----

이 출력에서 확인할 수 있듯, **LLM은 SHAP이 이미 산출한 원인 센서와 룰 기반 로직이 결정한 조치를 자연어로 풀어 설명할 뿐, 위험 여부나 조치 자체를 직접 판단하지 않습니다.**

GPT-4o-mini는 위 단계에서 산출된 결과(원인 센서, 위험도, 권고조치)를 **현장 엔지니어가 읽기 좋은 자연어 리포트로 변환**하는 역할만 수행합니다. 실제 판정/조치 결정은 SHAP과 룰 기반 로직이 담당하며, LLM에 의사결정을 위임하지 않는 구조로 설계했습니다.

---

## 7. 최종 결과 요약

| 항목 | 결과 |
|---|---|
| 최종 채택 증강 기법 | SMOTE (통계적 검증 기반) |
| Recall | 71.1% |
| Precision | 11.6% |
| AUC | 0.721 |
| 검사량감소 (Cost Ratio 15:1 기준) | 58.4% |
| 증강기법 간 통계 검정 | Wilcoxon signed-rank test, GMM+PCA의 Recall 열등성 확인 (p<0.05) |
| 발견 및 제거한 Artifact 센서 | sensor_67 (반복검증으로 구조적 artifact 판정) |

---

## 8. 기술적 한계 및 추후 개선 방향

- **Precision이 낮음 (11~13%)**: 위험으로 분류된 웨이퍼 중 실제 불량은 1/9 수준. 표본이 적은(불량 104개) SECOM 데이터의 본질적 한계로 보이며, 추가 공정 데이터/도메인 피처가 필요합니다.
- **Recall 상한선 존재**: Cost Ratio를 아무리 높여도 Recall이 98.1%에서 정체되어, 현재 센서만으로 예측 불가능한 불량 유형이 일부 존재함을 시사합니다.
- **Cost Ratio는 가정값**: 실제 운영 환경의 비용 데이터로 보정이 필요합니다.
- **Action Control은 모킹(mocking) 단계**: 실제 MES/장비 API 연동은 구현하지 않았습니다.

---

## 9. 실행 방법

```bash
# 패키지 설치
pip install -r requirements.txt

# 전체 파이프라인 실행
python secom_etch_agent.py
```

또는 `notebooks/` 폴더의 Jupyter 노트북을 셀 단위로 순서대로 실행하면 전처리부터 통계적 검증, Agent 데모까지 동일한 흐름을 따라갈 수 있습니다.

데이터는 [UCI SECOM Dataset](https://archive.ics.uci.edu/dataset/179/secom)을 다운로드해 `data/uci-secom.csv`에 위치시키거나, `SECOM_DATA_PATH` 환경변수로 경로를 지정할 수 있습니다. LLM 리포트 기능을 사용하려면 `OPENAI_API_KEY` 환경변수를 설정하세요. (선택 사항이며, 키가 없어도 SHAP 기반 가이던스는 정상 작동합니다.)

---

## 10. 폴더 구조

```
.
├── secom_etch_agent.py          # 전체 분석 파이프라인 (Cell 1~15 통합)
├── notebooks/
│   └── UCI_SECOM_경력_agent반영.ipynb
├── data/
│   └── uci-secom.csv            # (직접 다운로드 필요)
├── outputs/
│   └── repeated_nested_cv_results.csv
│   └── wilcoxon_test_results.csv
├── requirements.txt
└── README.md
```

---

## 기술 스택

`Python` `scikit-learn` `XGBoost` `imbalanced-learn (SMOTE)` `SHAP` `OpenAI API (GPT-4o-mini)` `pandas` `numpy` `scipy (Wilcoxon test)`
