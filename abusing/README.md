# Cooperative Trading Detection System

공모거래 탐지 시스템 v2.0

## 개요

복수 계정 간 협력하여 부당 이득을 취하는 패턴을 탐지합니다.

## 주요 기능

### 1. 탐지 알고리즘

-   **필수 조건 필터링**: 시간 동시성, 동일 심볼, 주요 코인 제외
-   **4차원 점수 시스템**:
    -   PnL 비대칭성 (35점) - 한쪽만 큰 이익
    -   시간 근접도 (25점)
    -   IP 공유 (25점)
    -   포지션 겹침 (15점)
-   **위험도 분류**: CRITICAL / HIGH / MEDIUM / LOW
-   **네트워크 분석**: 연결된 계정 그룹 탐지

### 2. 탐지 기준

#### 필수 조건

-   오픈 시간차 ≤ 2분
-   클로즈 시간차 ≤ 2분
-   동일 심볼에서 거래
-   동일 사이드 (LONG/SHORT)
-   포지션 시간 겹침
-   주요 심볼 제외 (BTC, ETH, SOL, XRP, BNB, DOGE)

#### 위험도 임계값

-   **CRITICAL**: 85점 이상
-   **HIGH**: 70점 이상
-   **MEDIUM**: 50점 이상
-   **LOW**: 50점 미만

## 사용법

### 기본 실행

```python
from cooperative_trading import run_detection

result = run_detection("problem_data_final.xlsx")
```

### 커스텀 설정

```python
from cooperative_trading import run_detection, DetectionConfig

config = DetectionConfig(
    max_open_time_diff_min=2.0,
    max_close_time_diff_min=2.0,
    exclude_major_symbols=True,
    critical_threshold=85,
    min_group_size=2,
    output_dir="./output/cooperative"
)

result = run_detection("problem_data_final.xlsx", config)
```

## 출력 파일

### 1. `trade_pairs_detailed.csv`

-   모든 의심 거래 쌍 상세 정보
-   점수 breakdown 포함
-   승자/패자 계정 정보

### 2. `cooperative_groups.csv`

-   연결된 계정 그룹 정보
-   그룹별 PnL 통계
-   IP 공유 정보

### 3. `visualization_data.json`

-   시각화용 데이터
-   네트워크 그래프 데이터 (nodes, edges)
-   위험도 분포

### 4. `summary_report.txt`

-   텍스트 요약 보고서
-   상위 5개 그룹 정보

### 5. `sanction_groups.json`

-   제재 대상 그룹 목록
-   Critical 그룹 또는 IP 공유가 있는 High 그룹

## 결과 해석

### 점수 체계

#### PnL 비대칭성 (35점)

한쪽이 큰 이익을 보는 정도

-   ≥ 80% 비대칭: 35점
-   60-80%: 26점
-   40-60%: 18점
-   20-40%: 9점

#### 시간 근접도 (25점)

오픈/클로즈 평균 시간차

-   ≤ 5초: 25점
-   5-15초: 20점
-   15-30초: 15점
-   30-60초: 10점
-   60-120초: 5점

#### IP 공유 (25점)

두 계정 간 공유 IP 개수

-   ≥ 5개: 25점
-   3-4개: 20점
-   2개: 15점
-   1개: 10점

#### 포지션 겹침 (15점)

포지션 보유 시간 겹침 비율

-   ≥ 90%: 15점
-   70-90%: 11점
-   50-70%: 8점
-   < 50%: 4점

## 네트워크 분석

### Union-Find 알고리즘

-   거래 쌍에서 연결된 계정 그룹 탐지
-   A↔B, B↔C → {A, B, C} 그룹으로 병합

### 그룹 위험도 분류

-   평균 점수 + (공유 IP 수 × 5)
-   IP 공유가 많을수록 위험도 상승

### 제재 기준

-   Critical 그룹: 즉시 제재
-   High + IP 공유 1개 이상: 제재 대상

## 설정 파라미터

### Filter Parameters

-   `max_open_time_diff_min`: 최대 오픈 시간차 (기본값: 2.0분)
-   `max_close_time_diff_min`: 최대 클로즈 시간차 (기본값: 2.0분)
-   `exclude_major_symbols`: 주요 심볼 제외 (기본값: True)
-   `major_symbols`: 제외할 주요 심볼 리스트

### Scoring Weights

-   `weight_pnl_asymmetry`: PnL 비대칭성 가중치 (기본값: 35)
-   `weight_time_proximity`: 시간 근접도 가중치 (기본값: 25)
-   `weight_ip_sharing`: IP 공유 가중치 (기본값: 25)
-   `weight_position_overlap`: 포지션 겹침 가중치 (기본값: 15)

### Risk Thresholds

-   `critical_threshold`: Critical 최소 점수 (기본값: 85)
-   `high_threshold`: High 최소 점수 (기본값: 70)
-   `medium_threshold`: Medium 최소 점수 (기본값: 50)

### Network Analysis Parameters

-   `min_group_size`: 최소 그룹 크기 (기본값: 2)
-   `min_shared_ips`: 제재 최소 공유 IP 수 (기본값: 1)

## 아키텍처

```
CooperativeTradingDetector
├── DataLoader
├── CandidateExtractor
├── FilterEngine
├── ScoringEngine
├── NetworkAnalyzer
│   ├── find_groups() - Union-Find 알고리즘
│   └── analyze_shared_ips()
└── ReportGenerator
```

## 예제

### 탐지 결과 예시

```
공모거래 탐지 보고서
======================================================================

📊 탐지 요약
  - 총 의심 거래 쌍: 245건
  - Critical (확실한 공모): 32건
  - High (높은 의심): 48건
  - Medium (중간 의심): 87건
  - Low (낮은 의심): 78건

👥 그룹 분석
  - 탐지된 그룹: 15개
  - IP 공유 그룹: 8개
  - 총 순수익: $12,450.80

🎯 상위 그룹 (Top 5)
  1. GROUP_0001
     - 멤버: ACC001, ACC002, ACC003
     - 멤버 수: 3명
     - 거래 수: 28건
     - 순수익: $3,245.60
     - 위험도: CRITICAL
     - 공유 IP: 2개
```

## 네트워크 그래프 데이터

### 노드 (Nodes)

-   각 계정이 노드로 표현

### 엣지 (Edges)

-   거래 쌍이 엣지로 연결
-   속성: PnL, 위험도, 점수

### 시각화 예시

```json
{
    "nodes": [{ "id": "ACC001" }, { "id": "ACC002" }],
    "edges": [
        {
            "source": "ACC001",
            "target": "ACC002",
            "value": 1250.5,
            "risk_level": "CRITICAL",
            "score": 92.5
        }
    ]
}
```

## 의존성

-   pandas
-   duckdb
-   dataclasses (Python 3.7+)
-   typing
-   collections (Counter)
-   pathlib

## 라이선스

Singapore Fintech Hackathon Team
