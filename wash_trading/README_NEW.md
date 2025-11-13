# 증정금 녹이기 탐지 시스템 v2.0

보너스 증정금을 악용한 무위험 현금화(Bonus Laundering) 행위를 탐지하는 2-Tier 시스템입니다.

## 🎯 주요 기능

### 2-Tier 탐지 시스템

#### Tier 1: Bot 기반 악의적 거래

-   **특징**: 완벽한 타이밍, 수량, 레버리지 매칭
-   **점수**: 90점 이상
-   **처리**: 즉시 제재 파이프라인 전송

#### Tier 2: 수동 기반 악의적 거래

-   **특징**: 느슨한 매칭이지만 증정금 녹이기 패턴 존재
-   **점수**: 70-89점
-   **처리**: 수익 계정 추적 및 네트워킹 분석
    -   동일 수익 계정 2회 이상 등장 → 제재
    -   A→B→C 형태 연결 체인 발견 → 제재

## 📊 탐지 프로세스

### Phase 1: Filter (필수 조건 선별)

| 지표             | 조건         | 설명                                |
| ---------------- | ------------ | ----------------------------------- |
| Time Since Bonus | 72시간 이내  | 보너스 수령 후 거래 개시까지의 시간 |
| Reverse Position | Long ↔ Short | 반대 방향 포지션                    |
| Equal Leverage   | 완전 동일    | 레버리지 배율 일치                  |
| Concurrency      | 30초 이내    | 거래 시간 동기화                    |
| Quantity Match   | ±2% 이내     | 거래 수량 근접성                    |

### Phase 2: Scoring (점수화)

| 지표                | 배점 | 설명                                |
| ------------------- | ---- | ----------------------------------- |
| P&L Mirroring       | 40점 | 손익 대칭성 (PnL_A + PnL_B ≈ 0)     |
| High Concurrency    | 25점 | 시간 근접도 (0초에 가까울수록 높음) |
| High Quantity Match | 20점 | 수량 일치도 (0%에 가까울수록 높음)  |
| Trade Value Ratio   | 15점 | 보너스 대비 거래액 비율             |

**총점 100점 만점**

### Tier 분류

-   **Bot (90점 이상)**: 즉시 제재
-   **Manual (70-89점)**: 네트워킹 분석
-   **Suspicious (50-69점)**: 모니터링
-   **Normal (50점 미만)**: 정상 거래

## 🚀 사용법

### 기본 실행

```python
from newWashTrading import run_detection

result = run_detection(data_filepath="problem_data_final.xlsx")
```

### 커스텀 설정

```python
from newWashTrading import run_detection, DetectionConfig

# 설정 커스터마이즈
config = DetectionConfig(
    # Filter 파라미터
    time_since_bonus_hours=72.0,      # 보너스 후 시간 창
    concurrency_threshold_sec=30.0,    # 동시성 임계값
    quantity_tolerance_pct=0.02,       # 수량 허용 오차 (2%)

    # Tier 임계값
    bot_tier_threshold=90,             # Bot 판정 점수
    manual_tier_threshold=70,          # Manual 판정 점수

    # 네트워크 파라미터
    min_profit_occurrences=2,          # 제재 대상 최소 수익 횟수

    # 출력 설정
    output_dir="./output/bonus",
    enable_detailed_logging=True
)

result = run_detection(
    data_filepath="problem_data_final.xlsx",
    config=config
)
```

### 커맨드라인 실행

```bash
python newWashTrading.py
```

## 📤 출력 파일

모든 출력은 `output/bonus/` 디렉토리에 저장됩니다:

### 1. 제재 파이프라인 데이터

**파일**: `sanction_cases.json`

```json
{
    "total_cases": 5,
    "generated_at": "2025-11-13T10:30:00",
    "cases": [
        {
            "case_id": "SANCTION_BOT_PAIR_000001",
            "sanction_type": "IMMEDIATE_BOT",
            "account_ids": ["ACC_001", "ACC_002"],
            "detection_timestamp": "2025-11-13T10:30:00",
            "trade_pair_ids": ["PAIR_000001"],
            "total_score": 95.5,
            "tier": "BOT",
            "total_laundered_amount": 1500.0,
            "evidence_summary": "완벽한 봇 패턴 탐지 (점수: 95.5/100)"
        }
    ]
}
```

### 2. 거래 쌍 상세 데이터

**파일**: `trade_pairs_detailed.csv`

| pair_id     | tier | total_score | loser_account | winner_account | symbol  | loser_pnl | winner_pnl | ... |
| ----------- | ---- | ----------- | ------------- | -------------- | ------- | --------- | ---------- | --- |
| PAIR_000001 | BOT  | 95.5        | ACC_001       | ACC_002        | BTCUSDT | -1500     | 1500       | ... |

### 3. 시각화 데이터

**파일**: `visualization_data.json`

```json
{
  "summary": {
    "total_pairs": 100,
    "bot_tier": 5,
    "manual_tier": 15,
    "suspicious": 30,
    "normal": 50,
    "total_sanctions": 8
  },
  "tier_distribution": {...},
  "score_distribution": {...},
  "time_patterns": {...},
  "network_graph": {
    "nodes": [...],
    "edges": [...]
  },
  "network_statistics": {...}
}
```

### 4. 요약 보고서

**파일**: `summary_report.txt`

```
======================================================================
증정금 녹이기 탐지 보고서
======================================================================

📊 탐지 요약
  - 총 분석 거래 쌍: 100건
  - Bot Tier (즉시 제재): 5건
  - Manual Tier (네트워크 분석): 15건
  ...
```

### 5. 탐지 로그

**파일**: `detection_YYYYMMDD_HHMMSS.log`

상세한 탐지 과정 로그

### 6. 설정 파일

**파일**: `detection_config.json`

실행 시 사용된 모든 파라미터

## 🔧 하이퍼파라미터 튜닝

### Filter 파라미터

```python
config = DetectionConfig(
    # 더 엄격하게 (봇만 잡기)
    concurrency_threshold_sec=5.0,     # 5초 이내만
    quantity_tolerance_pct=0.005,      # 0.5% 이내만

    # 또는 더 느슨하게 (수동 포함)
    concurrency_threshold_sec=60.0,    # 1분 이내
    quantity_tolerance_pct=0.05,       # 5% 이내
)
```

### Scoring 파라미터

```python
config = DetectionConfig(
    # 점수 배점 조정
    weight_pnl_mirroring=50,      # P&L 대칭성 강조
    weight_high_concurrency=30,    # 시간 강조
    weight_quantity_match=15,
    weight_trade_value_ratio=5,
)
```

### Tier 임계값

```python
config = DetectionConfig(
    bot_tier_threshold=95,        # 더 엄격하게
    manual_tier_threshold=75,
    suspicious_threshold=60,
)
```

## 📈 네트워크 분석

### 수익 계정 추적

-   Manual Tier 거래에서 수익을 얻은 계정 추적
-   동일 계정이 여러 번 수익 → 제재 대상

### 계정 체인 탐지

-   A가 B에게 손실, B가 C에게 손실 → A-B-C 체인
-   2개 이상 연결된 체인 발견 시 제재

## 🎨 시각화 연동

`visualization_data.json`을 웹 대시보드에서 사용:

```javascript
// 예시: 네트워크 그래프 렌더링
fetch('output/bonus/visualization_data.json')
    .then((res) => res.json())
    .then((data) => {
        renderNetworkGraph(data.network_graph)
        renderScoreDistribution(data.score_distribution)
    })
```

## 📝 제재 파이프라인 연동

```python
# sanction_cases.json 읽기
import json

with open('output/bonus/sanction_cases.json') as f:
    sanctions = json.load(f)

for case in sanctions['cases']:
    if case['sanction_type'] == 'IMMEDIATE_BOT':
        # 즉시 계정 정지
        suspend_accounts(case['account_ids'])
    elif case['sanction_type'] in ['NETWORK_REPEAT', 'NETWORK_CHAIN']:
        # 추가 조사 후 제재
        investigate_and_sanction(case)
```

## 🐛 디버깅

### 상세 로그 활성화

```python
config = DetectionConfig(
    enable_detailed_logging=True
)
```

### 특정 계정 추적

로그 파일에서 계정 ID로 검색:

```bash
grep "ACC_001" output/bonus/detection_*.log
```

## 📊 성능 최적화

-   DuckDB 기반 빠른 SQL 쿼리
-   단계별 필터링으로 처리량 최소화
-   메모리 효율적 네트워크 분석

## 🔒 제재 유형

### SanctionType.IMMEDIATE_BOT

완벽한 봇 패턴 → 즉시 제재

### SanctionType.NETWORK_REPEAT

동일 계정 반복 수익 → 제재

### SanctionType.NETWORK_CHAIN

연결된 계정 체인 → 제재

## 📚 참고 자료

-   `DESIGN.md`: 전체 시스템 설계 문서
-   `기준.md`: 탐지 지표 및 파라미터 근거
-   `증정금녹이기.md`: 증정금 녹이기 개념 설명

## 🤝 기여

버그 리포트 및 개선 제안은 이슈로 등록해주세요.

## 📜 라이선스

Singapore Fintech Hackathon 2025
