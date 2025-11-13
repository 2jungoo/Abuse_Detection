# 증정금 녹이기 탐지 시스템 설계서

## 📌 개요

보너스 증정금을 악용한 무위험 현금화(Bonus Laundering) 행위를 탐지하는 2단계 시스템

## 🎯 탐지 목표

### 증정금 녹이기란?

-   두 개 이상의 계정에서 동일 상품에 대해 반대 포지션(Long/Short)을 동시에 개설
-   한쪽이 손실을 보면 보너스가 차감되고, 다른 쪽은 이익 획득
-   한쪽이 이익을 보면 다른 쪽은 손실이지만, 전체 원금은 유지
-   결과적으로 보너스만큼의 무위험 이익 실현

## 🔍 탐지 전략

### 2-Tier Detection System

#### Tier 1: 봇 기반 악의적 거래 (Bot-Driven Abuse)

-   **특징**: 완벽한 타이밍, 수량, 레버리지 매칭
-   **처리**: 즉시 제재 파이프라인 전송
-   **기준**: 매우 엄격 (점수 >= 90)

#### Tier 2: 수동 기반 악의적 거래 (Manual Abuse)

-   **특징**: 느슨한 매칭이지만 증정금 녹이기 패턴 존재
-   **처리**: 수익 계정 추적 및 네트워킹 분석
-   **기준**: 상대적으로 느슨 (점수 >= 70)
-   **네트워킹**:
    -   동일 수익 계정이 2회 이상 등장
    -   A→B→C 형태의 연결된 계정 체인 발견
    -   발견 시 제재 파이프라인 전송

## 📊 탐지 프로세스

### Phase 1: Filter (필수 조건 선별)

| 지표                 | 조건         | 설명                                |
| -------------------- | ------------ | ----------------------------------- |
| **Time Since Bonus** | 72시간 이내  | 보너스 수령 후 거래 개시까지의 시간 |
| **Reverse Position** | Long ↔ Short | 반대 방향 포지션                    |
| **Equal Leverage**   | 완전 동일    | 레버리지 배율 일치                  |
| **Concurrency**      | 30초 이내    | 거래 시간 동기화                    |
| **Quantity Match**   | ±2% 이내     | 거래 수량 근접성                    |

**※ 하나라도 불충족 시 탐지 대상에서 제외**

### Phase 2: Scoring (의심도 점수화)

#### 점수 체계 (100점 만점)

| 지표                    | 배점 | 계산 방식                                                                                                         |
| ----------------------- | ---- | ----------------------------------------------------------------------------------------------------------------- |
| **P&L Mirroring**       | 40점 | `abs(PnL_A + PnL_B) / max(abs(PnL_A), abs(PnL_B))` 기반<br>완벽한 대칭(≈0): 40점<br>중간(≈0.1): 20점<br>낮음: 0점 |
| **High Concurrency**    | 25점 | 시간차 기반<br>0.1초 이내: 25점<br>1초 이내: 20점<br>10초 이내: 10점<br>30초 이내: 5점                            |
| **High Quantity Match** | 20점 | 수량 차이 기반<br>0.1% 이내: 20점<br>0.5% 이내: 15점<br>1% 이내: 10점<br>2% 이내: 5점                             |
| **Trade Value Ratio**   | 15점 | `거래증거금 / (입금액 + 보너스)`<br>95% 이상: 15점<br>80% 이상: 10점<br>50% 이상: 5점                             |

#### 판정 기준

-   **Bot Tier (90점 이상)**: 즉시 제재
-   **Manual Tier (70~89점)**: 수익 계정 추적
-   **Suspicious (50~69점)**: 모니터링 대상
-   **Normal (50점 미만)**: 정상 거래

## 🏗️ 시스템 아키텍처

### 모듈 구조

```
newWashTrading.py
├── 1. Configuration (설정)
│   ├── DetectionConfig (하이퍼파라미터)
│   └── SanctionThresholds (제재 기준)
│
├── 2. Data Models (데이터 타입)
│   ├── TradePair (거래 쌍)
│   ├── SuspiciousAccount (의심 계정)
│   ├── SanctionCase (제재 사례)
│   └── NetworkNode (네트워크 노드)
│
├── 3. Data Pipeline
│   ├── DataLoader (데이터 로드)
│   └── PositionBuilder (포지션 구성)
│
├── 4. Detection Engine
│   ├── FilterEngine (1단계: 필수 조건 검사)
│   ├── ScoringEngine (2단계: 점수 계산)
│   └── TierClassifier (Tier 분류)
│
├── 5. Network Analyzer
│   ├── ProfitAccountTracker (수익 계정 추적)
│   └── NetworkGraphBuilder (네트워크 그래프 구성)
│
├── 6. Sanction Pipeline
│   ├── SanctionCaseBuilder (제재 케이스 생성)
│   └── SanctionExporter (제재 데이터 출력)
│
└── 7. Reporting & Logging
    ├── Logger (상세 로깅)
    ├── ReportGenerator (분석 보고서)
    └── VisualizationDataExporter (시각화 데이터)
```

## 📤 출력 데이터

### 1. 제재 파이프라인 데이터

```json
{
  "sanction_type": "BOT" | "NETWORK",
  "account_ids": [...],
  "detection_timestamp": "ISO8601",
  "evidence": {
    "trade_pairs": [...],
    "total_score": 95,
    "tier": "BOT",
    "network_path": ["A", "B", "C"]  // NETWORK 타입만
  }
}
```

### 2. 시각화 데이터

-   거래 쌍 상세 정보 (CSV/JSON)
-   점수 분포 데이터
-   시간대별 패턴
-   네트워크 그래프 데이터 (nodes, edges)

### 3. 로그 파일

-   단계별 상세 로그
-   필터링 통과/실패 로그
-   점수 계산 상세 로그
-   네트워크 분석 로그

## 🔧 하이퍼파라미터

### Filter Parameters

-   `time_since_bonus_hours`: 72
-   `concurrency_threshold_sec`: 30
-   `quantity_tolerance_pct`: 0.02

### Scoring Parameters

-   `bot_tier_threshold`: 90
-   `manual_tier_threshold`: 70
-   `suspicious_threshold`: 50

### Network Parameters

-   `min_profit_occurrences`: 2
-   `max_network_depth`: 5

## 📈 성능 고려사항

-   DuckDB를 활용한 빠른 쿼리
-   단계별 필터링으로 처리량 최소화
-   배치 처리 지원
-   메모리 효율적 네트워크 분석

## 🚀 개발 단계

1. ✅ 분석 및 설계
2. ⏳ 데이터 모델 및 설정 구현
3. ⏳ 필터 엔진 구현
4. ⏳ 점수 엔진 구현
5. ⏳ 네트워크 분석 구현
6. ⏳ 제재 파이프라인 구현
7. ⏳ 로깅 및 보고서 구현
8. ⏳ 통합 테스트 및 디버깅
9. ⏳ 최적화 및 개선
