# Cryptocurrency Fraud Detection System

암호화폐 거래소 부정거래 탐지 시스템 v2.0

## 개요

암호화폐 거래소에서 발생할 수 있는 다양한 부정거래 패턴을 자동으로 탐지하는 시스템입니다.

## 탐지 모듈

### 1. 🎁 Bonus Laundering (증정금 녹이기)

**위치**: `wash_trading/wash_trading.py`

보너스를 받은 계정과 일반 계정 간의 협력 거래를 통해 보너스를 현금화하는 패턴을 탐지합니다.

**특징**:

-   2-Tier 탐지 시스템 (Bot/Manual)
-   네트워크 분석 기반 연결 계정 탐지
-   반복 수익 계정 추적

[자세한 내용 →](./wash_trading/README_NEW.md)

---

### 2. 💰 Funding Hunter (펀딩비 악용)

**위치**: `funding_fee/funding_hunter.py`

펀딩비 정산 시점을 노린 고빈도 포지션 개폐 패턴을 탐지합니다.

**특징**:

-   4차원 점수 시스템
-   심각도별 분류 (Critical/High/Medium/Low)
-   계정별 누적 분석

[자세한 내용 →](./funding_fee/README.md)

---

### 3. 🤝 Cooperative Trading (공모거래)

**위치**: `abusing/cooperative_trading.py`

복수 계정 간 협력하여 부당 이득을 취하는 패턴을 탐지합니다.

**특징**:

-   Union-Find 기반 네트워크 그룹 탐지
-   IP 공유 분석
-   시간 근접도 및 PnL 비대칭성 분석

[자세한 내용 →](./abusing/README.md)

---

## 프로젝트 구조

```
Singapore-Dev/
├── wash_trading/              # 증정금 녹이기 탐지
│   ├── wash_trading.py        # 메인 탐지 엔진
│   ├── README_NEW.md          # 상세 문서
│   └── DESIGN.md              # 설계 문서
│
├── funding_fee/               # 펀딩비 악용 탐지
│   ├── funding_hunter.py      # 메인 탐지 엔진
│   └── README.md              # 상세 문서
│
├── abusing/                   # 공모거래 탐지
│   ├── cooperative_trading.py # 메인 탐지 엔진
│   └── README.md              # 상세 문서
│
├── output/                    # 출력 디렉토리
│   ├── bonus/                 # 증정금 탐지 결과
│   ├── funding_fee/           # 펀딩비 탐지 결과
│   └── cooperative/           # 공모거래 탐지 결과
│
├── common/                    # 공통 유틸리티
├── Anomaly_Detection.py       # 레거시 통합 스크립트
└── README.md                  # 이 파일
```

## 빠른 시작

### 1. Python 가상환경(venv) 생성

```bash
python3 -m venv venv
```

### 2. 가상환경 활성화

-   **macOS/Linux**:
    ```bash
    source venv/bin/activate
    ```
-   **Windows**:
    ```bash
    .\venv\Scripts\activate
    ```

### 3. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

또는 개별 설치:

```bash
pip install pandas duckdb openpyxl
```

### 개별 모듈 실행

#### 1. Bonus Laundering 탐지

```python
from wash_trading.wash_trading import run_detection

result = run_detection("problem_data_final.xlsx")
```

#### 2. Funding Hunter 탐지

```python
from funding_fee.funding_hunter import run_detection

result = run_detection("problem_data_final.xlsx")
```

#### 3. Cooperative Trading 탐지

```python
from abusing.cooperative_trading import run_detection

result = run_detection("problem_data_final.xlsx")
```

### 전체 탐지 실행

레거시 방식 (모든 탐지 한번에):

```bash
python Anomaly_Detection.py
```

## 공통 설계 원칙

### 1. 모듈화된 아키텍처

각 탐지 모듈은 독립적으로 실행 가능하며, 다음과 같은 공통 구조를 따릅니다:

```
Detector
├── DataLoader        # 데이터 로딩
├── Extractor         # 후보 추출
├── FilterEngine      # 필수 조건 필터링
├── ScoringEngine     # 점수 계산
├── Analyzer          # 추가 분석 (네트워크 등)
└── ReportGenerator   # 보고서 생성
```

### 2. 설정 기반 시스템

모든 탐지 파라미터는 `DetectionConfig` 클래스로 관리:

-   필터 조건
-   점수 가중치
-   임계값
-   출력 설정

### 3. 다단계 탐지 프로세스

```
1. 데이터 로드
   ↓
2. 후보 추출 (SQL)
   ↓
3. 필수 조건 필터링
   ↓
4. 점수 계산 및 분류
   ↓
5. 추가 분석 (네트워크 등)
   ↓
6. 보고서 생성
```

### 4. 풍부한 출력 형식

-   **CSV**: 상세 데이터 분석용
-   **JSON**: 시각화 및 API 연동용
-   **TXT**: 사람이 읽을 수 있는 요약 보고서
-   **로그**: 디버깅 및 추적용

## 출력 데이터 구조

### 공통 출력 파일

모든 모듈은 다음과 같은 파일을 생성합니다:

1. **`*_detailed.csv`**: 탐지 케이스 상세 정보
2. **`visualization_data.json`**: 시각화용 데이터
3. **`summary_report.txt`**: 텍스트 요약 보고서
4. **`detection_config.json`**: 사용된 설정 값
5. **`detection_*.log`**: 실행 로그

## 점수 시스템 비교

| 모듈                | 최고점 | 주요 지표                                          |
| ------------------- | ------ | -------------------------------------------------- |
| Bonus Laundering    | 100점  | PnL 대칭(40), 동시성(25), 수량(20), 비율(15)       |
| Funding Hunter      | 100점  | 펀딩비(40), 보유시간(25), 레버리지(20), 포지션(15) |
| Cooperative Trading | 100점  | PnL 비대칭(35), 시간(25), IP(25), 겹침(15)         |

## 위험도/심각도 분류

모든 모듈은 탐지 결과를 4단계로 분류합니다:

-   **CRITICAL/BOT** (≥85점): 즉시 제재 대상
-   **HIGH/MANUAL** (70-84점): 수동 검토 후 제재
-   **MEDIUM/SUSPICIOUS** (50-69점): 모니터링 대상
-   **LOW/NORMAL** (<50점): 정상 또는 낮은 의심

## 성능 최적화

### DuckDB 활용

-   SQL 기반 대용량 데이터 처리
-   메모리 효율적인 쿼리 실행
-   복잡한 조인 및 집계 최적화

### 병렬 처리 가능

각 모듈은 독립적으로 실행 가능하므로 병렬 처리 가능:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=3) as executor:
    bonus_future = executor.submit(run_bonus_detection)
    funding_future = executor.submit(run_funding_detection)
    coop_future = executor.submit(run_coop_detection)

    results = [f.result() for f in [bonus_future, funding_future, coop_future]]
```

## 데이터 요구사항

### 필수 시트

-   **Trade**: 거래 내역
-   **Funding**: 펀딩비 내역
-   **Reward**: 보너스 지급 내역
-   **IP**: IP 접속 기록
-   **Spec**: 거래 상품 사양

### Trade 테이블 필수 컬럼

-   `account_id`: 계정 ID
-   `position_id`: 포지션 ID
-   `ts`: 타임스탬프
-   `symbol`: 거래 심볼
-   `side`: LONG/SHORT
-   `openclose`: OPEN/CLOSE
-   `amount`: 거래량
-   `leverage`: 레버리지

## 확장 가능성

### 새로운 탐지 모듈 추가

각 탐지 모듈은 표준화된 인터페이스를 따르므로 쉽게 확장 가능:

```python
class NewDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger(config)

    def detect(self, data_filepath: str) -> Dict:
        # 1. 데이터 로드
        # 2. 후보 추출
        # 3. 필터링
        # 4. 점수 계산
        # 5. 보고서 생성
        return result
```

## 의존성

```
pandas>=1.5.0
duckdb>=0.8.0
openpyxl>=3.0.0
```

## 개발 팀

Singapore Fintech Hackathon Team

## 라이선스

본 프로젝트는 Singapore Fintech Hackathon 2025 출품작입니다.

## 참고 자료

-   [Bonus Laundering 설계 문서](./wash_trading/DESIGN.md)
-   [Bonus Laundering 완료 보고서](./wash_trading/COMPLETION_REPORT.md)
-   각 모듈별 README 파일 참조

---

**버전**: 2.0  
**최종 업데이트**: 2025-11-13
