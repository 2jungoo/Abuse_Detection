"""
Funding Hunter Detection System (펀딩비 악용 탐지 시스템)
Version: 2.0
Author: Singapore Fintech Hackathon Team

탐지 대상: 펀딩비 정산 시점을 노린 고빈도 포지션 개폐 패턴
"""

import pandas as pd
import duckdb as dd
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
from common.data_manager import get_data_manager

# ============================================================================
# 1. CONFIGURATION & TYPES
# ============================================================================

class SeverityLevel(Enum):
    """탐지 심각도"""
    CRITICAL = "CRITICAL"      # 확실한 악용
    HIGH = "HIGH"              # 높은 의심
    MEDIUM = "MEDIUM"          # 중간 의심
    LOW = "LOW"                # 낮은 의심


@dataclass
class DetectionConfig:
    """탐지 설정 및 하이퍼파라미터"""
    
    # ===== Filter Parameters (필수 조건) =====
    min_leverage: int = 5                          # 최소 레버리지
    min_amount_ratio: float = 0.3                  # 최소 거래량 비율 (최대 주문량 대비)
    max_holding_minutes: float = 20.0              # 최대 보유 시간 (분)
    require_hour_change: bool = True               # 오픈/클로즈 시간대 변경 필수
    
    # ===== Scoring Weights (점수 배점) =====
    weight_funding_profit: int = 40                # 펀딩비 수익
    weight_short_holding: int = 25                 # 짧은 보유 시간
    weight_high_leverage: int = 20                 # 높은 레버리지
    weight_large_position: int = 15                # 큰 포지션 크기
    
    # ===== Severity Thresholds (심각도 판정 기준) =====
    critical_threshold: int = 85                   # Critical 최소 점수
    high_threshold: int = 70                       # High 최소 점수
    medium_threshold: int = 50                     # Medium 최소 점수
    
    # ===== Output Settings =====
    output_dir: str = "output/funding_fee"
    enable_detailed_logging: bool = True
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)
    
    def save(self, filepath: str):
        """JSON 파일로 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"설정 저장 완료: {filepath}")


@dataclass
class ScoreBreakdown:
    """점수 상세 정보"""
    funding_profit: float = 0.0
    short_holding: float = 0.0
    high_leverage: float = 0.0
    large_position: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FundingHunterCase:
    """펀딩 헌터 케이스"""
    case_id: str
    account_id: str
    position_id: str
    
    # 거래 정보
    symbol: str
    side: str
    leverage: int
    amount: float
    
    # 시간 정보
    open_ts: datetime
    closing_ts: datetime
    holding_minutes: float
    
    # 펀딩 정보
    fund_period_hr: int
    closing_hour: int
    total_funding: float
    
    # 포지션 크기
    max_order_amount: float
    amount_ratio: float
    
    # 점수 및 판정
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    severity: SeverityLevel = SeverityLevel.LOW
    
    # 필터 통과 여부
    passed_filter: bool = False
    filter_failures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['open_ts'] = self.open_ts.isoformat()
        data['closing_ts'] = self.closing_ts.isoformat()
        return data


@dataclass
class AccountSummary:
    """계정별 요약 정보"""
    account_id: str
    total_cases: int = 0
    total_funding_profit: float = 0.0
    avg_score: float = 0.0
    max_score: float = 0.0
    critical_count: int = 0
    high_count: int = 0
    case_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 2. LOGGING SETUP
# ============================================================================

class DetectionLogger:
    """탐지 시스템 전용 로거"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로그 파일 경로
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"detection_{timestamp}.log"
        
        # 로거 설정
        logging.basicConfig(
            level=logging.DEBUG if config.enable_detailed_logging else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("펀딩 헌터 탐지 시스템 시작")
        self.logger.info("="*70)
    
    def log_phase(self, phase_name: str):
        """단계 시작 로그"""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"PHASE: {phase_name}")
        self.logger.info("="*70)
    
    def log_filter_result(self, total: int, passed: int, failed: int):
        """필터 결과 로그"""
        self.logger.info(f"필터 결과: 총 {total}건 → 통과 {passed}건, 실패 {failed}건")
    
    def log_severity_distribution(self, severity_counts: Dict[SeverityLevel, int]):
        """심각도 분포 로그"""
        self.logger.info("심각도 분포:")
        for severity, count in severity_counts.items():
            self.logger.info(f"  - {severity.value}: {count}건")
    
    def log_hunter_case(self, case: FundingHunterCase):
        """펀딩 헌터 케이스 로그"""
        self.logger.warning(f"펀딩 헌터 탐지: {case.case_id}")
        self.logger.warning(f"  - 계정: {case.account_id}")
        self.logger.warning(f"  - 심각도: {case.severity.value}")
        self.logger.warning(f"  - 점수: {case.score.total:.2f}")
        self.logger.warning(f"  - 펀딩비 수익: ${case.total_funding:.2f}")


# ============================================================================
# 3. DATA PIPELINE
# ============================================================================
# DataLoader removed - using common.data_manager.DataManager singleton instead.


# ============================================================================
# 4. CANDIDATE EXTRACTOR
# ============================================================================

class CandidateExtractor:
    """후보 케이스 추출"""
    
    def __init__(self, con: dd.DuckDBPyConnection):
        self.con = con
    
    def extract_candidates(self) -> List[Dict]:
        """SQL을 통해 후보 케이스 추출"""
        print("후보 케이스 추출 중...")
        
        query = """
        WITH spec_clean AS (
            SELECT
                symbol,
                CAST(funding_interval AS INTEGER) AS fund_period_hr,
                max_order_amount,
                CAST(day AS DATE) AS spec_day
            FROM Spec
        ),
        position AS (
            SELECT
                account_id,
                position_id,
                MAX(leverage) AS leverage,
                CAST(MIN(ts) AS TIMESTAMP) as open_ts,
                CAST(MAX(ts) AS TIMESTAMP) as closing_ts,
                MAX(symbol) as symbol,
                MAX(side) as side,
                DATE(MAX(ts)) as closing_day,
                SUM(CASE WHEN openclose='OPEN' THEN amount ELSE 0 END) as amount
            FROM Trade
            GROUP BY account_id, position_id
        ),
        funding_agg AS (
            SELECT
                account_id,
                -SUM(funding_fee) AS total_funding
            FROM Funding
            GROUP BY account_id
        ),
        joined AS (
            SELECT
                ct.account_id,
                ct.symbol,
                ct.position_id,
                ct.side,
                ct.open_ts,
                ct.closing_ts,
                ct.leverage,
                ct.amount,
                fa.total_funding,
                (epoch(ct.closing_ts) - epoch(ct.open_ts)) / 60.0 AS holding_minutes,
                sc.fund_period_hr,
                sc.max_order_amount,
                CAST(STRFTIME('%H', ct.closing_ts) AS INTEGER) AS closing_hour,
                CAST(STRFTIME('%H', ct.open_ts) AS INTEGER) AS opening_hour
            FROM position ct
            LEFT JOIN funding_agg fa ON ct.account_id = fa.account_id
            LEFT JOIN spec_clean sc
                ON ct.symbol = sc.symbol AND ct.closing_day = sc.spec_day
        )
        SELECT 
            account_id,
            position_id,
            symbol,
            side,
            open_ts,
            closing_ts,
            leverage,
            amount,
            total_funding,
            holding_minutes,
            fund_period_hr,
            max_order_amount,
            closing_hour,
            opening_hour,
            amount / NULLIF(max_order_amount, 0) AS amount_ratio
        FROM joined
        WHERE 
            total_funding > 0
            AND fund_period_hr IS NOT NULL
            AND max_order_amount IS NOT NULL
            AND closing_hour % fund_period_hr = 0
        ORDER BY total_funding DESC, holding_minutes ASC
        """
        
        df = self.con.execute(query).fetchdf()
        print(f"후보 케이스 {len(df)}개 추출")
        
        return df.to_dict('records')


# ============================================================================
# 5. FILTER ENGINE
# ============================================================================

class FilterEngine:
    """필수 조건 필터링"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def apply_filters(self, candidates: List[Dict]) -> tuple[List[FundingHunterCase], List[Dict]]:
        """필터 적용 및 FundingHunterCase 객체 생성"""
        print("필터 엔진 시작...")
        
        passed_cases = []
        failed_data = []
        
        for idx, row in enumerate(candidates):
            case_id = f"FUND_{idx:06d}"
            failures = []
            
            # Filter 1: Minimum Leverage
            if not self._check_min_leverage(row):
                failures.append("min_leverage")
            
            # Filter 2: Minimum Amount Ratio
            if not self._check_min_amount_ratio(row):
                failures.append("min_amount_ratio")
            
            # Filter 3: Maximum Holding Time
            if not self._check_max_holding_time(row):
                failures.append("max_holding_time")
            
            # Filter 4: Hour Change Required
            if self.config.require_hour_change and not self._check_hour_change(row):
                failures.append("hour_change")
            
            # 모든 필터 통과 여부
            if len(failures) == 0:
                # FundingHunterCase 객체 생성
                hunter_case = self._create_hunter_case(case_id, row)
                hunter_case.passed_filter = True
                passed_cases.append(hunter_case)
            else:
                # 실패 정보 기록
                row['case_id'] = case_id
                row['filter_failures'] = failures
                failed_data.append(row)
                
                # if self.config.enable_detailed_logging:
                #     print(f"{case_id} 필터 실패: {', '.join(failures)}")
        
        print(f"필터 완료: {len(passed_cases)}/{len(candidates)} 통과")
        
        return passed_cases, failed_data
    
    def _check_min_leverage(self, row: Dict) -> bool:
        """최소 레버리지 확인"""
        return row.get('leverage', 0) >= self.config.min_leverage
    
    def _check_min_amount_ratio(self, row: Dict) -> bool:
        """최소 거래량 비율 확인"""
        ratio = row.get('amount_ratio', 0)
        return ratio >= self.config.min_amount_ratio
    
    def _check_max_holding_time(self, row: Dict) -> bool:
        """최대 보유 시간 확인"""
        minutes = row.get('holding_minutes', float('inf'))
        return minutes <= self.config.max_holding_minutes
    
    def _check_hour_change(self, row: Dict) -> bool:
        """시간대 변경 확인"""
        opening_hour = row.get('opening_hour', -1)
        closing_hour = row.get('closing_hour', -1)
        return opening_hour != closing_hour
    
    def _create_hunter_case(self, case_id: str, row: Dict) -> FundingHunterCase:
        """FundingHunterCase 객체 생성"""
        return FundingHunterCase(
            case_id=case_id,
            account_id=row['account_id'],
            position_id=row['position_id'],
            symbol=row['symbol'],
            side=row['side'],
            leverage=row['leverage'],
            amount=row['amount'],
            open_ts=row['open_ts'],
            closing_ts=row['closing_ts'],
            holding_minutes=row['holding_minutes'],
            fund_period_hr=row['fund_period_hr'],
            closing_hour=row['closing_hour'],
            total_funding=row['total_funding'],
            max_order_amount=row['max_order_amount'],
            amount_ratio=row['amount_ratio'],
        )


# ============================================================================
# 6. SCORING ENGINE
# ============================================================================

class ScoringEngine:
    """점수 계산 및 심각도 분류"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def score_all_cases(self, cases: List[FundingHunterCase]) -> List[FundingHunterCase]:
        """모든 케이스 점수 계산"""
        print("점수 엔진 시작...")
        
        for case in cases:
            # 각 지표별 점수 계산
            funding_score = self._score_funding_profit(case)
            holding_score = self._score_short_holding(case)
            leverage_score = self._score_high_leverage(case)
            position_score = self._score_large_position(case)
            
            # 점수 저장
            case.score = ScoreBreakdown(
                funding_profit=funding_score,
                short_holding=holding_score,
                high_leverage=leverage_score,
                large_position=position_score,
                total=funding_score + holding_score + leverage_score + position_score
            )
            
            # 심각도 분류
            case.severity = self._classify_severity(case.score.total)
            
            # if self.config.enable_detailed_logging:
            #     print(
            #         f"{case.case_id}: 점수={case.score.total:.1f} "
            #         f"(Funding:{funding_score:.1f}, Holding:{holding_score:.1f}, "
            #         f"Leverage:{leverage_score:.1f}, Position:{position_score:.1f}) "
            #         f"→ {case.severity.value}"
            #     )
        
        print(f"점수 계산 완료: {len(cases)}개")
        
        return cases
    
    def _score_funding_profit(self, case: FundingHunterCase) -> float:
        """펀딩비 수익 점수 (40점)"""
        max_weight = self.config.weight_funding_profit
        profit = case.total_funding
        
        if profit >= 1000:
            return max_weight
        elif profit >= 500:
            return max_weight * 0.85
        elif profit >= 200:
            return max_weight * 0.65
        elif profit >= 100:
            return max_weight * 0.45
        elif profit >= 50:
            return max_weight * 0.25
        else:
            return max_weight * 0.10
    
    def _score_short_holding(self, case: FundingHunterCase) -> float:
        """짧은 보유 시간 점수 (25점)"""
        max_weight = self.config.weight_short_holding
        minutes = case.holding_minutes
        
        if minutes <= 5:
            return max_weight
        elif minutes <= 10:
            return max_weight * 0.80
        elif minutes <= 15:
            return max_weight * 0.60
        else:  # <= 20
            return max_weight * 0.35
    
    def _score_high_leverage(self, case: FundingHunterCase) -> float:
        """높은 레버리지 점수 (20점)"""
        max_weight = self.config.weight_high_leverage
        leverage = case.leverage
        
        if leverage >= 20:
            return max_weight
        elif leverage >= 15:
            return max_weight * 0.80
        elif leverage >= 10:
            return max_weight * 0.60
        else:  # >= 5
            return max_weight * 0.35
    
    def _score_large_position(self, case: FundingHunterCase) -> float:
        """큰 포지션 크기 점수 (15점)"""
        max_weight = self.config.weight_large_position
        ratio = case.amount_ratio
        
        if ratio >= 0.8:
            return max_weight
        elif ratio >= 0.6:
            return max_weight * 0.75
        elif ratio >= 0.5:
            return max_weight * 0.50
        else:  # >= 0.3
            return max_weight * 0.25
    
    def _classify_severity(self, total_score: float) -> SeverityLevel:
        """점수 기반 심각도 분류"""
        if total_score >= self.config.critical_threshold:
            return SeverityLevel.CRITICAL
        elif total_score >= self.config.high_threshold:
            return SeverityLevel.HIGH
        elif total_score >= self.config.medium_threshold:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


# ============================================================================
# 7. ACCOUNT ANALYZER
# ============================================================================

class AccountAnalyzer:
    """계정별 분석"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def analyze_accounts(self, cases: List[FundingHunterCase]) -> Dict[str, AccountSummary]:
        """계정별 요약 생성"""
        print("계정별 분석 중...")
        
        account_map = {}
        
        for case in cases:
            account_id = case.account_id
            
            if account_id not in account_map:
                account_map[account_id] = AccountSummary(account_id=account_id)
            
            summary = account_map[account_id]
            summary.total_cases += 1
            summary.total_funding_profit += case.total_funding
            summary.case_ids.append(case.case_id)
            
            if case.severity == SeverityLevel.CRITICAL:
                summary.critical_count += 1
            elif case.severity == SeverityLevel.HIGH:
                summary.high_count += 1
            
            # 점수 누적
            if summary.total_cases == 1:
                summary.avg_score = case.score.total
                summary.max_score = case.score.total
            else:
                summary.avg_score = (
                    (summary.avg_score * (summary.total_cases - 1) + case.score.total) 
                    / summary.total_cases
                )
                summary.max_score = max(summary.max_score, case.score.total)
        
        print(f"계정 분석 완료: {len(account_map)}개 계정")
        
        return account_map


# ============================================================================
# 8. REPORTING & VISUALIZATION
# ============================================================================

class ReportGenerator:
    """분석 보고서 및 시각화 데이터 생성"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def generate_all_reports(
        self,
        all_cases: List[FundingHunterCase],
        account_summaries: Dict[str, AccountSummary]
    ):
        """모든 보고서 생성"""
        print("보고서 생성 중...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 케이스 상세 데이터 (CSV)
        self._export_cases_csv(all_cases)
        
        # 2. 계정 요약 (CSV)
        self._export_account_summary_csv(account_summaries)
        
        # 3. 시각화용 JSON 데이터
        self._export_visualization_data(all_cases, account_summaries)
        
        # 4. 요약 보고서 (텍스트)
        self._generate_summary_report(all_cases, account_summaries)
        
        # 5. 제재 대상 계정 (JSON)
        self._export_sanction_accounts(account_summaries)
        
        print("보고서 생성 완료")
    
    def _export_cases_csv(self, cases: List[FundingHunterCase]):
        """케이스 상세 CSV"""
        if not cases:
            return
        
        records = []
        for case in cases:
            record = {
                'case_id': case.case_id,
                'account_id': case.account_id,
                'position_id': case.position_id,
                'severity': case.severity.value,
                'total_score': case.score.total,
                'symbol': case.symbol,
                'side': case.side,
                'leverage': case.leverage,
                'amount': case.amount,
                'total_funding': case.total_funding,
                'holding_minutes': case.holding_minutes,
                'amount_ratio': case.amount_ratio,
                'fund_period_hr': case.fund_period_hr,
                'closing_hour': case.closing_hour,
                'score_funding': case.score.funding_profit,
                'score_holding': case.score.short_holding,
                'score_leverage': case.score.high_leverage,
                'score_position': case.score.large_position,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = self.output_dir / "funding_hunter_cases.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"케이스 CSV 저장: {filepath}")
    
    def _export_account_summary_csv(self, account_summaries: Dict[str, AccountSummary]):
        """계정 요약 CSV"""
        if not account_summaries:
            return
        
        records = [summary.to_dict() for summary in account_summaries.values()]
        df = pd.DataFrame(records)
        df = df.sort_values('total_funding_profit', ascending=False)
        
        filepath = self.output_dir / "account_summaries.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"계정 요약 CSV 저장: {filepath}")
    
    def _export_visualization_data(
        self, 
        cases: List[FundingHunterCase],
        account_summaries: Dict[str, AccountSummary]
    ):
        """시각화용 JSON 데이터"""
        from collections import defaultdict
        
        # 심각도별 분포
        severity_counts = defaultdict(int)
        for case in cases:
            severity_counts[case.severity.value] += 1
        
        # 점수 분포
        score_distribution = defaultdict(int)
        for case in cases:
            bucket = int(case.score.total // 10) * 10
            score_distribution[f"{bucket}-{bucket+10}"] += 1
        
        # 시간대별 패턴
        hourly_dist = defaultdict(int)
        for case in cases:
            hourly_dist[case.closing_hour] += 1
        
        # 심볼별 분포
        symbol_dist = defaultdict(int)
        for case in cases:
            symbol_dist[case.symbol] += 1
        
        vis_data = {
            'summary': {
                'total_cases': len(cases),
                'critical': severity_counts.get('CRITICAL', 0),
                'high': severity_counts.get('HIGH', 0),
                'medium': severity_counts.get('MEDIUM', 0),
                'low': severity_counts.get('LOW', 0),
                'total_accounts': len(account_summaries),
                'total_funding_profit': sum(s.total_funding_profit for s in account_summaries.values()),
            },
            'severity_distribution': dict(severity_counts),
            'score_distribution': dict(score_distribution),
            'hourly_distribution': dict(hourly_dist),
            'symbol_distribution': dict(symbol_dist),
        }
        
        filepath = self.output_dir / "visualization_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vis_data, f, indent=2, ensure_ascii=False)
        
        print(f"시각화 데이터 저장: {filepath}")
    
    def _generate_summary_report(
        self,
        cases: List[FundingHunterCase],
        account_summaries: Dict[str, AccountSummary]
    ):
        """요약 보고서 텍스트"""
        from collections import defaultdict
        
        severity_counts = defaultdict(int)
        for case in cases:
            severity_counts[case.severity] += 1
        
        total_funding = sum(s.total_funding_profit for s in account_summaries.values())
        
        # 상위 계정
        top_accounts = sorted(
            account_summaries.values(), 
            key=lambda x: x.total_funding_profit, 
            reverse=True
        )[:10]
        
        report = f"""
{'='*70}
펀딩 헌터 탐지 보고서
{'='*70}

📊 탐지 요약
  - 총 탐지 케이스: {len(cases)}건
  - Critical (확실한 악용): {severity_counts[SeverityLevel.CRITICAL]}건
  - High (높은 의심): {severity_counts[SeverityLevel.HIGH]}건
  - Medium (중간 의심): {severity_counts[SeverityLevel.MEDIUM]}건
  - Low (낮은 의심): {severity_counts[SeverityLevel.LOW]}건

💰 펀딩비 악용 규모
  - 총 펀딩비 수익: ${total_funding:,.2f}
  - 연루 계정 수: {len(account_summaries)}개
  - 평균 계정당 수익: ${total_funding/len(account_summaries):,.2f}

🎯 상위 계정 (Top 10)
"""
        
        for idx, acc in enumerate(top_accounts, 1):
            report += f"""  {idx}. {acc.account_id}
     - 총 수익: ${acc.total_funding_profit:,.2f}
     - 탐지 횟수: {acc.total_cases}건
     - 평균 점수: {acc.avg_score:.1f}
     - Critical: {acc.critical_count}건, High: {acc.high_count}건
"""
        
        report += f"""
{'='*70}
"""
        
        filepath = self.output_dir / "summary_report.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"요약 보고서 저장: {filepath}")
        print(report)
    
    def _export_sanction_accounts(self, account_summaries: Dict[str, AccountSummary]):
        """제재 대상 계정 JSON"""
        
        # Critical 또는 High가 2건 이상인 계정
        sanction_accounts = [
            summary for summary in account_summaries.values()
            if summary.critical_count >= 1 or summary.high_count >= 2
        ]
        
        sanction_data = {
            'total_sanction_accounts': len(sanction_accounts),
            'generated_at': datetime.now().isoformat(),
            'accounts': [acc.to_dict() for acc in sanction_accounts]
        }
        
        filepath = self.output_dir / "sanction_accounts.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sanction_data, f, indent=2, ensure_ascii=False)
        
        print(f"제재 계정 저장: {filepath} ({len(sanction_accounts)}개)")


# ============================================================================
# 9. MAIN DETECTOR ENGINE
# ============================================================================

class FundingHunterDetector:
    """메인 탐지 엔진"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger(config)
        
        # 엔진 컴포넌트
        self.filter_engine = FilterEngine(config)
        self.scoring_engine = ScoringEngine(config)
        self.account_analyzer = AccountAnalyzer(config)
        self.report_generator = ReportGenerator(config)
    
    def detect(self, data_filepath: str) -> Dict:
        """전체 탐지 프로세스 실행"""
        
        # 1. 데이터 로드 (공통 DataManager 사용)
        self.logger.log_phase("데이터 로드")
        dm = get_data_manager(data_filepath)
        data = dm.get_all_sheets()
        con = dm.get_connection()
        
        # 2. 후보 추출
        self.logger.log_phase("후보 케이스 추출")
        extractor = CandidateExtractor(con)
        candidates = extractor.extract_candidates()
        
        if len(candidates) == 0:
            print("후보 케이스가 없습니다. 탐지 종료.")
            return self._empty_result()
        
        # 3. 필터 적용
        self.logger.log_phase("필터 적용")
        passed_cases, failed_cases = self.filter_engine.apply_filters(candidates)
        self.logger.log_filter_result(len(candidates), len(passed_cases), len(failed_cases))
        
        if len(passed_cases) == 0:
            print("필터를 통과한 케이스가 없습니다.")
            return self._empty_result()
        
        # 4. 점수 계산
        self.logger.log_phase("점수 계산 및 심각도 분류")
        scored_cases = self.scoring_engine.score_all_cases(passed_cases)
        
        # 심각도 분포 로깅
        from collections import defaultdict
        severity_counts = defaultdict(int)
        for case in scored_cases:
            severity_counts[case.severity] += 1
        self.logger.log_severity_distribution(severity_counts)
        
        # 5. 계정별 분석
        self.logger.log_phase("계정별 분석")
        account_summaries = self.account_analyzer.analyze_accounts(scored_cases)
        
        # 6. 보고서 생성
        self.logger.log_phase("보고서 생성")
        self.report_generator.generate_all_reports(scored_cases, account_summaries)
        
        # 7. 결과 반환
        return {
            'config': self.config.to_dict(),
            'total_candidates': len(candidates),
            'passed_filter': len(passed_cases),
            'severity_distribution': {k.value: v for k, v in severity_counts.items()},
            'total_accounts': len(account_summaries),
            'total_funding_profit': sum(s.total_funding_profit for s in account_summaries.values()),
            'output_directory': self.config.output_dir,
        }
    
    def _empty_result(self) -> Dict:
        """빈 결과 반환"""
        return {
            'config': self.config.to_dict(),
            'total_candidates': 0,
            'passed_filter': 0,
            'severity_distribution': {},
            'total_accounts': 0,
            'total_funding_profit': 0.0,
            'output_directory': self.config.output_dir,
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_detection(
    data_filepath: str,
    config: Optional[DetectionConfig] = None
) -> Dict:
    """
    펀딩 헌터 탐지 실행
    
    Args:
        data_filepath: Excel 데이터 파일 경로
        config: 탐지 설정 (None이면 기본값 사용)
    
    Returns:
        탐지 결과 딕셔너리
    """
    if config is None:
        config = DetectionConfig()
    
    # 설정 저장
    config.save(str(Path(config.output_dir) / 'detection_config.json'))
    
    # 탐지 실행
    detector = FundingHunterDetector(config)
    result = detector.detect(data_filepath)
    
    return result


if __name__ == "__main__":
    # 커스텀 설정
    custom_config = DetectionConfig(
        # Filter 파라미터
        min_leverage=5,
        min_amount_ratio=0.3,
        max_holding_minutes=20.0,
        require_hour_change=True,
        
        # 심각도 임계값
        critical_threshold=85,
        high_threshold=70,
        medium_threshold=50,
        
        # 출력 설정
        output_dir="./output/funding_fee",
        enable_detailed_logging=True
    )
    
    # 실행
    print("\n" + "="*70)
    print("펀딩 헌터 탐지 시스템 v2.0")
    print("="*70 + "\n")
    
    result = run_detection(
        data_filepath="problem_data_final.xlsx",
        config=custom_config
    )
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("탐지 완료!")
    print("="*70)
    print(f"총 후보: {result['total_candidates']}건")
    print(f"필터 통과: {result['passed_filter']}건")
    print(f"\n심각도 분포:")
    for severity, count in result['severity_distribution'].items():
        print(f"  - {severity}: {count}건")
    print(f"\n총 계정: {result['total_accounts']}개")
    print(f"총 펀딩비 수익: ${result['total_funding_profit']:,.2f}")
    print(f"\n결과 저장 위치: {result['output_directory']}")
    print("="*70 + "\n")
