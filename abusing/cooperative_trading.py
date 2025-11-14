"""
Cooperative Trading Detection System (공모거래 탐지 시스템)
Version: 2.1
Author: Singapore Fintech Hackathon Team

주요 개선 사항 (v2.1):
1. PnL 계산 시 position_id 중복 제거 로직 추가
   - 동일 position_id가 여러 거래 쌍에 나타날 수 있어 중복 집계 방지
   - AD_2.py 로직 참고하여 개선
2. SQL 쿼리 시간 계산 개선 (epoch 함수 사용)

탐지 대상: 복수 계정 간 협력하여 부당 이득을 취하는 패턴
- 동시 매매 패턴
- IP 공유
- 네트워크 분석
"""

import pandas as pd
import duckdb as dd
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
import logging
from collections import defaultdict, Counter
# Detectors read model tables directly from the persistent DuckDB file

# ============================================================================
# 1. CONFIGURATION & TYPES
# ============================================================================

class RiskLevel(Enum):
    """위험도 분류"""
    CRITICAL = "CRITICAL"      # 확실한 공모
    HIGH = "HIGH"              # 높은 의심
    MEDIUM = "MEDIUM"          # 중간 의심
    LOW = "LOW"                # 낮은 의심


class SanctionType(Enum):
    """제재 유형"""
    IMMEDIATE_CRITICAL = "IMMEDIATE_CRITICAL"      # Critical 즉시 제재
    IP_SHARED_NETWORK = "IP_SHARED_NETWORK"        # IP 공유 네트워크
    REPEATED_PATTERN = "REPEATED_PATTERN"          # 반복 패턴


@dataclass
class DetectionConfig:
    """탐지 설정 및 하이퍼파라미터"""
    
    # ===== Filter Parameters (필수 조건) =====
    max_open_time_diff_min: float = 2.0           # 최대 오픈 시간차 (분)
    max_close_time_diff_min: float = 2.0          # 최대 클로즈 시간차 (분)
    exclude_major_symbols: bool = True             # 주요 심볼 제외
    major_symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT.PERP', 'ETHUSDT.PERP', 'SOLUSDT.PERP',
        'XRPUSDT.PERP', 'BNBUSDT.PERP', 'DOGEUSDT.PERP'
    ])
    
    # ===== Scoring Weights (점수 배점) =====
    weight_pnl_asymmetry: int = 35                 # PnL 비대칭성 (한쪽 큰 이익)
    weight_time_proximity: int = 25                # 시간 근접도
    weight_ip_sharing: int = 25                    # IP 공유
    weight_position_overlap: int = 15              # 포지션 겹침
    
    # ===== Risk Thresholds (위험도 판정 기준) =====
    critical_threshold: int = 85                   # Critical 최소 점수
    high_threshold: int = 70                       # High 최소 점수
    medium_threshold: int = 50                     # Medium 최소 점수
    
    # ===== Network Analysis Parameters =====
    min_group_size: int = 2                        # 최소 그룹 크기
    min_shared_ips: int = 1                        # 제재 최소 공유 IP 수
    
    # ===== Output Settings =====
    output_dir: str = "output/cooperative"
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
    pnl_asymmetry: float = 0.0
    time_proximity: float = 0.0
    ip_sharing: float = 0.0
    position_overlap: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradePair:
    """거래 쌍 정보"""
    pair_id: str
    account_id1: str
    account_id2: str
    
    # 거래 정보
    symbol: str
    side1: str
    side2: str
    
    # 시간 정보
    open_ts1: datetime
    open_ts2: datetime
    closing_ts1: datetime
    closing_ts2: datetime
    open_time_diff_sec: float
    close_time_diff_sec: float
    
    # 거래 상세
    amount1: float
    amount2: float
    leverage: int
    position_id1: str
    position_id2: str
    
    # 손익 정보
    rpnl1: float
    rpnl2: float
    total_pnl: float
    pnl_winner: str  # account with positive pnl
    pnl_loser: str   # account with negative pnl
    
    # 점수 및 판정
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    risk_level: RiskLevel = RiskLevel.LOW
    
    # 필터 통과 여부
    passed_filter: bool = False
    filter_failures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        data['open_ts1'] = self.open_ts1.isoformat()
        data['open_ts2'] = self.open_ts2.isoformat()
        data['closing_ts1'] = self.closing_ts1.isoformat()
        data['closing_ts2'] = self.closing_ts2.isoformat()
        return data


@dataclass
class CooperativeGroup:
    """공모 그룹"""
    group_id: str
    members: List[str]
    
    # 거래 정보
    trade_pair_ids: List[str]
    trade_count: int
    
    # 손익 정보
    pnl_positive_sum: float
    pnl_negative_sum: float
    pnl_total: float
    
    # IP 정보
    shared_ip_count: int
    shared_ips: Dict[str, List[str]] = field(default_factory=dict)  # IP -> accounts
    
    # 점수 및 판정
    avg_score: float = 0.0
    max_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['risk_level'] = self.risk_level.value
        return data


@dataclass
class SanctionCase:
    """제재 케이스"""
    case_id: str
    sanction_type: SanctionType
    group_id: str
    account_ids: List[str]
    detection_timestamp: datetime
    
    # 증거 데이터
    trade_pair_ids: List[str]
    total_score: float
    risk_level: RiskLevel
    
    # IP 정보
    shared_ip_count: int = 0
    shared_ips: Dict[str, List[str]] = field(default_factory=dict)
    
    # 패턴 정보
    pattern_count: Optional[int] = None
    
    # 추가 메타데이터
    total_pnl: float = 0.0
    pnl_positive_sum: float = 0.0
    pnl_negative_sum: float = 0.0
    evidence_summary: str = ""
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['sanction_type'] = self.sanction_type.value
        data['risk_level'] = self.risk_level.value
        data['detection_timestamp'] = self.detection_timestamp.isoformat()
        return data


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
        self.logger.info("공모거래 탐지 시스템 시작")
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
    
    def log_risk_distribution(self, risk_counts: Dict[RiskLevel, int]):
        """위험도 분포 로그"""
        self.logger.info("위험도 분포:")
        for risk, count in risk_counts.items():
            self.logger.info(f"  - {risk.value}: {count}건")
    
    def log_group(self, group: CooperativeGroup):
        """그룹 로그"""
        self.logger.warning(f"공모 그룹 탐지: {group.group_id}")
        self.logger.warning(f"  - 멤버: {', '.join(group.members)}")
        self.logger.warning(f"  - 위험도: {group.risk_level.value}")
        self.logger.warning(f"  - 총 PnL: ${group.pnl_total:.2f}")


# ============================================================================
# 3. DATA PIPELINE
# ============================================================================
# DataLoader removed - using common.data_manager.DataManager singleton instead.


# ============================================================================
# 4. CANDIDATE EXTRACTOR
# ============================================================================

class CandidateExtractor:
    """후보 거래 쌍 추출"""
    
    def __init__(self, con: dd.DuckDBPyConnection, config: DetectionConfig):
        self.con = con
        self.config = config
    
    def extract_candidates(self) -> List[Dict]:
        """SQL을 통해 후보 거래 쌍 추출"""
        print("후보 거래 쌍 추출 중...")
        
        # 주요 심볼 제외 조건
        exclude_clause = ""
        if self.config.exclude_major_symbols:
            symbols_str = "', '".join(self.config.major_symbols)
            exclude_clause = f"AND t1.symbol NOT IN ('{symbols_str}')"
        
        query = f"""
        WITH position AS (
            SELECT
                account_id,
                position_id,
                MAX(leverage) AS leverage,
                CAST(MIN(ts) AS TIMESTAMP) as open_ts,
                CAST(MAX(ts) AS TIMESTAMP) as closing_ts,
                MAX(symbol) as symbol,
                MAX(side) as side,
                DATE(MAX(ts)) as closing_day,
                SUM(CASE WHEN openclose='OPEN' THEN amount ELSE 0 END) as amount,
                SUM(
                    CASE WHEN openclose='OPEN' THEN -amount ELSE amount END * 
                    CASE WHEN side='LONG' THEN 1 ELSE -1 END
                ) as rpnl
            FROM Trade
            GROUP BY account_id, position_id
        ),
        joined AS (
            SELECT
                t1.account_id AS account_id1,
                t2.account_id AS account_id2,
                t1.symbol,
                t1.open_ts AS open_ts1,
                t2.open_ts AS open_ts2,
                t1.closing_ts AS closing_ts1,
                t2.closing_ts AS closing_ts2,
                t1.leverage,
                t1.amount AS amount1,
                t2.amount AS amount2,
                t1.position_id AS position_id1,
                t2.position_id AS position_id2,
                t1.side as side1,
                t2.side as side2,
                t1.rpnl as rpnl1,
                t2.rpnl as rpnl2,
                ABS(epoch(t1.open_ts) - epoch(t2.open_ts)) AS open_time_diff_sec,
                ABS(epoch(t1.closing_ts) - epoch(t2.closing_ts)) AS close_time_diff_sec
            FROM position t1 
            INNER JOIN position t2 ON
                t1.symbol = t2.symbol
                AND ABS(epoch(t1.open_ts) - epoch(t2.open_ts)) <= {self.config.max_open_time_diff_min * 60}
                AND ABS(epoch(t1.closing_ts) - epoch(t2.closing_ts)) <= {self.config.max_close_time_diff_min * 60}
                AND t1.open_ts < t2.open_ts
                AND GREATEST(t1.open_ts, t2.open_ts) < LEAST(t1.closing_ts, t2.closing_ts)
                AND t1.account_id != t2.account_id
                AND t1.side = t2.side
                {exclude_clause}
        )
        SELECT DISTINCT *
        FROM joined
        ORDER BY symbol, open_ts1
        """
        
        df = self.con.execute(query).fetchdf()
        print(f"후보 거래 쌍 {len(df)}개 추출")
        
        return df.to_dict('records')


# ============================================================================
# 5. FILTER ENGINE
# ============================================================================

class FilterEngine:
    """필수 조건 필터링"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def apply_filters(self, candidates: List[Dict]) -> Tuple[List[TradePair], List[Dict]]:
        """필터 적용 및 TradePair 객체 생성"""
        print("필터 엔진 시작...")
        
        passed_pairs = []
        failed_data = []
        
        for idx, row in enumerate(candidates):
            pair_id = f"COOP_{idx:06d}"
            failures = []
            
            # 기본적으로 SQL에서 이미 필터링됨
            # 추가 필터가 필요한 경우 여기에 추가
            
            # TradePair 객체 생성
            trade_pair = self._create_trade_pair(pair_id, row)
            trade_pair.passed_filter = True
            passed_pairs.append(trade_pair)
        
        print(f"필터 완료: {len(passed_pairs)}/{len(candidates)} 통과")
        
        return passed_pairs, failed_data
    
    def _create_trade_pair(self, pair_id: str, row: Dict) -> TradePair:
        """TradePair 객체 생성"""
        rpnl1 = row['rpnl1']
        rpnl2 = row['rpnl2']
        
        # 승자/패자 결정
        if rpnl1 > rpnl2:
            pnl_winner = row['account_id1']
            pnl_loser = row['account_id2']
        else:
            pnl_winner = row['account_id2']
            pnl_loser = row['account_id1']
        
        return TradePair(
            pair_id=pair_id,
            account_id1=row['account_id1'],
            account_id2=row['account_id2'],
            symbol=row['symbol'],
            side1=row['side1'],
            side2=row['side2'],
            open_ts1=row['open_ts1'],
            open_ts2=row['open_ts2'],
            closing_ts1=row['closing_ts1'],
            closing_ts2=row['closing_ts2'],
            open_time_diff_sec=row['open_time_diff_sec'],
            close_time_diff_sec=row['close_time_diff_sec'],
            amount1=row['amount1'],
            amount2=row['amount2'],
            leverage=row['leverage'],
            position_id1=row['position_id1'],
            position_id2=row['position_id2'],
            rpnl1=rpnl1,
            rpnl2=rpnl2,
            total_pnl=rpnl1 + rpnl2,
            pnl_winner=pnl_winner,
            pnl_loser=pnl_loser,
        )


# ============================================================================
# 6. SCORING ENGINE
# ============================================================================

class ScoringEngine:
    """점수 계산 및 위험도 분류"""
    
    def __init__(self, config: DetectionConfig, ip_data: pd.DataFrame):
        self.config = config
        self.ip_data = ip_data
        
        # IP 매핑 생성 (account_id -> set of IPs)
        self.account_ips = self._build_ip_mapping()
    
    def _build_ip_mapping(self) -> Dict[str, Set[str]]:
        """계정별 IP 매핑 생성"""
        mapping = defaultdict(set)
        for _, row in self.ip_data.iterrows():
            mapping[row['account_id']].add(row['ip'])
        return mapping
    
    def score_all_pairs(self, pairs: List[TradePair]) -> List[TradePair]:
        """모든 거래 쌍 점수 계산"""
        print("점수 엔진 시작...")
        
        for pair in pairs:
            # 각 지표별 점수 계산
            pnl_score = self._score_pnl_asymmetry(pair)
            time_score = self._score_time_proximity(pair)
            ip_score = self._score_ip_sharing(pair)
            overlap_score = self._score_position_overlap(pair)
            
            # 점수 저장
            pair.score = ScoreBreakdown(
                pnl_asymmetry=pnl_score,
                time_proximity=time_score,
                ip_sharing=ip_score,
                position_overlap=overlap_score,
                total=pnl_score + time_score + ip_score + overlap_score
            )
            
            # 위험도 분류
            pair.risk_level = self._classify_risk(pair.score.total)
            
            # if self.config.enable_detailed_logging:
            #     print(
            #         f"{pair.pair_id}: 점수={pair.score.total:.1f} "
            #         f"(PnL:{pnl_score:.1f}, Time:{time_score:.1f}, "
            #         f"IP:{ip_score:.1f}, Overlap:{overlap_score:.1f}) "
            #         f"→ {pair.risk_level.value}"
            #     )
        
        print(f"점수 계산 완료: {len(pairs)}개")
        
        return pairs
    
    def _score_pnl_asymmetry(self, pair: TradePair) -> float:
        """PnL 비대칭성 점수 (35점) - 한쪽이 큰 이익"""
        max_weight = self.config.weight_pnl_asymmetry
        
        total_pnl = pair.rpnl1 + pair.rpnl2
        max_pnl = max(abs(pair.rpnl1), abs(pair.rpnl2))
        
        if max_pnl == 0:
            return 0.0
        
        # 비대칭 비율: 클수록 한쪽만 이득
        asymmetry_ratio = abs(total_pnl) / max_pnl
        
        if asymmetry_ratio >= 0.8:  # 80% 이상: 한쪽만 큰 이득
            return max_weight
        elif asymmetry_ratio >= 0.6:
            return max_weight * 0.75
        elif asymmetry_ratio >= 0.4:
            return max_weight * 0.50
        elif asymmetry_ratio >= 0.2:
            return max_weight * 0.25
        else:
            return 0.0
    
    def _score_time_proximity(self, pair: TradePair) -> float:
        """시간 근접도 점수 (25점)"""
        max_weight = self.config.weight_time_proximity
        
        avg_diff = (pair.open_time_diff_sec + pair.close_time_diff_sec) / 2
        
        if avg_diff <= 5:  # 5초 이내
            return max_weight
        elif avg_diff <= 15:  # 15초 이내
            return max_weight * 0.80
        elif avg_diff <= 30:  # 30초 이내
            return max_weight * 0.60
        elif avg_diff <= 60:  # 1분 이내
            return max_weight * 0.40
        else:  # 2분 이내
            return max_weight * 0.20
    
    def _score_ip_sharing(self, pair: TradePair) -> float:
        """IP 공유 점수 (25점)"""
        max_weight = self.config.weight_ip_sharing
        
        ips1 = self.account_ips.get(pair.account_id1, set())
        ips2 = self.account_ips.get(pair.account_id2, set())
        
        shared_ips = ips1 & ips2
        shared_count = len(shared_ips)
        
        if shared_count >= 5:
            return max_weight
        elif shared_count >= 3:
            return max_weight * 0.80
        elif shared_count >= 2:
            return max_weight * 0.60
        elif shared_count >= 1:
            return max_weight * 0.40
        else:
            return 0.0
    
    def _score_position_overlap(self, pair: TradePair) -> float:
        """포지션 겹침 점수 (15점)"""
        max_weight = self.config.weight_position_overlap
        
        # 오픈/클로즈 시간 겹침 계산
        overlap_start = max(pair.open_ts1, pair.open_ts2)
        overlap_end = min(pair.closing_ts1, pair.closing_ts2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        total_duration1 = (pair.closing_ts1 - pair.open_ts1).total_seconds()
        total_duration2 = (pair.closing_ts2 - pair.open_ts2).total_seconds()
        min_duration = min(total_duration1, total_duration2)
        
        if min_duration == 0:
            return 0.0
        
        overlap_ratio = overlap_duration / min_duration
        
        if overlap_ratio >= 0.9:
            return max_weight
        elif overlap_ratio >= 0.7:
            return max_weight * 0.75
        elif overlap_ratio >= 0.5:
            return max_weight * 0.50
        else:
            return max_weight * 0.25
    
    def _classify_risk(self, total_score: float) -> RiskLevel:
        """점수 기반 위험도 분류"""
        if total_score >= self.config.critical_threshold:
            return RiskLevel.CRITICAL
        elif total_score >= self.config.high_threshold:
            return RiskLevel.HIGH
        elif total_score >= self.config.medium_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ============================================================================
# 7. NETWORK ANALYZER
# ============================================================================

class NetworkAnalyzer:
    """네트워크 분석 및 그룹 탐지"""
    
    def __init__(self, config: DetectionConfig, ip_data: pd.DataFrame):
        self.config = config
        self.ip_data = ip_data
    
    def find_groups(self, pairs: List[TradePair]) -> List[CooperativeGroup]:
        """연결된 계정 그룹 찾기"""
        print("네트워크 그룹 탐색 중...")
        
        # 계정 쌍 수집
        unique_pairs = set()
        for pair in pairs:
            sorted_accounts = tuple(sorted([pair.account_id1, pair.account_id2]))
            unique_pairs.add(sorted_accounts)
        
        # Union-Find 알고리즘으로 그룹 찾기
        groups = []
        
        for a, b in unique_pairs:
            found = []
            for g in groups:
                if a in g or b in g:
                    g.update([a, b])
                    found.append(g)
            
            if not found:
                groups.append(set([a, b]))
            elif len(found) > 1:
                merged = set().union(*found)
                groups = [g for g in groups if g not in found]
                groups.append(merged)
        
        # CooperativeGroup 객체 생성
        cooperative_groups = []
        
        for idx, group_set in enumerate(groups):
            if len(group_set) < self.config.min_group_size:
                continue
            
            group_members = sorted(list(group_set))
            
            # 그룹 관련 거래 쌍 찾기
            group_pairs = [
                p for p in pairs
                if p.account_id1 in group_set or p.account_id2 in group_set
            ]
            
            # PnL 계산 (position_id 중복 제거)
            # AD_2.py 로직 참고: 동일 position_id가 여러 pair에 나타날 수 있어 중복 제거 필요
            unique_rpnl = []
            
            # position_id1들의 rpnl (중복 제거)
            seen_position_ids = set()
            for p in group_pairs:
                if p.position_id1 not in seen_position_ids:
                    unique_rpnl.append(p.rpnl1)
                    seen_position_ids.add(p.position_id1)
            
            # position_id2들의 rpnl (중복 제거)
            for p in group_pairs:
                if p.position_id2 not in seen_position_ids:
                    unique_rpnl.append(p.rpnl2)
                    seen_position_ids.add(p.position_id2)
            
            # 합계 계산
            total_pos = sum(max(0, rpnl) for rpnl in unique_rpnl)
            total_neg = sum(min(0, rpnl) for rpnl in unique_rpnl)
            total_pnl = total_pos + total_neg
            
            # IP 공유 분석
            shared_ip_info = self._analyze_shared_ips(group_members)
            
            # 평균 점수 계산
            scores = [p.score.total for p in group_pairs]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            
            group = CooperativeGroup(
                group_id=f"GROUP_{idx:04d}",
                members=group_members,
                trade_pair_ids=[p.pair_id for p in group_pairs],
                trade_count=len(group_pairs),
                pnl_positive_sum=total_pos,
                pnl_negative_sum=total_neg,
                pnl_total=total_pnl,
                shared_ip_count=len(shared_ip_info),
                shared_ips=shared_ip_info,
                avg_score=avg_score,
                max_score=max_score,
                risk_level=self._classify_group_risk(avg_score, len(shared_ip_info))
            )
            
            cooperative_groups.append(group)
        
        # PnL 기준 정렬
        cooperative_groups.sort(key=lambda x: x.pnl_total, reverse=True)
        
        print(f"그룹 탐색 완료: {len(cooperative_groups)}개 그룹")
        
        return cooperative_groups
    
    def _analyze_shared_ips(self, members: List[str]) -> Dict[str, List[str]]:
        """그룹 내 공유 IP 분석"""
        group_ips = self.ip_data[self.ip_data['account_id'].isin(members)]
        ip_counter = Counter(group_ips['ip'])
        
        shared_ips = {}
        for ip, count in ip_counter.items():
            if count > 1:
                accounts = group_ips[group_ips['ip'] == ip]['account_id'].tolist()
                shared_ips[ip] = accounts
        
        return shared_ips
    
    def _classify_group_risk(self, avg_score: float, shared_ip_count: int) -> RiskLevel:
        """그룹 위험도 분류"""
        # IP 공유가 많으면 위험도 상승
        bonus = shared_ip_count * 5
        adjusted_score = avg_score + bonus
        
        if adjusted_score >= 85:
            return RiskLevel.CRITICAL
        elif adjusted_score >= 70:
            return RiskLevel.HIGH
        elif adjusted_score >= 50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ============================================================================
# 8. SANCTION PIPELINE
# ============================================================================

class SanctionPipeline:
    """제재 케이스 생성 및 출력"""
    
    def __init__(self, config: DetectionConfig, logger: DetectionLogger):
        self.config = config
        self.logger = logger
        self.sanction_cases: List[SanctionCase] = []
    
    def process_critical_groups(self, groups: List[CooperativeGroup]) -> List[SanctionCase]:
        """Critical 그룹 즉시 제재 케이스 생성"""
        print("Critical 그룹 제재 생성 중...")
        
        critical_groups = [g for g in groups if g.risk_level == RiskLevel.CRITICAL]
        
        if len(critical_groups) == 0:
            print("Critical 그룹 없음")
            return []
        
        for group in critical_groups:
            sanction = SanctionCase(
                case_id=f"SANCTION_CRITICAL_{group.group_id}",
                sanction_type=SanctionType.IMMEDIATE_CRITICAL,
                group_id=group.group_id,
                account_ids=group.members,
                detection_timestamp=datetime.now(),
                trade_pair_ids=group.trade_pair_ids,
                total_score=group.max_score,
                risk_level=RiskLevel.CRITICAL,
                shared_ip_count=group.shared_ip_count,
                shared_ips=group.shared_ips,
                total_pnl=group.pnl_total,
                pnl_positive_sum=group.pnl_positive_sum,
                pnl_negative_sum=group.pnl_negative_sum,
                evidence_summary=f"확실한 공모 거래 패턴 (점수: {group.max_score:.1f}/100, 거래: {group.trade_count}건)"
            )
            
            self.sanction_cases.append(sanction)
            self.logger.logger.warning(f"제재 케이스 생성: {sanction.case_id}")
        
        print(f"Critical 제재: {len(critical_groups)}건")
        
        return [c for c in self.sanction_cases if c.sanction_type == SanctionType.IMMEDIATE_CRITICAL]
    
    def process_ip_shared_groups(self, groups: List[CooperativeGroup]) -> List[SanctionCase]:
        """IP 공유 네트워크 제재 케이스 생성"""
        print("IP 공유 네트워크 제재 케이스 생성 중...")
        
        # HIGH 위험도 + IP 공유가 있는 그룹
        ip_shared_groups = [
            g for g in groups
            if g.risk_level == RiskLevel.HIGH 
            and g.shared_ip_count >= self.config.min_shared_ips
        ]
        
        ip_sanctions = []
        
        for group in ip_shared_groups:
            # 이미 Critical로 제재된 경우 스킵
            if any(c.group_id == group.group_id and c.sanction_type == SanctionType.IMMEDIATE_CRITICAL 
                   for c in self.sanction_cases):
                continue
            
            sanction = SanctionCase(
                case_id=f"SANCTION_IP_{group.group_id}",
                sanction_type=SanctionType.IP_SHARED_NETWORK,
                group_id=group.group_id,
                account_ids=group.members,
                detection_timestamp=datetime.now(),
                trade_pair_ids=group.trade_pair_ids,
                total_score=group.avg_score,
                risk_level=RiskLevel.HIGH,
                shared_ip_count=group.shared_ip_count,
                shared_ips=group.shared_ips,
                total_pnl=group.pnl_total,
                pnl_positive_sum=group.pnl_positive_sum,
                pnl_negative_sum=group.pnl_negative_sum,
                evidence_summary=f"IP 공유 네트워크 ({group.shared_ip_count}개 IP 공유, 거래: {group.trade_count}건)"
            )
            
            ip_sanctions.append(sanction)
            self.sanction_cases.append(sanction)
            self.logger.logger.warning(f"제재 케이스 생성: {sanction.case_id}")
        
        print(f"IP 공유 제재: {len(ip_sanctions)}건")
        
        return ip_sanctions
    
    def export_sanctions(self, output_dir: Path) -> str:
        """제재 케이스를 JSON 파일로 출력"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "sanction_groups.json"
        
        data = {
            'total_sanction_groups': len(self.sanction_cases),
            'generated_at': datetime.now().isoformat(),
            'sanctions': [case.to_dict() for case in self.sanction_cases]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"제재 케이스 저장: {filepath} ({len(self.sanction_cases)}건)")
        
        return str(filepath)


# ============================================================================
# 9. REPORTING & VISUALIZATION
# ============================================================================

class ReportGenerator:
    """분석 보고서 및 시각화 데이터 생성"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def generate_all_reports(
        self,
        all_pairs: List[TradePair],
        groups: List[CooperativeGroup]
    ):
        """모든 보고서 생성"""
        print("보고서 생성 중...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 거래 쌍 상세 데이터 (CSV)
        self._export_trade_pairs_csv(all_pairs)
        
        # 2. 그룹 상세 데이터 (CSV)
        self._export_groups_csv(groups)
        
        # 3. 시각화용 JSON 데이터
        self._export_visualization_data(all_pairs, groups)
        
        # 4. 요약 보고서 (텍스트)
        self._generate_summary_report(all_pairs, groups)
        
        print("보고서 생성 완료")
    
    def _export_trade_pairs_csv(self, pairs: List[TradePair]):
        """거래 쌍 상세 CSV"""
        if not pairs:
            return
        
        records = []
        for pair in pairs:
            record = {
                'pair_id': pair.pair_id,
                'account_id1': pair.account_id1,
                'account_id2': pair.account_id2,
                'risk_level': pair.risk_level.value,
                'total_score': pair.score.total,
                'symbol': pair.symbol,
                'rpnl1': pair.rpnl1,
                'rpnl2': pair.rpnl2,
                'total_pnl': pair.total_pnl,
                'pnl_winner': pair.pnl_winner,
                'pnl_loser': pair.pnl_loser,
                'open_time_diff_sec': pair.open_time_diff_sec,
                'close_time_diff_sec': pair.close_time_diff_sec,
                'score_pnl_asymmetry': pair.score.pnl_asymmetry,
                'score_time_proximity': pair.score.time_proximity,
                'score_ip_sharing': pair.score.ip_sharing,
                'score_position_overlap': pair.score.position_overlap,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = self.output_dir / "trade_pairs_detailed.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"거래 쌍 CSV 저장: {filepath}")
    
    def _export_groups_csv(self, groups: List[CooperativeGroup]):
        """그룹 상세 CSV"""
        if not groups:
            return
        
        records = []
        for group in groups:
            record = {
                'group_id': group.group_id,
                'members': ', '.join(group.members),
                'member_count': len(group.members),
                'trade_count': group.trade_count,
                'risk_level': group.risk_level.value,
                'avg_score': group.avg_score,
                'max_score': group.max_score,
                'pnl_positive_sum': group.pnl_positive_sum,
                'pnl_negative_sum': group.pnl_negative_sum,
                'pnl_total': group.pnl_total,
                'shared_ip_count': group.shared_ip_count,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = self.output_dir / "cooperative_groups.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"그룹 CSV 저장: {filepath}")
    
    def _export_visualization_data(
        self, 
        pairs: List[TradePair],
        groups: List[CooperativeGroup]
    ):
        """시각화용 JSON 데이터"""
        
        # 위험도별 분포
        risk_counts = defaultdict(int)
        for pair in pairs:
            risk_counts[pair.risk_level.value] += 1
        
        # 네트워크 그래프 데이터
        nodes = set()
        edges = []
        
        for pair in pairs:
            if pair.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                nodes.add(pair.account_id1)
                nodes.add(pair.account_id2)
                
                edges.append({
                    'source': pair.account_id1,
                    'target': pair.account_id2,
                    'value': abs(pair.total_pnl),
                    'risk_level': pair.risk_level.value,
                    'score': pair.score.total,
                })
        
        vis_data = {
            'summary': {
                'total_pairs': len(pairs),
                'critical': risk_counts.get('CRITICAL', 0),
                'high': risk_counts.get('HIGH', 0),
                'medium': risk_counts.get('MEDIUM', 0),
                'low': risk_counts.get('LOW', 0),
                'total_groups': len(groups),
                'total_pnl': sum(g.pnl_total for g in groups),
            },
            'risk_distribution': dict(risk_counts),
            'network_graph': {
                'nodes': [{'id': node} for node in nodes],
                'edges': edges,
            },
            'group_stats': [
                {
                    'group_id': g.group_id,
                    'member_count': len(g.members),
                    'pnl_total': g.pnl_total,
                    'risk_level': g.risk_level.value,
                }
                for g in groups
            ]
        }
        
        filepath = self.output_dir / "visualization_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vis_data, f, indent=2, ensure_ascii=False)
        
        print(f"시각화 데이터 저장: {filepath}")
    
    def _generate_summary_report(
        self,
        pairs: List[TradePair],
        groups: List[CooperativeGroup]
    ):
        """요약 보고서 텍스트"""
        
        risk_counts = defaultdict(int)
        for pair in pairs:
            risk_counts[pair.risk_level] += 1
        
        total_pnl = sum(g.pnl_total for g in groups)
        groups_with_shared_ip = sum(1 for g in groups if g.shared_ip_count > 0)
        
        # 상위 그룹
        top_groups = groups[:5]
        
        report = f"""
{'='*70}
공모거래 탐지 보고서
{'='*70}

📊 탐지 요약
  - 총 의심 거래 쌍: {len(pairs)}건
  - Critical (확실한 공모): {risk_counts[RiskLevel.CRITICAL]}건
  - High (높은 의심): {risk_counts[RiskLevel.HIGH]}건
  - Medium (중간 의심): {risk_counts[RiskLevel.MEDIUM]}건
  - Low (낮은 의심): {risk_counts[RiskLevel.LOW]}건

👥 그룹 분석
  - 탐지된 그룹: {len(groups)}개
  - IP 공유 그룹: {groups_with_shared_ip}개
  - 총 순수익: ${total_pnl:,.2f}

🎯 상위 그룹 (Top 5)
"""
        
        for idx, group in enumerate(top_groups, 1):
            report += f"""  {idx}. {group.group_id}
     - 멤버: {', '.join(group.members[:5])}{'...' if len(group.members) > 5 else ''}
     - 멤버 수: {len(group.members)}명
     - 거래 수: {group.trade_count}건
     - 순수익: ${group.pnl_total:,.2f}
     - 위험도: {group.risk_level.value}
     - 공유 IP: {group.shared_ip_count}개
"""
        
        report += f"""
{'='*70}
"""
        
        filepath = self.output_dir / "summary_report.txt"
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"요약 보고서 저장: {filepath}")
        print(report)


# ============================================================================
# 10. MAIN DETECTOR ENGINE
# ============================================================================

class CooperativeTradingDetector:
    """메인 탐지 엔진"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger(config)
        self.sanction_pipeline = SanctionPipeline(config, self.logger)
    
    def detect(self, data_filepath: str) -> Dict:
        """전체 탐지 프로세스 실행"""
        
        # 1. 데이터 로드 (공통 DataManager 사용)
        self.logger.log_phase("데이터 로드")
        # detectors should use the persistent DuckDB file created by main
        db_path = Path.cwd() / 'data' / 'ingest.duckdb'
        con = dd.connect(database=str(db_path))

        # load auxiliary tables used by non-SQL parts of the detector
        data = {}
        try:
            data['IP'] = con.execute('SELECT * FROM "IP"').fetchdf()
        except Exception:
            data['IP'] = pd.DataFrame()
        
        # 2. 후보 추출
        self.logger.log_phase("후보 거래 쌍 추출")
        extractor = CandidateExtractor(con, self.config)
        candidates = extractor.extract_candidates()
        
        if len(candidates) == 0:
            print("후보 거래 쌍이 없습니다. 탐지 종료.")
            return self._empty_result()
        
        # 3. 필터 적용
        self.logger.log_phase("필터 적용")
        filter_engine = FilterEngine(self.config)
        passed_pairs, failed_pairs = filter_engine.apply_filters(candidates)
        self.logger.log_filter_result(len(candidates), len(passed_pairs), len(failed_pairs))
        
        if len(passed_pairs) == 0:
            print("필터를 통과한 거래 쌍이 없습니다.")
            return self._empty_result()
        
        # 4. 점수 계산
        self.logger.log_phase("점수 계산 및 위험도 분류")
        scoring_engine = ScoringEngine(self.config, data['IP'])
        scored_pairs = scoring_engine.score_all_pairs(passed_pairs)
        
        # 위험도 분포 로깅
        risk_counts = defaultdict(int)
        for pair in scored_pairs:
            risk_counts[pair.risk_level] += 1
        self.logger.log_risk_distribution(risk_counts)
        
        # 5. 네트워크 분석
        self.logger.log_phase("네트워크 분석 및 그룹 탐지")
        network_analyzer = NetworkAnalyzer(self.config, data['IP'])
        groups = network_analyzer.find_groups(scored_pairs)
        
        # 6. Critical 그룹 즉시 제재
        self.logger.log_phase("Critical 그룹 제재")
        critical_sanctions = self.sanction_pipeline.process_critical_groups(groups)
        
        # 7. IP 공유 네트워크 제재
        self.logger.log_phase("IP 공유 네트워크 제재")
        ip_sanctions = self.sanction_pipeline.process_ip_shared_groups(groups)
        
        # 8. 제재 케이스 출력
        self.logger.log_phase("제재 케이스 출력")
        sanction_file = self.sanction_pipeline.export_sanctions(Path(self.config.output_dir))
        
        # 9. 보고서 생성
        self.logger.log_phase("보고서 생성")
        report_generator = ReportGenerator(self.config)
        report_generator.generate_all_reports(scored_pairs, groups)
        
        # 10. 결과 반환
        all_sanctions = critical_sanctions + ip_sanctions
        return {
            'config': self.config.to_dict(),
            'total_candidates': len(candidates),
            'passed_filter': len(passed_pairs),
            'risk_distribution': {k.value: v for k, v in risk_counts.items()},
            'total_groups': len(groups),
            'total_pnl': sum(g.pnl_total for g in groups),
            'sanction_cases': len(all_sanctions),
            'critical_sanctions': len(critical_sanctions),
            'ip_sanctions': len(ip_sanctions),
            'output_directory': self.config.output_dir,
        }
    
    def _empty_result(self) -> Dict:
        """빈 결과 반환"""
        return {
            'config': self.config.to_dict(),
            'total_candidates': 0,
            'passed_filter': 0,
            'risk_distribution': {},
            'total_groups': 0,
            'total_pnl': 0.0,
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
    공모거래 탐지 실행
    
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
    detector = CooperativeTradingDetector(config)
    result = detector.detect(data_filepath)
    
    return result


if __name__ == "__main__":
    # 커스텀 설정
    custom_config = DetectionConfig(
        # Filter 파라미터
        max_open_time_diff_min=2.0,
        max_close_time_diff_min=2.0,
        exclude_major_symbols=True,
        
        # 위험도 임계값
        critical_threshold=85,
        high_threshold=70,
        medium_threshold=50,
        
        # 네트워크 파라미터
        min_group_size=2,
        min_shared_ips=1,
        
        # 출력 설정
        output_dir="./output/cooperative",
        enable_detailed_logging=True
    )
    
    # 실행
    print("\n" + "="*70)
    print("공모거래 탐지 시스템 v2.0")
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
    print(f"\n위험도 분포:")
    for risk, count in result['risk_distribution'].items():
        print(f"  - {risk}: {count}건")
    print(f"\n총 그룹: {result['total_groups']}개")
    print(f"총 순수익: ${result['total_pnl']:,.2f}")
    print(f"\n결과 저장 위치: {result['output_directory']}")
    print("="*70 + "\n")
