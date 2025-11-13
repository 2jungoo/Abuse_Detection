"""
Bonus Laundering Detection System (증정금 녹이기 탐지 시스템)
Version: 2.0
Author: Singapore Fintech Hackathon Team

2-Tier 탐지 시스템:
- Tier 1 (Bot): 완벽한 매칭 패턴 → 즉시 제재
- Tier 2 (Manual): 느슨한 패턴 → 네트워킹 분석 → 반복 시 제재
"""

import pandas as pd
import duckdb as dd
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging
from common.data_manager import get_data_manager
from collections import defaultdict

# ============================================================================
# 1. CONFIGURATION & TYPES
# ============================================================================

class TierType(Enum):
    """탐지 Tier 분류"""
    BOT = "BOT"                    # 봇 기반 악의적 거래
    MANUAL = "MANUAL"              # 수동 악의적 거래
    SUSPICIOUS = "SUSPICIOUS"      # 의심 거래
    NORMAL = "NORMAL"              # 정상 거래


class SanctionType(Enum):
    """제재 유형"""
    IMMEDIATE_BOT = "IMMEDIATE_BOT"        # 봇 탐지 즉시 제재
    NETWORK_REPEAT = "NETWORK_REPEAT"      # 반복 수익 계정
    NETWORK_CHAIN = "NETWORK_CHAIN"        # 연결된 계정 체인


@dataclass
class DetectionConfig:
    """탐지 설정 및 하이퍼파라미터"""
    
    # ===== Filter Parameters (필수 조건) =====
    time_since_bonus_hours: float = 72.0      # 보너스 후 거래 시간 창 (72시간)
    concurrency_threshold_sec: float = 30.0    # 거래 동시성 임계값 (30초)
    quantity_tolerance_pct: float = 0.02       # 수량 허용 오차 (±2%)
    
    # ===== Scoring Weights (점수 배점) =====
    weight_pnl_mirroring: int = 40        # P&L 대칭성
    weight_high_concurrency: int = 25     # 시간 근접도
    weight_quantity_match: int = 20       # 수량 일치도
    weight_trade_value_ratio: int = 15    # 보너스 대비 거래액
    
    # ===== Tier Thresholds (Tier 판정 기준) =====
    bot_tier_threshold: int = 90          # Bot Tier 최소 점수
    manual_tier_threshold: int = 70       # Manual Tier 최소 점수
    suspicious_threshold: int = 50        # Suspicious 최소 점수
    
    # ===== Network Analysis Parameters =====
    min_profit_occurrences: int = 2       # 제재 대상 최소 수익 횟수
    max_network_depth: int = 5            # 네트워크 탐색 최대 깊이
    
    # ===== Output Settings =====
    output_dir: str = "output/bonus"
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
    pnl_mirroring: float = 0.0
    high_concurrency: float = 0.0
    quantity_match: float = 0.0
    trade_value_ratio: float = 0.0
    total: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradePair:
    """거래 쌍 정보"""
    # 기본 정보
    pair_id: str
    loser_account: str
    winner_account: str
    symbol: str
    
    # 시간 정보
    loser_open_ts: datetime
    loser_close_ts: datetime
    winner_open_ts: datetime
    winner_close_ts: datetime
    bonus_ts: datetime
    
    # 거래 정보
    loser_side: str  # LONG or SHORT
    winner_side: str
    loser_amount: float
    winner_amount: float
    loser_leverage: int
    winner_leverage: int
    
    # 손익 정보
    loser_pnl: float
    winner_pnl: float
    linked_bonus: float
    
    # 입금 정보 (Trade Value Ratio 계산용)
    loser_deposit: float = 0.0
    winner_deposit: float = 0.0
    
    # 계산된 메트릭
    open_time_diff_sec: float = 0.0
    close_time_diff_sec: float = 0.0
    amount_diff_pct: float = 0.0
    time_since_bonus_hours: float = 0.0
    
    # 점수 및 판정
    score: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    tier: TierType = TierType.NORMAL
    
    # 필터 통과 여부
    passed_filter: bool = False
    filter_failures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화 가능)"""
        data = asdict(self)
        # Enum과 datetime 처리
        data['tier'] = self.tier.value
        data['loser_open_ts'] = self.loser_open_ts.isoformat()
        data['loser_close_ts'] = self.loser_close_ts.isoformat()
        data['winner_open_ts'] = self.winner_open_ts.isoformat()
        data['winner_close_ts'] = self.winner_close_ts.isoformat()
        data['bonus_ts'] = self.bonus_ts.isoformat()
        return data


@dataclass
class SanctionCase:
    """제재 케이스"""
    case_id: str
    sanction_type: SanctionType
    account_ids: List[str]
    detection_timestamp: datetime
    
    # 증거 데이터
    trade_pair_ids: List[str]
    total_score: float
    tier: TierType
    
    # 네트워크 정보 (NETWORK 타입만)
    network_path: Optional[List[str]] = None
    profit_occurrence_count: Optional[int] = None
    
    # 추가 메타데이터
    total_laundered_amount: float = 0.0
    evidence_summary: str = ""
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['sanction_type'] = self.sanction_type.value
        data['tier'] = self.tier.value
        data['detection_timestamp'] = self.detection_timestamp.isoformat()
        return data


@dataclass
class NetworkNode:
    """네트워크 노드 (수익 계정)"""
    account_id: str
    profit_count: int = 0
    total_profit: float = 0.0
    connected_losers: Set[str] = field(default_factory=set)
    trade_pair_ids: List[str] = field(default_factory=list)
    
    def add_profit_link(self, loser_account: str, profit: float, pair_id: str):
        """수익 연결 추가"""
        self.profit_count += 1
        self.total_profit += profit
        self.connected_losers.add(loser_account)
        self.trade_pair_ids.append(pair_id)


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
        self.logger.info("증정금 녹이기 탐지 시스템 시작")
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
    
    def log_tier_distribution(self, tier_counts: Dict[TierType, int]):
        """Tier 분포 로그"""
        self.logger.info("Tier 분포:")
        for tier, count in tier_counts.items():
            self.logger.info(f"  - {tier.value}: {count}건")
    
    def log_sanction_case(self, case: SanctionCase):
        """제재 케이스 로그"""
        self.logger.warning(f"제재 케이스 생성: {case.case_id}")
        self.logger.warning(f"  - 유형: {case.sanction_type.value}")
        self.logger.warning(f"  - 계정: {', '.join(case.account_ids)}")
        self.logger.warning(f"  - 점수: {case.total_score:.2f}")


# ============================================================================
# 3. DATA PIPELINE
# ============================================================================
# DataLoader removed - using common.data_manager.DataManager singleton instead.


class PositionBuilder:
    """포지션 데이터 구성"""
    
    def __init__(self, con: dd.DuckDBPyConnection):
        self.con = con
    
    def build_positions(self) -> pd.DataFrame:
        """포지션별 집계 데이터 생성"""
        print("포지션 데이터 구성 중...")
        
        query = """
        SELECT 
            account_id,
            position_id,
            MAX(leverage) AS leverage,
            CAST(MIN(ts) AS TIMESTAMP) as open_ts,
            CAST(MAX(ts) AS TIMESTAMP) as close_ts,
            MAX(symbol) as symbol, 
            MAX(side) as side,
            SUM(CASE WHEN openclose='OPEN' THEN amount ELSE 0 END) as amount,
            SUM(
                CASE WHEN openclose='OPEN' THEN -amount ELSE amount END * 
                CASE WHEN side='LONG' THEN 1 ELSE -1 END
            ) as pnl
        FROM Trade
        GROUP BY account_id, position_id
        HAVING pnl != 0
        ORDER BY open_ts
        """
        
        df = self.con.execute(query).fetchdf()
        print(f"포지션 {len(df)}개 생성 완료")
        
        return df
    
    def build_bonuses(self) -> pd.DataFrame:
        """보너스 데이터 생성"""
        print("보너스 데이터 구성 중...")
        
        query = """
        SELECT 
            account_id, 
            CAST(ts AS TIMESTAMP) as bonus_ts, 
            reward_amount
        FROM Reward
        ORDER BY bonus_ts
        """
        
        df = self.con.execute(query).fetchdf()
        print(f"보너스 {len(df)}개 생성 완료")
        
        return df


# ============================================================================
# 4. FILTER ENGINE
# ============================================================================

class FilterEngine:
    """1단계: 필수 조건 필터링"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def apply_filters(self, candidate_pairs: List[Dict]) -> Tuple[List[TradePair], List[Dict]]:
        """
        필터 적용 및 TradePair 객체 생성
        
        Returns:
            (통과한 TradePair 리스트, 실패한 원본 데이터 리스트)
        """
        print("필터 엔진 시작...")
        
        passed_pairs = []
        failed_data = []
        
        for idx, row in enumerate(candidate_pairs):
            pair_id = f"PAIR_{idx:06d}"
            failures = []
            
            # Filter 1: Time Since Bonus
            if not self._check_time_since_bonus(row):
                failures.append("time_since_bonus")
            
            # Filter 2: Reverse Position
            if not self._check_reverse_position(row):
                failures.append("reverse_position")
            
            # Filter 3: Equal Leverage
            if not self._check_equal_leverage(row):
                failures.append("equal_leverage")
            
            # Filter 4: Concurrency
            if not self._check_concurrency(row):
                failures.append("concurrency")
            
            # Filter 5: Quantity Match
            if not self._check_quantity_match(row):
                failures.append("quantity_match")
            
            # 모든 필터 통과 여부
            if len(failures) == 0:
                # TradePair 객체 생성
                trade_pair = self._create_trade_pair(pair_id, row)
                trade_pair.passed_filter = True
                passed_pairs.append(trade_pair)
            else:
                # 실패 정보 기록
                row['pair_id'] = pair_id
                row['filter_failures'] = failures
                failed_data.append(row)
                
                # if self.config.enable_detailed_logging:
                #     print(f"{pair_id} 필터 실패: {', '.join(failures)}")
        
        print(f"필터 완료: {len(passed_pairs)}/{len(candidate_pairs)} 통과")
        
        return passed_pairs, failed_data
    
    def _check_time_since_bonus(self, row: Dict) -> bool:
        """보너스 후 72시간 이내 확인"""
        hours = row.get('time_since_bonus_hours', 0)
        return 0 <= hours <= self.config.time_since_bonus_hours
    
    def _check_reverse_position(self, row: Dict) -> bool:
        """반대 포지션 확인"""
        loser_side = row.get('loser_side', '')
        winner_side = row.get('winner_side', '')
        return loser_side != winner_side and loser_side in ['LONG', 'SHORT'] and winner_side in ['LONG', 'SHORT']
    
    def _check_equal_leverage(self, row: Dict) -> bool:
        """동일 레버리지 확인"""
        loser_lev = row.get('loser_leverage', 0)
        winner_lev = row.get('winner_leverage', 0)
        return loser_lev == winner_lev and loser_lev > 0
    
    def _check_concurrency(self, row: Dict) -> bool:
        """30초 이내 동시성 확인"""
        open_diff = row.get('open_time_diff_sec', float('inf'))
        close_diff = row.get('close_time_diff_sec', float('inf'))
        threshold = self.config.concurrency_threshold_sec
        return open_diff <= threshold and close_diff <= threshold
    
    def _check_quantity_match(self, row: Dict) -> bool:
        """±2% 수량 매칭 확인"""
        diff_ratio = row.get('amount_diff_ratio', float('inf'))
        return diff_ratio <= self.config.quantity_tolerance_pct
    
    def _create_trade_pair(self, pair_id: str, row: Dict) -> TradePair:
        """TradePair 객체 생성"""
        return TradePair(
            pair_id=pair_id,
            loser_account=row['loser_account'],
            winner_account=row['winner_account'],
            symbol=row['symbol'],
            loser_open_ts=row['loser_open_ts'],
            loser_close_ts=row['loser_close_ts'],
            winner_open_ts=row['winner_open_ts'],
            winner_close_ts=row['winner_close_ts'],
            bonus_ts=row['bonus_ts'],
            loser_side=row['loser_side'],
            winner_side=row['winner_side'],
            loser_amount=row['loser_amount'],
            winner_amount=row['winner_amount'],
            loser_leverage=row['loser_leverage'],
            winner_leverage=row['winner_leverage'],
            loser_pnl=row['loser_pnl'],
            winner_pnl=row['winner_pnl'],
            linked_bonus=row['reward_amount'],
            loser_deposit=row.get('loser_deposit', 0.0),
            winner_deposit=row.get('winner_deposit', 0.0),
            open_time_diff_sec=row['open_time_diff_sec'],
            close_time_diff_sec=row['close_time_diff_sec'],
            amount_diff_pct=row['amount_diff_ratio'] * 100,
            time_since_bonus_hours=row['time_since_bonus_hours'],
        )


# ============================================================================
# 5. SCORING ENGINE
# ============================================================================

class ScoringEngine:
    """2단계: 점수 계산 및 Tier 분류"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def score_all_pairs(self, pairs: List[TradePair]) -> List[TradePair]:
        """모든 거래 쌍 점수 계산"""
        print("점수 엔진 시작...")
        
        for pair in pairs:
            # 각 지표별 점수 계산
            pnl_score = self._score_pnl_mirroring(pair)
            conc_score = self._score_high_concurrency(pair)
            qty_score = self._score_quantity_match(pair)
            ratio_score = self._score_trade_value_ratio(pair)
            
            # 점수 저장
            pair.score = ScoreBreakdown(
                pnl_mirroring=pnl_score,
                high_concurrency=conc_score,
                quantity_match=qty_score,
                trade_value_ratio=ratio_score,
                total=pnl_score + conc_score + qty_score + ratio_score
            )
            
            # Tier 분류
            pair.tier = self._classify_tier(pair.score.total)
            
            # if self.config.enable_detailed_logging:
            #     print(
            #         f"{pair.pair_id}: 점수={pair.score.total:.1f} "
            #         f"(PnL:{pnl_score:.1f}, Conc:{conc_score:.1f}, "
            #         f"Qty:{qty_score:.1f}, Ratio:{ratio_score:.1f}) "
            #         f"→ {pair.tier.value}"
            #     )
        
        print(f"점수 계산 완료: {len(pairs)}개")
        
        return pairs
    
    def _score_pnl_mirroring(self, pair: TradePair) -> float:
        """
        P&L 대칭성 점수 (40점)
        완벽한 헤징: PnL_A + PnL_B ≈ 0
        """
        max_weight = self.config.weight_pnl_mirroring
        
        total_pnl = pair.loser_pnl + pair.winner_pnl
        max_pnl = max(abs(pair.loser_pnl), abs(pair.winner_pnl))
        
        if max_pnl == 0:
            return 0.0
        
        # 대칭 비율: 0에 가까울수록 완벽
        asymmetry_ratio = abs(total_pnl) / max_pnl
        
        if asymmetry_ratio <= 0.01:  # 1% 이내: 거의 완벽
            return max_weight
        elif asymmetry_ratio <= 0.05:  # 5% 이내: 매우 좋음
            return max_weight * 0.85
        elif asymmetry_ratio <= 0.10:  # 10% 이내: 좋음
            return max_weight * 0.65
        elif asymmetry_ratio <= 0.20:  # 20% 이내: 보통
            return max_weight * 0.40
        else:
            return max_weight * 0.10
    
    def _score_high_concurrency(self, pair: TradePair) -> float:
        """
        시간 근접도 점수 (25점)
        오픈/클로즈 평균 시간차
        """
        max_weight = self.config.weight_high_concurrency
        
        avg_time_diff = (pair.open_time_diff_sec + pair.close_time_diff_sec) / 2
        
        if avg_time_diff <= 0.1:  # 0.1초 이내: 봇
            return max_weight
        elif avg_time_diff <= 1.0:  # 1초 이내: 매우 의심
            return max_weight * 0.90
        elif avg_time_diff <= 5.0:  # 5초 이내: 의심
            return max_weight * 0.70
        elif avg_time_diff <= 10.0:  # 10초 이내
            return max_weight * 0.45
        elif avg_time_diff <= 20.0:  # 20초 이내
            return max_weight * 0.25
        else:  # 30초 이내 (필터 통과 범위)
            return max_weight * 0.10
    
    def _score_quantity_match(self, pair: TradePair) -> float:
        """
        수량 일치도 점수 (20점)
        """
        max_weight = self.config.weight_quantity_match
        
        diff_pct = pair.amount_diff_pct
        
        if diff_pct <= 0.1:  # 0.1% 이내: 거의 완벽
            return max_weight
        elif diff_pct <= 0.5:  # 0.5% 이내: 매우 좋음
            return max_weight * 0.85
        elif diff_pct <= 1.0:  # 1% 이내: 좋음
            return max_weight * 0.65
        elif diff_pct <= 1.5:  # 1.5% 이내: 보통
            return max_weight * 0.40
        else:  # 2% 이내 (필터 통과 범위)
            return max_weight * 0.20
    
    def _score_trade_value_ratio(self, pair: TradePair) -> float:
        """
        보너스 대비 거래액 점수 (15점)
        거래 증거금 / (입금액 + 보너스) 비율
        """
        max_weight = self.config.weight_trade_value_ratio
        
        # 거래 증거금 = amount / leverage
        loser_margin = pair.loser_amount / pair.loser_leverage if pair.loser_leverage > 0 else 0
        loser_capital = pair.loser_deposit + pair.linked_bonus
        
        if loser_capital == 0:
            return 0.0
        
        ratio = loser_margin / loser_capital
        
        if ratio >= 0.95:  # 95% 이상: 올인
            return max_weight
        elif ratio >= 0.80:  # 80% 이상
            return max_weight * 0.75
        elif ratio >= 0.60:  # 60% 이상
            return max_weight * 0.50
        elif ratio >= 0.40:  # 40% 이상
            return max_weight * 0.25
        else:
            return 0.0
    
    def _classify_tier(self, total_score: float) -> TierType:
        """점수 기반 Tier 분류"""
        if total_score >= self.config.bot_tier_threshold:
            return TierType.BOT
        elif total_score >= self.config.manual_tier_threshold:
            return TierType.MANUAL
        elif total_score >= self.config.suspicious_threshold:
            return TierType.SUSPICIOUS
        else:
            return TierType.NORMAL


# ============================================================================
# 6. NETWORK ANALYZER
# ============================================================================

class NetworkAnalyzer:
    """수익 계정 네트워킹 분석"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.profit_network: Dict[str, NetworkNode] = {}
    
    def analyze_manual_tier_pairs(self, pairs: List[TradePair]) -> Dict[str, NetworkNode]:
        """
        Manual Tier 거래 쌍에서 수익 계정 네트워크 구성
        
        Returns:
            수익 계정 ID -> NetworkNode 매핑
        """
        print("네트워크 분석 시작...")
        
        # Manual Tier만 필터링
        manual_pairs = [p for p in pairs if p.tier == TierType.MANUAL]
        print(f"Manual Tier 거래: {len(manual_pairs)}건")
        
        if len(manual_pairs) == 0:
            print("네트워크 분석 대상 없음")
            return {}
        
        # 수익 계정 네트워크 구성
        for pair in manual_pairs:
            winner = pair.winner_account
            loser = pair.loser_account
            profit = pair.winner_pnl
            
            if winner not in self.profit_network:
                self.profit_network[winner] = NetworkNode(account_id=winner)
            
            self.profit_network[winner].add_profit_link(loser, profit, pair.pair_id)
        
        # 반복 수익 계정 찾기
        repeat_accounts = {
            acc: node for acc, node in self.profit_network.items()
            if node.profit_count >= self.config.min_profit_occurrences
        }
        
        print(f"수익 계정 총 {len(self.profit_network)}개")
        print(f"반복 수익 계정 (>={self.config.min_profit_occurrences}회): {len(repeat_accounts)}개")
        
        return self.profit_network
    
    def find_network_chains(self) -> List[List[str]]:
        """
        연결된 계정 체인 탐지 (A → B → C)
        
        Returns:
            계정 체인 리스트 (예: [['A', 'B', 'C'], ['D', 'E']])
        """
        print("계정 체인 탐색 중...")
        
        chains = []
        visited = set()
        
        for winner_account in self.profit_network.keys():
            if winner_account in visited:
                continue
            
            # DFS로 체인 탐색
            chain = self._dfs_chain(winner_account, visited, set())
            
            if len(chain) >= 2:  # 최소 2개 이상 연결
                chains.append(chain)
                print(f"체인 발견: {' → '.join(chain)}")
        
        print(f"총 {len(chains)}개 체인 발견")
        
        return chains
    
    def _dfs_chain(self, account: str, visited: Set[str], current_path: Set[str]) -> List[str]:
        """DFS로 체인 탐색 (순환 방지)"""
        if account in current_path:  # 순환 감지
            return []
        
        if account not in self.profit_network:  # 더 이상 연결 없음
            return [account]
        
        visited.add(account)
        current_path.add(account)
        
        node = self.profit_network[account]
        
        # 가장 긴 체인 찾기
        longest_chain = [account]
        
        for loser in node.connected_losers:
            if loser in self.profit_network:  # loser도 다른 거래에서 winner인 경우
                sub_chain = self._dfs_chain(loser, visited, current_path.copy())
                if len(sub_chain) + 1 > len(longest_chain):
                    longest_chain = [account] + sub_chain
        
        return longest_chain
    
    def get_network_statistics(self) -> Dict:
        """네트워크 통계"""
        if not self.profit_network:
            return {}
        
        profit_counts = [node.profit_count for node in self.profit_network.values()]
        total_profits = [node.total_profit for node in self.profit_network.values()]
        
        return {
            'total_profit_accounts': len(self.profit_network),
            'repeat_accounts': sum(1 for c in profit_counts if c >= self.config.min_profit_occurrences),
            'max_profit_count': max(profit_counts) if profit_counts else 0,
            'total_network_profit': sum(total_profits),
            'avg_profit_per_account': sum(total_profits) / len(total_profits) if total_profits else 0,
        }


# ============================================================================
# 7. SANCTION PIPELINE
# ============================================================================

class SanctionPipeline:
    """제재 케이스 생성 및 출력"""
    
    def __init__(self, config: DetectionConfig, logger: DetectionLogger):
        self.config = config
        self.logger = logger
        self.sanction_cases: List[SanctionCase] = []
    
    def process_bot_tier(self, pairs: List[TradePair]) -> List[SanctionCase]:
        """Bot Tier 즉시 제재 케이스 생성"""
        print("Bot Tier 제재 케이스 생성 중...")
        
        bot_pairs = [p for p in pairs if p.tier == TierType.BOT]
        
        if len(bot_pairs) == 0:
            print("Bot Tier 거래 없음")
            return []
        
        for pair in bot_pairs:
            case = SanctionCase(
                case_id=f"SANCTION_BOT_{pair.pair_id}",
                sanction_type=SanctionType.IMMEDIATE_BOT,
                account_ids=[pair.loser_account, pair.winner_account],
                detection_timestamp=datetime.now(),
                trade_pair_ids=[pair.pair_id],
                total_score=pair.score.total,
                tier=TierType.BOT,
                total_laundered_amount=pair.winner_pnl,
                evidence_summary=f"완벽한 봇 패턴 탐지 (점수: {pair.score.total:.1f}/100)"
            )
            
            self.sanction_cases.append(case)
            self.logger.log_sanction_case(case)
        
        print(f"Bot Tier 제재: {len(bot_pairs)}건")
        
        return [c for c in self.sanction_cases if c.sanction_type == SanctionType.IMMEDIATE_BOT]
    
    def process_network_analysis(
        self, 
        network: Dict[str, NetworkNode], 
        chains: List[List[str]],
        pairs: List[TradePair]
    ) -> List[SanctionCase]:
        """네트워크 분석 기반 제재 케이스 생성"""
        print("네트워크 제재 케이스 생성 중...")
        
        network_cases = []
        
        # 1. 반복 수익 계정 제재
        repeat_accounts = {
            acc: node for acc, node in network.items()
            if node.profit_count >= self.config.min_profit_occurrences
        }
        
        for account_id, node in repeat_accounts.items():
            case = SanctionCase(
                case_id=f"SANCTION_REPEAT_{account_id}",
                sanction_type=SanctionType.NETWORK_REPEAT,
                account_ids=[account_id],
                detection_timestamp=datetime.now(),
                trade_pair_ids=node.trade_pair_ids,
                total_score=self._calculate_avg_score(node.trade_pair_ids, pairs),
                tier=TierType.MANUAL,
                profit_occurrence_count=node.profit_count,
                total_laundered_amount=node.total_profit,
                evidence_summary=f"반복 수익 계정 ({node.profit_count}회 수익)"
            )
            
            network_cases.append(case)
            self.sanction_cases.append(case)
            self.logger.log_sanction_case(case)
        
        # 2. 연결된 체인 제재
        for chain in chains:
            if len(chain) >= 2:
                # 체인 관련 거래 쌍 찾기
                chain_pair_ids = []
                chain_accounts = set(chain)
                
                for pair in pairs:
                    if pair.winner_account in chain_accounts or pair.loser_account in chain_accounts:
                        chain_pair_ids.append(pair.pair_id)
                
                case = SanctionCase(
                    case_id=f"SANCTION_CHAIN_{'_'.join(chain[:3])}",
                    sanction_type=SanctionType.NETWORK_CHAIN,
                    account_ids=chain,
                    detection_timestamp=datetime.now(),
                    trade_pair_ids=chain_pair_ids,
                    total_score=self._calculate_avg_score(chain_pair_ids, pairs),
                    tier=TierType.MANUAL,
                    network_path=chain,
                    total_laundered_amount=sum(
                        network[acc].total_profit for acc in chain if acc in network
                    ),
                    evidence_summary=f"연결된 계정 체인 ({len(chain)}개 계정)"
                )
                
                network_cases.append(case)
                self.sanction_cases.append(case)
                self.logger.log_sanction_case(case)
        
        print(f"네트워크 제재: {len(network_cases)}건")
        
        return network_cases
    
    def _calculate_avg_score(self, pair_ids: List[str], pairs: List[TradePair]) -> float:
        """평균 점수 계산"""
        scores = [p.score.total for p in pairs if p.pair_id in pair_ids]
        return sum(scores) / len(scores) if scores else 0.0
    
    def export_sanctions(self, output_dir: Path) -> str:
        """제재 케이스를 JSON 파일로 출력"""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "sanction_cases.json"
        
        data = {
            'total_cases': len(self.sanction_cases),
            'generated_at': datetime.now().isoformat(),
            'cases': [case.to_dict() for case in self.sanction_cases]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"제재 케이스 저장: {filepath} ({len(self.sanction_cases)}건)")
        
        return str(filepath)


# ============================================================================
# 8. REPORTING & VISUALIZATION DATA
# ============================================================================

class ReportGenerator:
    """분석 보고서 및 시각화 데이터 생성"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
    
    def generate_all_reports(
        self,
        all_pairs: List[TradePair],
        sanction_cases: List[SanctionCase],
        network_stats: Dict
    ):
        """모든 보고서 생성"""
        print("보고서 생성 중...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 거래 쌍 상세 데이터 (CSV)
        self._export_trade_pairs_csv(all_pairs)
        
        # 2. 시각화용 JSON 데이터
        self._export_visualization_data(all_pairs, sanction_cases, network_stats)
        
        # 3. 요약 보고서 (텍스트)
        self._generate_summary_report(all_pairs, sanction_cases, network_stats)
        
        print("보고서 생성 완료")
    
    def _export_trade_pairs_csv(self, pairs: List[TradePair]):
        """거래 쌍 상세 CSV"""
        if not pairs:
            return
        
        # TradePair를 DataFrame으로 변환
        records = []
        for pair in pairs:
            record = {
                'pair_id': pair.pair_id,
                'tier': pair.tier.value,
                'total_score': pair.score.total,
                'loser_account': pair.loser_account,
                'winner_account': pair.winner_account,
                'symbol': pair.symbol,
                'loser_pnl': pair.loser_pnl,
                'winner_pnl': pair.winner_pnl,
                'laundered_amount': pair.winner_pnl,
                'linked_bonus': pair.linked_bonus,
                'time_since_bonus_hours': pair.time_since_bonus_hours,
                'open_time_diff_sec': pair.open_time_diff_sec,
                'close_time_diff_sec': pair.close_time_diff_sec,
                'amount_diff_pct': pair.amount_diff_pct,
                'leverage': pair.loser_leverage,
                'score_pnl_mirroring': pair.score.pnl_mirroring,
                'score_concurrency': pair.score.high_concurrency,
                'score_quantity': pair.score.quantity_match,
                'score_trade_ratio': pair.score.trade_value_ratio,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        filepath = self.output_dir / "trade_pairs_detailed.csv"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"거래 쌍 CSV 저장: {filepath}")
    
    def _export_visualization_data(
        self, 
        pairs: List[TradePair],
        sanction_cases: List[SanctionCase],
        network_stats: Dict
    ):
        """시각화용 JSON 데이터"""
        
        # Tier별 분포
        tier_counts = defaultdict(int)
        for pair in pairs:
            tier_counts[pair.tier.value] += 1
        
        # 점수 분포 (10점 단위 구간)
        score_distribution = defaultdict(int)
        for pair in pairs:
            bucket = int(pair.score.total // 10) * 10
            score_distribution[f"{bucket}-{bucket+10}"] += 1
        
        # 시간대별 패턴
        time_patterns = self._analyze_time_patterns(pairs)
        
        # 네트워크 그래프 데이터
        network_graph = self._build_network_graph_data(pairs)
        
        vis_data = {
            'summary': {
                'total_pairs': len(pairs),
                'bot_tier': tier_counts.get('BOT', 0),
                'manual_tier': tier_counts.get('MANUAL', 0),
                'suspicious': tier_counts.get('SUSPICIOUS', 0),
                'normal': tier_counts.get('NORMAL', 0),
                'total_sanctions': len(sanction_cases),
            },
            'tier_distribution': dict(tier_counts),
            'score_distribution': dict(score_distribution),
            'time_patterns': time_patterns,
            'network_graph': network_graph,
            'network_statistics': network_stats,
        }
        
        filepath = self.output_dir / "visualization_data.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vis_data, f, indent=2, ensure_ascii=False)
        
        print(f"시각화 데이터 저장: {filepath}")
    
    def _analyze_time_patterns(self, pairs: List[TradePair]) -> Dict:
        """시간대별 패턴 분석"""
        hourly_dist = defaultdict(int)
        
        for pair in pairs:
            hour = pair.loser_open_ts.hour
            hourly_dist[hour] += 1
        
        return {
            'hourly_distribution': dict(hourly_dist),
            'peak_hour': max(hourly_dist.items(), key=lambda x: x[1])[0] if hourly_dist else None,
        }
    
    def _build_network_graph_data(self, pairs: List[TradePair]) -> Dict:
        """네트워크 그래프 데이터 (nodes, edges)"""
        nodes = set()
        edges = []
        
        for pair in pairs:
            if pair.tier in [TierType.BOT, TierType.MANUAL]:
                nodes.add(pair.loser_account)
                nodes.add(pair.winner_account)
                
                edges.append({
                    'source': pair.loser_account,
                    'target': pair.winner_account,
                    'value': pair.winner_pnl,
                    'tier': pair.tier.value,
                    'score': pair.score.total,
                })
        
        return {
            'nodes': [{'id': node} for node in nodes],
            'edges': edges,
        }
    
    def _generate_summary_report(
        self,
        pairs: List[TradePair],
        sanction_cases: List[SanctionCase],
        network_stats: Dict
    ):
        """요약 보고서 텍스트"""
        
        tier_counts = defaultdict(int)
        for pair in pairs:
            tier_counts[pair.tier] += 1
        
        total_laundered = sum(p.winner_pnl for p in pairs if p.tier != TierType.NORMAL)
        
        report = f"""
{'='*70}
증정금 녹이기 탐지 보고서
{'='*70}

📊 탐지 요약
  - 총 분석 거래 쌍: {len(pairs)}건
  - Bot Tier (즉시 제재): {tier_counts[TierType.BOT]}건
  - Manual Tier (네트워크 분석): {tier_counts[TierType.MANUAL]}건
  - Suspicious (모니터링): {tier_counts[TierType.SUSPICIOUS]}건
  - Normal: {tier_counts[TierType.NORMAL]}건

💰 증정금 현금화 규모
  - 총 현금화 금액: ${total_laundered:,.2f}

🚨 제재 케이스
  - 총 제재 케이스: {len(sanction_cases)}건
  - Bot 즉시 제재: {sum(1 for c in sanction_cases if c.sanction_type == SanctionType.IMMEDIATE_BOT)}건
  - 네트워크 반복 제재: {sum(1 for c in sanction_cases if c.sanction_type == SanctionType.NETWORK_REPEAT)}건
  - 네트워크 체인 제재: {sum(1 for c in sanction_cases if c.sanction_type == SanctionType.NETWORK_CHAIN)}건

🔗 네트워크 분석
"""
        
        if network_stats:
            report += f"""  - 수익 계정 수: {network_stats.get('total_profit_accounts', 0)}개
  - 반복 수익 계정: {network_stats.get('repeat_accounts', 0)}개
  - 최대 수익 횟수: {network_stats.get('max_profit_count', 0)}회
  - 네트워크 총 수익: ${network_stats.get('total_network_profit', 0):,.2f}
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
# 9. MAIN DETECTOR ENGINE
# ============================================================================

class BonusLaunderingDetector:
    """메인 탐지 엔진"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger(config)
        
        # 엔진 컴포넌트
        self.filter_engine = FilterEngine(config)
        self.scoring_engine = ScoringEngine(config)
        self.network_analyzer = NetworkAnalyzer(config)
        self.sanction_pipeline = SanctionPipeline(config, self.logger)
        self.report_generator = ReportGenerator(config)
    
    def detect(self, data_filepath: str) -> Dict:
        """
        전체 탐지 프로세스 실행
        
        Returns:
            탐지 결과 딕셔너리
        """
        print("데이터 로드")
        
        # 1. 데이터 로드 (공통 DataManager 사용)
        dm = get_data_manager(data_filepath)
        dm.get_all_sheets()
        con = dm.get_connection()
        
        # 2. 포지션 구성
        print("포지션 구성")
        builder = PositionBuilder(con)
        positions = builder.build_positions()
        bonuses = builder.build_bonuses()
        
        # 3. 후보 쌍 추출
        print("후보 쌍 추출")
        candidate_pairs = self._extract_candidate_pairs(con, positions, bonuses)
        
        if len(candidate_pairs) == 0:
            print("후보 쌍이 없습니다. 탐지 종료.")
            return self._empty_result()
        
        # 4. 필터 적용
        print("필터 적용")
        passed_pairs, failed_pairs = self.filter_engine.apply_filters(candidate_pairs)
        
        if len(passed_pairs) == 0:
            print("필터를 통과한 거래 쌍이 없습니다.")
            return self._empty_result()
        
        # 5. 점수 계산
        print("점수 계산 및 Tier 분류")
        scored_pairs = self.scoring_engine.score_all_pairs(passed_pairs)
        
        # Tier 분포 로깅
        tier_counts = defaultdict(int)
        for pair in scored_pairs:
            tier_counts[pair.tier] += 1
        self.logger.log_tier_distribution(tier_counts)
        
        # 6. Bot Tier 즉시 제재
        print("Bot Tier 제재")
        bot_sanctions = self.sanction_pipeline.process_bot_tier(scored_pairs)
        
        # 7. Manual Tier 네트워크 분석
        print("네트워크 분석")
        network = self.network_analyzer.analyze_manual_tier_pairs(scored_pairs)
        chains = self.network_analyzer.find_network_chains()
        network_stats = self.network_analyzer.get_network_statistics()
        
        # 8. 네트워크 기반 제재
        print("네트워크 제재")
        network_sanctions = self.sanction_pipeline.process_network_analysis(
            network, chains, scored_pairs
        )
        
        # 9. 제재 케이스 출력
        print("제재 케이스 출력")
        sanction_file = self.sanction_pipeline.export_sanctions(Path(self.config.output_dir))
        
        # 10. 보고서 생성
        print("보고서 생성")
        all_sanctions = bot_sanctions + network_sanctions
        self.report_generator.generate_all_reports(scored_pairs, all_sanctions, network_stats)
        
        # 11. 결과 반환
        return {
            'config': self.config.to_dict(),
            'total_candidates': len(candidate_pairs),
            'passed_filter': len(passed_pairs),
            'tier_distribution': {tier.value: count for tier, count in tier_counts.items()},
            'sanction_cases': len(all_sanctions),
            'bot_sanctions': len(bot_sanctions),
            'network_sanctions': len(network_sanctions),
            'network_statistics': network_stats,
            'output_directory': self.config.output_dir,
        }
    
    def _extract_candidate_pairs(
        self, 
        con: dd.DuckDBPyConnection,
        positions: pd.DataFrame,
        bonuses: pd.DataFrame
    ) -> List[Dict]:
        """SQL을 통해 후보 거래 쌍 추출"""
        
        # DuckDB에 등록
        con.register('positions', positions)
        con.register('bonuses', bonuses)
        
        query = f"""
        -- 1. 보너스를 받은 기록이 있는 '손실 거래'만 미리 필터링합니다.
        WITH losers_with_bonus AS (
            SELECT
                p.account_id,
                p.open_ts,
                p.close_ts,
                p.symbol,
                p.side,
                p.amount,
                p.leverage,
                p.pnl,
                b.bonus_ts,
                b.reward_amount
            FROM positions p
            JOIN bonuses b ON p.account_id = b.account_id
            WHERE p.pnl < 0 -- 손실 거래
              AND b.bonus_ts <= p.open_ts -- 거래 전에 보너스를 받음
        ),

        -- 2. 필터링된 '손실 거래'와 '이익 거래'를 강력한 조건으로 조인합니다.
        candidate_pairs AS (
            SELECT
                t1.account_id AS loser_account,
                t2.account_id AS winner_account,
                t1.open_ts AS loser_open_ts,
                t1.close_ts AS loser_close_ts,
                t2.open_ts AS winner_open_ts,
                t2.close_ts AS winner_close_ts,
                t1.symbol,
                t1.side AS loser_side,
                t2.side AS winner_side,
                t1.amount AS loser_amount,
                t2.amount AS winner_amount,
                t1.leverage AS loser_leverage,
                t2.leverage AS winner_leverage,
                t1.pnl AS loser_pnl,
                t2.pnl AS winner_pnl,
                t1.bonus_ts,
                t1.reward_amount,

                ABS(epoch(t1.open_ts - t2.open_ts)) AS open_time_diff_sec,
                ABS(epoch(t1.close_ts - t2.close_ts)) AS close_time_diff_sec,
                ABS(t1.amount - t2.amount) / LEAST(t1.amount, t2.amount) AS amount_diff_ratio,
                epoch(t1.open_ts - t1.bonus_ts) / 3600.0 AS time_since_bonus_hours
                
            FROM losers_with_bonus t1 -- (매우 작아진 t1 세트)
            JOIN positions t2 ON -- (전체 t2 세트)
                t1.account_id != t2.account_id
                AND t1.side != t2.side
                AND t1.symbol = t2.symbol
                AND t2.pnl > 0 -- t2는 이익 거래
                
                -- [핵심 필터 1] 시간 제한: 두 거래의 오픈 시간이 5분 이내
                AND t2.open_ts BETWEEN (t1.open_ts - INTERVAL '5 minutes') AND (t1.open_ts + INTERVAL '5 minutes')
                
                -- [핵심 필터 2] 금액 제한: 두 거래의 금액 차이가 10% 이내
                AND ABS(t1.amount - t2.amount) / LEAST(t1.amount, t2.amount) < 0.1 
        )

        -- 3. 최종 결과 선택 (이제 'bonuses' 테이블을 다시 조인할 필요가 없습니다.)
        SELECT
            cp.*,
            0.0 AS loser_deposit,  -- 이 컬럼들은 로직을 추가해야 합니다.
            0.0 AS winner_deposit
        FROM candidate_pairs cp
        ORDER BY time_since_bonus_hours ASC, open_time_diff_sec ASC;
        """
        
        df = con.execute(query).fetchdf()
        print(f"후보 쌍 {len(df)}개 추출")
        
        return df.to_dict('records')
    
    def _empty_result(self) -> Dict:
        """빈 결과 반환"""
        return {
            'config': self.config.to_dict(),
            'total_candidates': 0,
            'passed_filter': 0,
            'tier_distribution': {},
            'sanction_cases': 0,
            'bot_sanctions': 0,
            'network_sanctions': 0,
            'network_statistics': {},
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
    증정금 녹이기 탐지 실행
    
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
    detector = BonusLaunderingDetector(config)
    result = detector.detect(data_filepath)
    
    return result


if __name__ == "__main__":
    # 커스텀 설정
    custom_config = DetectionConfig(
        # Filter 파라미터
        time_since_bonus_hours=72.0,
        concurrency_threshold_sec=30.0,
        quantity_tolerance_pct=0.02,
        
        # Tier 임계값
        bot_tier_threshold=90,
        manual_tier_threshold=70,
        suspicious_threshold=50,
        
        # 네트워크 파라미터
        min_profit_occurrences=2,
        max_network_depth=5,
        
        # 출력 설정
        output_dir="./output/bonus",
        enable_detailed_logging=True
    )
    
    # 실행
    print("\n" + "="*70)
    print("증정금 녹이기 탐지 시스템 v2.0")
    print("="*70 + "\n")
    
    result = run_detection(
        data_filepath="problem_data_final.xlsx",
        config=custom_config
    )
    
    # 결과 요약 출력
    print("\n" + "="*70)
    print("탐지 완료!")
    print("="*70)
    print(f"총 후보 쌍: {result['total_candidates']}건")
    print(f"필터 통과: {result['passed_filter']}건")
    print(f"\nTier 분포:")
    for tier, count in result['tier_distribution'].items():
        print(f"  - {tier}: {count}건")
    print(f"\n제재 케이스: {result['sanction_cases']}건")
    print(f"  - Bot 즉시 제재: {result['bot_sanctions']}건")
    print(f"  - 네트워크 제재: {result['network_sanctions']}건")
    print(f"\n결과 저장 위치: {result['output_directory']}")
    print("="*70 + "\n")

