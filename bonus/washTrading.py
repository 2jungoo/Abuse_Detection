"""
Wash Trading Detection System
보너스 현금화(Laundering) 탐지 프로그램

점수 기반 탐지 시스템으로 의심 거래를 식별합니다.
"""

import pandas as pd
import duckdb as dd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# 한글 폰트 설정 (macOS)
import matplotlib
matplotlib.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 하이퍼파라미터 클래스
# ============================================================================

@dataclass
class DetectionConfig:
    """탐지 설정 및 하이퍼파라미터"""
    
    # 시간 관련 파라미터 (초 단위)
    open_time_sync_threshold: int = 120  # 2분: 포지션 오픈 시간 동기화 임계값
    close_time_sync_threshold: int = 120  # 2분: 포지션 클로즈 시간 동기화 임계값
    bonus_time_window: int = 172800  # 48시간: 보너스 후 세탁 거래 시간 창
    
    # 거래량 관련 파라미터
    amount_similarity_tolerance: float = 0.02  # 2%: 거래량 유사도 허용 오차
    
    # 점수 시스템 파라미터
    score_threshold: int = 70  # 의심 거래로 판단하는 최소 점수
    
    # 점수 배점 (총 100점 만점)
    score_weights: Dict[str, int] = field(default_factory=lambda: {
        'time_sync': 30,          # 시간 동기화 점수 (오픈+클로즈)
        'amount_similarity': 25,  # 거래량 유사도 점수
        'pnl_imbalance': 25,      # 손익 불균형 점수
        'bonus_linkage': 20,      # 보너스 연계 점수
    })
    
    # 점수 세부 배점
    time_sync_perfect: int = 30   # 완벽한 시간 동기화 (5초 이내)
    time_sync_good: int = 20      # 좋은 시간 동기화 (30초 이내)
    time_sync_acceptable: int = 10  # 허용 가능한 동기화 (2분 이내)
    
    amount_perfect: int = 25      # 완벽한 수량 일치 (0.5% 이내)
    amount_good: int = 18         # 좋은 수량 일치 (1% 이내)
    amount_acceptable: int = 10   # 허용 가능한 일치 (2% 이내)
    
    pnl_clear_imbalance: int = 25  # 명확한 손익 불균형 (한쪽 손실 > 다른쪽 이익의 90%)
    pnl_moderate_imbalance: int = 15  # 중간 손익 불균형
    
    bonus_immediate: int = 20     # 보너스 후 즉시 거래 (6시간 이내)
    bonus_recent: int = 15        # 보너스 후 최근 거래 (24시간 이내)
    bonus_within_window: int = 10  # 보너스 시간 창 내 거래 (48시간 이내)
    
    # 보고서 생성 설정
    top_n_pairs: int = 10
    top_n_accounts: int = 10
    
    # 출력 파일 경로
    output_dir: str = "output"
    
    def to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            'time_thresholds': {
                'open_sync_sec': self.open_time_sync_threshold,
                'close_sync_sec': self.close_time_sync_threshold,
                'bonus_window_sec': self.bonus_time_window,
            },
            'amount_tolerance': self.amount_similarity_tolerance,
            'score_system': {
                'threshold': self.score_threshold,
                'weights': self.score_weights,
            }
        }
    
    def save_config(self, filepath: str):
        """설정을 JSON 파일로 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)  # 디렉토리 생성
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================================
# 데이터 로더
# ============================================================================

class DataLoader:
    """데이터 로드 및 전처리"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = {}
        self.con = dd.connect()
    
    def load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """Excel 파일에서 모든 시트 로드 및 DuckDB에 등록"""
        print(f"데이터 로드 중: {self.filepath}")
        
        # 각 시트 로드
        sheets = ['Trade', 'Funding', 'Reward', 'IP', 'Spec']
        for sheet_name in sheets:
            df = pd.read_excel(self.filepath, sheet_name=sheet_name)
            
            # 타임스탬프 변환
            if 'ts' in df.columns:
                df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
                null_count = df['ts'].isna().sum()
                if null_count > 0:
                    print(f"  경고: {sheet_name}에서 {null_count}개의 타임스탬프 변환 실패")
            
            self.data[sheet_name] = df
            print(f"  {sheet_name}: {len(df)} 레코드 로드됨")
            # DuckDB에 테이블로 등록
            self.con.register(sheet_name, df)
        
        return self.data
    
    def get_duckdb_connection(self):
        """DuckDB 연결 반환"""
        return self.con
    
    def to_dict_records(self, df: pd.DataFrame) -> List[Dict]:
        """DataFrame을 딕셔너리 리스트로 변환"""
        return df.to_dict('records')


# ============================================================================
# 점수 계산기
# ============================================================================

class ScoreCalculator:
    """점수 기반 탐지를 위한 점수 계산"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def calculate_time_sync_score(self, open_diff: float, close_diff: float) -> Dict[str, any]:
        """시간 동기화 점수 계산"""
        # 오픈/클로즈 각각 점수 계산 후 평균
        open_score = self._get_time_score(open_diff)
        close_score = self._get_time_score(close_diff)
        
        total_score = (open_score + close_score) / 2
        
        return {
            'score': total_score,
            'open_diff_sec': open_diff,
            'close_diff_sec': close_diff,
            'open_score': open_score,
            'close_score': close_score,
        }
    
    def _get_time_score(self, time_diff: float) -> float:
        """시간 차이에 따른 점수 반환"""
        if time_diff <= 5:
            return self.config.time_sync_perfect
        elif time_diff <= 30:
            return self.config.time_sync_good
        elif time_diff <= self.config.open_time_sync_threshold:
            return self.config.time_sync_acceptable
        else:
            return 0
    
    def calculate_amount_similarity_score(self, amount_diff_ratio: float) -> Dict[str, any]:
        """거래량 유사도 점수 계산"""
        if amount_diff_ratio <= 0.005:  # 0.5% 이내
            score = self.config.amount_perfect
        elif amount_diff_ratio <= 0.01:  # 1% 이내
            score = self.config.amount_good
        elif amount_diff_ratio <= self.config.amount_similarity_tolerance:  # 2% 이내
            score = self.config.amount_acceptable
        else:
            score = 0
        
        return {
            'score': score,
            'diff_ratio': amount_diff_ratio,
            'diff_percent': amount_diff_ratio * 100,
        }
    
    def calculate_pnl_imbalance_score(self, loser_pnl: float, winner_pnl: float) -> Dict[str, any]:
        """손익 불균형 점수 계산"""
        # 손실과 이익의 비율 확인
        loss_ratio = abs(loser_pnl) / winner_pnl if winner_pnl > 0 else 0
        
        if loss_ratio >= 0.9:  # 손실이 이익의 90% 이상 (명확한 전가)
            score = self.config.pnl_clear_imbalance
        elif loss_ratio >= 0.7:  # 70% 이상
            score = self.config.pnl_moderate_imbalance
        else:
            score = 0
        
        return {
            'score': score,
            'loser_pnl': loser_pnl,
            'winner_pnl': winner_pnl,
            'loss_ratio': loss_ratio,
        }
    
    def calculate_bonus_linkage_score(self, time_since_bonus_hours: float) -> Dict[str, any]:
        """보너스 연계 점수 계산"""
        if time_since_bonus_hours <= 6:
            score = self.config.bonus_immediate
        elif time_since_bonus_hours <= 24:
            score = self.config.bonus_recent
        elif time_since_bonus_hours <= 48:
            score = self.config.bonus_within_window
        else:
            score = 0
        
        return {
            'score': score,
            'hours_since_bonus': time_since_bonus_hours,
        }
    
    def calculate_total_score(self, 
                            time_sync_data: Dict,
                            amount_data: Dict,
                            pnl_data: Dict,
                            bonus_data: Dict) -> Dict[str, any]:
        """전체 점수 계산 및 의심도 판단"""
        total = (
            time_sync_data['score'] +
            amount_data['score'] +
            pnl_data['score'] +
            bonus_data['score']
        )
        
        is_suspicious = total >= self.config.score_threshold
        
        return {
            'total_score': total,
            'is_suspicious': is_suspicious,
            'breakdown': {
                'time_sync': time_sync_data['score'],
                'amount_similarity': amount_data['score'],
                'pnl_imbalance': pnl_data['score'],
                'bonus_linkage': bonus_data['score'],
            },
            'details': {
                'time_sync': time_sync_data,
                'amount_similarity': amount_data,
                'pnl_imbalance': pnl_data,
                'bonus_linkage': bonus_data,
            }
        }


# ============================================================================
# 보너스 현금화 탐지기
# ============================================================================

class LaunderingDetector:
    """보너스 현금화(Laundering) Wash Trading 탐지"""
    
    def __init__(self, data_loader: DataLoader, config: DetectionConfig):
        self.loader = data_loader
        self.config = config
        self.score_calc = ScoreCalculator(config)
        self.con = data_loader.get_duckdb_connection()
    
    def build_detection_query(self) -> str:
        """탐지 SQL 쿼리 생성"""
        query = f"""
        WITH
        position AS (
            -- 1. 포지션별 PnL 집계
            SELECT 
                account_id,
                position_id,
                MAX(leverage) AS leverage,
                CAST(MIN(ts) AS TIMESTAMP) as open_ts,
                CAST(MAX(ts) AS TIMESTAMP) as closing_ts,
                MAX(symbol) as symbol, 
                MAX(side) as side,
                SUM(IF(openclose='OPEN', amount, 0)) as open_amount,
                SUM(IF(openclose='OPEN', -amount, amount) * IF(side='LONG', 1, -1)) as rpnl
            FROM Trade
            GROUP BY account_id, position_id
            HAVING rpnl != 0
        ),

        bonuses AS (
            -- 2. 보너스 내역 준비
            SELECT 
                account_id, 
                CAST(ts AS TIMESTAMP) as bonus_ts, 
                reward_amount
            FROM Reward
        ),

        LaunderingPairs AS (
            -- 3. 워시 트레이딩 쌍 탐색
            SELECT
                t1.account_id AS loser_account,
                t2.account_id AS winner_account,
                t1.open_ts AS loser_open_ts,
                t1.closing_ts AS loser_close_ts,
                t2.open_ts AS winner_open_ts,
                t2.closing_ts AS winner_close_ts,
                t2.rpnl AS laundered_amount,
                t1.rpnl AS loser_pnl,
                t1.position_id AS loser_pos,
                t2.position_id AS winner_pos,
                t1.symbol,
                t1.side AS loser_side,
                t2.side AS winner_side,
                t1.open_amount AS loser_amount,
                t2.open_amount AS winner_amount,
                
                ABS(epoch(t1.open_ts - t2.open_ts)) AS open_time_diff_sec,
                ABS(epoch(t1.closing_ts - t2.closing_ts)) AS close_time_diff_sec,
                ABS(t1.open_amount - t2.open_amount) / LEAST(t1.open_amount, t2.open_amount) AS amount_diff_ratio
                
            FROM 
                position t1
            JOIN 
                position t2 ON
                    t1.account_id != t2.account_id
                    AND t1.side != t2.side
                    AND t1.symbol = t2.symbol
                    AND ABS(epoch(t1.open_ts - t2.open_ts)) <= {self.config.open_time_sync_threshold}
                    AND ABS(epoch(t1.closing_ts - t2.closing_ts)) <= {self.config.close_time_sync_threshold}
                    AND t1.open_amount <= {1 + self.config.amount_similarity_tolerance} * t2.open_amount 
                    AND t1.open_amount >= {1 - self.config.amount_similarity_tolerance} * t2.open_amount
                    AND t1.rpnl < 0 AND t2.rpnl > 0
        ),

        LinkedTrades AS (
            -- 4. 보너스 연결
            SELECT
                lp.*,
                b.bonus_ts,
                b.reward_amount,
                epoch(lp.loser_open_ts - b.bonus_ts) AS time_since_bonus_sec,
                epoch(lp.loser_open_ts - b.bonus_ts) / 3600.0 AS time_since_bonus_hours
            FROM LaunderingPairs lp
            JOIN bonuses b ON lp.loser_account = b.account_id
            WHERE
                b.bonus_ts <= lp.loser_open_ts
                AND epoch(lp.loser_open_ts - b.bonus_ts) <= {self.config.bonus_time_window}
        )

        SELECT * FROM LinkedTrades
        ORDER BY laundered_amount DESC, time_since_bonus_hours ASC;
        """
        return query
    
    def detect(self) -> List[Dict]:
        """탐지 실행 및 점수 계산"""
        print("="*70)
        print("보너스 현금화(Laundering) 탐지 시작")
        print("="*70)
        
        # SQL 쿼리 실행
        query = self.build_detection_query()
        df = self.con.execute(query).fetchdf()
        
        print(f"\n초기 탐지: {len(df)} 건의 후보 거래")
        
        if len(df) == 0:
            print("탐지된 거래가 없습니다.")
            return []
        
        # 각 거래에 점수 계산
        results = []
        for idx, row in df.iterrows():
            # 시간 동기화 점수
            time_score = self.score_calc.calculate_time_sync_score(
                row['open_time_diff_sec'],
                row['close_time_diff_sec']
            )
            
            # 거래량 유사도 점수
            amount_score = self.score_calc.calculate_amount_similarity_score(
                row['amount_diff_ratio']
            )
            
            # 손익 불균형 점수
            pnl_score = self.score_calc.calculate_pnl_imbalance_score(
                row['loser_pnl'],
                row['laundered_amount']
            )
            
            # 보너스 연계 점수
            bonus_score = self.score_calc.calculate_bonus_linkage_score(
                row['time_since_bonus_hours']
            )
            
            # 총 점수 계산
            total_score_data = self.score_calc.calculate_total_score(
                time_score, amount_score, pnl_score, bonus_score
            )
            
            # 결과 딕셔너리 생성
            result = {
                # 기본 정보
                'loser_account': row['loser_account'],
                'winner_account': row['winner_account'],
                'symbol': row['symbol'],
                'loser_pos': row['loser_pos'],
                'winner_pos': row['winner_pos'],
                
                # 금액 정보
                'laundered_amount': float(row['laundered_amount']),
                'loser_pnl': float(row['loser_pnl']),
                'linked_bonus': float(row['reward_amount']),
                
                # 시간 정보
                'bonus_ts': row['bonus_ts'].isoformat() if pd.notna(row['bonus_ts']) else None,
                'loser_open_ts': row['loser_open_ts'].isoformat() if pd.notna(row['loser_open_ts']) else None,
                'loser_close_ts': row['loser_close_ts'].isoformat() if pd.notna(row['loser_close_ts']) else None,
                'winner_open_ts': row['winner_open_ts'].isoformat() if pd.notna(row['winner_open_ts']) else None,
                'winner_close_ts': row['winner_close_ts'].isoformat() if pd.notna(row['winner_close_ts']) else None,
                'time_since_bonus_hours': float(row['time_since_bonus_hours']),
                
                # 거래 세부사항
                'loser_side': row['loser_side'],
                'winner_side': row['winner_side'],
                'loser_amount': float(row['loser_amount']),
                'winner_amount': float(row['winner_amount']),
                'amount_diff_ratio': float(row['amount_diff_ratio']),
                
                # 점수 정보
                'total_score': total_score_data['total_score'],
                'is_suspicious': total_score_data['is_suspicious'],
                'score_breakdown': total_score_data['breakdown'],
                'score_details': total_score_data['details'],
            }
            
            results.append(result)
        
        # 의심 거래만 필터링
        suspicious_trades = [r for r in results if r['is_suspicious']]
        
        print(f"점수 기반 필터링: {len(suspicious_trades)} 건의 의심 거래 (점수 >= {self.config.score_threshold})")
        
        if len(suspicious_trades) > 0:
            avg_score = sum(r['total_score'] for r in suspicious_trades) / len(suspicious_trades)
            max_score = max(r['total_score'] for r in suspicious_trades)
            min_score = min(r['total_score'] for r in suspicious_trades)
            
            print(f"\n점수 분포:")
            print(f"  - 평균 점수: {avg_score:.1f}")
            print(f"  - 최고 점수: {max_score:.1f}")
            print(f"  - 최저 점수: {min_score:.1f}")
        
        print("="*70)
        
        return suspicious_trades


# ============================================================================
# 분석기
# ============================================================================

class LaunderingAnalyzer:
    """탐지 결과 분석"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """전체 결과 분석"""
        if not results:
            return {
                'summary': {},
                'pair_analysis': [],
                'account_analysis': {},
                'symbol_analysis': [],
                'time_analysis': [],
            }
        
        # 기본 통계
        summary = self._calculate_summary(results)
        
        # 계정 쌍별 분석
        pair_analysis = self._analyze_pairs(results)
        
        # 계정별 분석
        account_analysis = self._analyze_accounts(results)
        
        # 심볼별 분석
        symbol_analysis = self._analyze_symbols(results)
        
        # 시간대별 분석
        time_analysis = self._analyze_time_patterns(results)
        
        return {
            'summary': summary,
            'pair_analysis': pair_analysis,
            'account_analysis': account_analysis,
            'symbol_analysis': symbol_analysis,
            'time_analysis': time_analysis,
        }
    
    def _calculate_summary(self, results: List[Dict]) -> Dict:
        """요약 통계 계산"""
        total_laundered = sum(r['laundered_amount'] for r in results)
        total_loss = sum(r['loser_pnl'] for r in results)
        total_bonus = sum(r['linked_bonus'] for r in results)
        
        losers = set(r['loser_account'] for r in results)
        winners = set(r['winner_account'] for r in results)
        
        avg_time_since_bonus = sum(r['time_since_bonus_hours'] for r in results) / len(results)
        avg_score = sum(r['total_score'] for r in results) / len(results)
        
        return {
            'trade_count': len(results),
            'total_laundered_amount': total_laundered,
            'total_intentional_loss': total_loss,
            'total_linked_bonus': total_bonus,
            'laundering_efficiency': (total_laundered / total_bonus * 100) if total_bonus > 0 else 0,
            'unique_loser_accounts': len(losers),
            'unique_winner_accounts': len(winners),
            'avg_time_since_bonus_hours': avg_time_since_bonus,
            'avg_suspicion_score': avg_score,
        }
    
    def _analyze_pairs(self, results: List[Dict]) -> List[Dict]:
        """계정 쌍별 분석"""
        pairs = defaultdict(lambda: {
            'trade_count': 0,
            'total_laundered': 0,
            'total_loss': 0,
            'total_bonus': 0,
            'avg_score': 0,
            'symbols': set(),
        })
        
        for r in results:
            key = (r['loser_account'], r['winner_account'])
            pairs[key]['trade_count'] += 1
            pairs[key]['total_laundered'] += r['laundered_amount']
            pairs[key]['total_loss'] += r['loser_pnl']
            pairs[key]['total_bonus'] += r['linked_bonus']
            pairs[key]['avg_score'] += r['total_score']
            pairs[key]['symbols'].add(r['symbol'])
        
        # 평균 점수 계산 및 정렬
        pair_list = []
        for (loser, winner), data in pairs.items():
            pair_list.append({
                'loser_account': loser,
                'winner_account': winner,
                'trade_count': data['trade_count'],
                'total_laundered_profit': data['total_laundered'],
                'total_intentional_loss': data['total_loss'],
                'total_linked_bonus': data['total_bonus'],
                'avg_suspicion_score': data['avg_score'] / data['trade_count'],
                'symbols_traded': ', '.join(sorted(data['symbols'])),
            })
        
        # 세탁 금액 기준 정렬
        pair_list.sort(key=lambda x: x['total_laundered_profit'], reverse=True)
        
        return pair_list
    
    def _analyze_accounts(self, results: List[Dict]) -> Dict:
        """계정별 분석 (손실/수익 분리)"""
        losers = defaultdict(lambda: {
            'trade_count': 0,
            'total_loss': 0,
            'total_bonus_used': 0,
            'avg_score': 0,
            'unique_winners': set(),
        })
        
        winners = defaultdict(lambda: {
            'trade_count': 0,
            'total_profit': 0,
            'avg_score': 0,
            'unique_losers': set(),
        })
        
        for r in results:
            # 손실 계정
            losers[r['loser_account']]['trade_count'] += 1
            losers[r['loser_account']]['total_loss'] += r['loser_pnl']
            losers[r['loser_account']]['total_bonus_used'] += r['linked_bonus']
            losers[r['loser_account']]['avg_score'] += r['total_score']
            losers[r['loser_account']]['unique_winners'].add(r['winner_account'])
            
            # 수익 계정
            winners[r['winner_account']]['trade_count'] += 1
            winners[r['winner_account']]['total_profit'] += r['laundered_amount']
            winners[r['winner_account']]['avg_score'] += r['total_score']
            winners[r['winner_account']]['unique_losers'].add(r['loser_account'])
        
        # 리스트로 변환
        loser_list = []
        for acc, data in losers.items():
            loser_list.append({
                'account_id': acc,
                'laundering_count': data['trade_count'],
                'total_loss': data['total_loss'],
                'total_bonus_used': data['total_bonus_used'],
                'avg_suspicion_score': data['avg_score'] / data['trade_count'],
                'unique_winner_count': len(data['unique_winners']),
            })
        loser_list.sort(key=lambda x: x['total_loss'])
        
        winner_list = []
        for acc, data in winners.items():
            winner_list.append({
                'account_id': acc,
                'laundering_count': data['trade_count'],
                'total_profit': data['total_profit'],
                'avg_suspicion_score': data['avg_score'] / data['trade_count'],
                'unique_loser_count': len(data['unique_losers']),
            })
        winner_list.sort(key=lambda x: x['total_profit'], reverse=True)
        
        return {
            'losers': loser_list,
            'winners': winner_list,
        }
    
    def _analyze_symbols(self, results: List[Dict]) -> List[Dict]:
        """심볼별 분석"""
        symbols = defaultdict(lambda: {
            'trade_count': 0,
            'total_laundered': 0,
            'avg_score': 0,
            'pairs': set(),
        })
        
        for r in results:
            symbols[r['symbol']]['trade_count'] += 1
            symbols[r['symbol']]['total_laundered'] += r['laundered_amount']
            symbols[r['symbol']]['avg_score'] += r['total_score']
            symbols[r['symbol']]['pairs'].add((r['loser_account'], r['winner_account']))
        
        symbol_list = []
        for sym, data in symbols.items():
            symbol_list.append({
                'symbol': sym,
                'trade_count': data['trade_count'],
                'total_laundered': data['total_laundered'],
                'avg_laundered': data['total_laundered'] / data['trade_count'],
                'avg_suspicion_score': data['avg_score'] / data['trade_count'],
                'unique_pairs': len(data['pairs']),
            })
        
        symbol_list.sort(key=lambda x: x['total_laundered'], reverse=True)
        
        return symbol_list
    
    def _analyze_time_patterns(self, results: List[Dict]) -> List[Dict]:
        """시간대별 패턴 분석"""
        time_brackets = {
            '0-6h': [],
            '6-12h': [],
            '12-24h': [],
            '24-48h': [],
        }
        
        for r in results:
            hours = r['time_since_bonus_hours']
            if hours <= 6:
                bracket = '0-6h'
            elif hours <= 12:
                bracket = '6-12h'
            elif hours <= 24:
                bracket = '12-24h'
            else:
                bracket = '24-48h'
            
            time_brackets[bracket].append(r)
        
        time_analysis = []
        for bracket, trades in time_brackets.items():
            if trades:
                time_analysis.append({
                    'time_bracket': bracket,
                    'trade_count': len(trades),
                    'total_laundered': sum(t['laundered_amount'] for t in trades),
                    'avg_laundered': sum(t['laundered_amount'] for t in trades) / len(trades),
                    'avg_suspicion_score': sum(t['total_score'] for t in trades) / len(trades),
                })
        
        return time_analysis


# ============================================================================
# 보고서 생성기
# ============================================================================

class ReportGenerator:
    """분석 결과 보고서 생성"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_reports(self, results: List[Dict], analysis: Dict):
        """모든 보고서 생성"""
        if not results:
            print("결과가 없어 보고서를 생성하지 않습니다.")
            return
        
        print("\n" + "="*70)
        print("보고서 생성 중...")
        print("="*70)
        
        # JSON 보고서
        self._save_json_report(results, analysis)
        
        # CSV 보고서
        self._save_csv_reports(results, analysis)
        
        # Excel 보고서
        self._save_excel_report(results, analysis)
        
        # 텍스트 요약 보고서
        self._print_summary_report(analysis)
        
        print("="*70)
    
    def _save_json_report(self, results: List[Dict], analysis: Dict):
        """JSON 형식으로 전체 결과 저장"""
        report = {
            'detection_config': self.config.to_dict(),
            'results': results,
            'analysis': {
                'summary': analysis['summary'],
                'top_pairs': analysis['pair_analysis'][:self.config.top_n_pairs],
                'top_losers': analysis['account_analysis']['losers'][:self.config.top_n_accounts],
                'top_winners': analysis['account_analysis']['winners'][:self.config.top_n_accounts],
                'symbol_analysis': analysis['symbol_analysis'],
                'time_analysis': analysis['time_analysis'],
            }
        }
        
        filepath = self.output_dir / 'laundering_detection_report.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  JSON 보고서 저장: {filepath}")
    
    def _save_csv_reports(self, results: List[Dict], analysis: Dict):
        """CSV 형식으로 분석 결과 저장"""
        # 전체 거래 내역
        df_trades = pd.DataFrame(results)
        df_trades.to_csv(self.output_dir / 'laundering_all_trades.csv', index=False)
        print(f"  CSV 저장: laundering_all_trades.csv")
        
        # 계정 쌍별 집계
        df_pairs = pd.DataFrame(analysis['pair_analysis'])
        df_pairs.to_csv(self.output_dir / 'laundering_pairs.csv', index=False)
        print(f"  CSV 저장: laundering_pairs.csv")
        
        # 손실 계정
        df_losers = pd.DataFrame(analysis['account_analysis']['losers'])
        df_losers.to_csv(self.output_dir / 'laundering_loser_accounts.csv', index=False)
        print(f"  CSV 저장: laundering_loser_accounts.csv")
        
        # 수익 계정
        df_winners = pd.DataFrame(analysis['account_analysis']['winners'])
        df_winners.to_csv(self.output_dir / 'laundering_winner_accounts.csv', index=False)
        print(f"  CSV 저장: laundering_winner_accounts.csv")
    
    def _save_excel_report(self, results: List[Dict], analysis: Dict):
        """Excel 형식으로 종합 보고서 저장"""
        filepath = self.output_dir / 'laundering_detailed_report.xlsx'
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약
            summary_df = pd.DataFrame([analysis['summary']])
            summary_df.to_excel(writer, sheet_name='요약', index=False)
            
            # 전체 거래
            trades_df = pd.DataFrame(results)
            trades_df.to_excel(writer, sheet_name='전체_거래', index=False)
            
            # 계정 쌍
            pairs_df = pd.DataFrame(analysis['pair_analysis'])
            pairs_df.to_excel(writer, sheet_name='계정_쌍', index=False)
            
            # 손실 계정
            losers_df = pd.DataFrame(analysis['account_analysis']['losers'])
            losers_df.to_excel(writer, sheet_name='손실_계정', index=False)
            
            # 수익 계정
            winners_df = pd.DataFrame(analysis['account_analysis']['winners'])
            winners_df.to_excel(writer, sheet_name='수익_계정', index=False)
            
            # 심볼별
            symbols_df = pd.DataFrame(analysis['symbol_analysis'])
            symbols_df.to_excel(writer, sheet_name='심볼별', index=False)
            
            # 시간대별
            time_df = pd.DataFrame(analysis['time_analysis'])
            time_df.to_excel(writer, sheet_name='시간대별', index=False)
        
        print(f"  Excel 보고서 저장: {filepath}")
    
    def _print_summary_report(self, analysis: Dict):
        """콘솔에 요약 보고서 출력"""
        summary = analysis['summary']
        
        print("\n" + "="*70)
        print(" 보너스 현금화(Laundering) 탐지 최종 요약")
        print("="*70)
        
        if not summary:
            print("  분석할 데이터가 없습니다.")
            return
        
        print(f"\n 탐지 규모:")
        print(f"  - 총 의심 거래: {summary['trade_count']:,} 건")
        print(f"  - 손실 계정 수: {summary['unique_loser_accounts']} 개")
        print(f"  - 수익 계정 수: {summary['unique_winner_accounts']} 개")
        print(f"  - 평균 의심 점수: {summary['avg_suspicion_score']:.1f} / 100")
        
        print(f"\n 금액 규모:")
        print(f"  - 총 세탁 금액: ${summary['total_laundered_amount']:,.2f}")
        print(f"  - 총 의도적 손실: ${summary['total_intentional_loss']:,.2f}")
        print(f"  - 연결된 보너스: ${summary['total_linked_bonus']:,.2f}")
        print(f"  - 세탁 효율: {summary['laundering_efficiency']:.1f}%")
        
        print(f"\n 시간 패턴:")
        print(f"  - 평균 보너스 후 거래 시간: {summary['avg_time_since_bonus_hours']:.1f} 시간")
        
        # Top 계정 쌍
        if analysis['pair_analysis']:
            top_pair = analysis['pair_analysis'][0]
            print(f"\n 최대 세탁 계정 쌍:")
            print(f"  - {top_pair['loser_account']} → {top_pair['winner_account']}")
            print(f"  - 총 세탁 금액: ${top_pair['total_laundered_profit']:,.2f}")
            print(f"  - 거래 횟수: {top_pair['trade_count']} 건")
            print(f"  - 평균 의심 점수: {top_pair['avg_suspicion_score']:.1f}")
        
        print("\n" + "="*70)


# ============================================================================
# 시각화 모듈
# ============================================================================

class Visualizer:
    """탐지 결과 시각화"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_visualizations(self, results: List[Dict], analysis: Dict):
        """모든 시각화 생성"""
        if not results:
            print("시각화할 데이터가 없습니다.")
            return
        
        print("\n" + "="*70)
        print("시각화 생성 중...")
        print("="*70)
        
        # 대시보드
        self._create_dashboard(results, analysis)
        
        # 패턴 분석
        self._create_pattern_analysis(analysis)
        
        print("="*70)
    
    def _create_dashboard(self, results: List[Dict], analysis: Dict):
        """종합 대시보드 생성"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('보너스 현금화(Laundering) 탐지 대시보드', 
                     fontsize=18, fontweight='bold', y=0.995)
        
        # 1. 세탁 금액 분포
        ax1 = axes[0, 0]
        amounts = [r['laundered_amount'] for r in results]
        sns.histplot(amounts, kde=True, bins=30, color='crimson', alpha=0.7, ax=ax1)
        mean_amount = sum(amounts) / len(amounts)
        ax1.axvline(mean_amount, color='darkred', linestyle='--', linewidth=2,
                    label=f'평균: ${mean_amount:,.2f}')
        ax1.set_title('세탁 금액 분포', fontsize=14, fontweight='bold')
        ax1.set_xlabel('세탁 금액 (USD)')
        ax1.set_ylabel('거래 횟수')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 의심 점수 분포
        ax2 = axes[0, 1]
        scores = [r['total_score'] for r in results]
        sns.histplot(scores, kde=True, bins=20, color='orange', alpha=0.7, ax=ax2)
        ax2.axvline(self.config.score_threshold, color='red', linestyle='--', 
                    linewidth=2, label=f'임계값: {self.config.score_threshold}')
        ax2.set_title('의심 점수 분포', fontsize=14, fontweight='bold')
        ax2.set_xlabel('의심 점수')
        ax2.set_ylabel('거래 횟수')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 보너스 vs 세탁 금액
        ax3 = axes[1, 0]
        bonuses = [r['linked_bonus'] for r in results]
        laundered = [r['laundered_amount'] for r in results]
        time_colors = [r['time_since_bonus_hours'] for r in results]
        
        scatter = ax3.scatter(bonuses, laundered, c=time_colors, cmap='viridis',
                             alpha=0.6, s=80, edgecolors='black', linewidth=0.5)
        max_val = max(max(bonuses), max(laundered))
        ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='1:1 라인')
        ax3.set_title('보너스 금액 vs 세탁 금액', fontsize=14, fontweight='bold')
        ax3.set_xlabel('보너스 금액 (USD)')
        ax3.set_ylabel('세탁 금액 (USD)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('보너스 후 시간(h)', fontsize=10)
        
        # 4. Top 계정 쌍
        ax4 = axes[1, 1]
        top_pairs = analysis['pair_analysis'][:15]
        if top_pairs:
            labels = [f"{p['loser_account']} → {p['winner_account']}" for p in top_pairs]
            values = [p['total_laundered_profit'] for p in top_pairs]
            scores = [p['avg_suspicion_score'] for p in top_pairs]
            
            bars = ax4.barh(range(len(top_pairs)), values, color='steelblue', alpha=0.8)
            
            # 점수에 따라 색상 조절
            colors = plt.cm.Reds(np.array(scores) / 100)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax4.set_yticks(range(len(top_pairs)))
            ax4.set_yticklabels(labels, fontsize=9)
            ax4.set_title('계정 쌍별 총 세탁 금액 Top 15', fontsize=14, fontweight='bold')
            ax4.set_xlabel('총 세탁 금액 (USD)')
            ax4.grid(True, axis='x', alpha=0.3)
            ax4.invert_yaxis()
            
            # 거래 횟수 표시
            for idx, pair in enumerate(top_pairs):
                ax4.text(pair['total_laundered_profit'] * 0.02, idx,
                        f"{pair['trade_count']}건",
                        va='center', fontsize=8, color='white', fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'laundering_dashboard.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  대시보드 저장: {filepath}")
    
    def _create_pattern_analysis(self, analysis: Dict):
        """패턴 분석 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('보너스 현금화 패턴 분석', fontsize=16, fontweight='bold')
        
        # 1. 심볼별 세탁 금액
        ax1 = axes[0]
        symbols = analysis['symbol_analysis']
        if symbols:
            sym_labels = [s['symbol'] for s in symbols]
            sym_values = [s['total_laundered'] for s in symbols]
            sym_counts = [s['trade_count'] for s in symbols]
            
            bars = ax1.bar(range(len(symbols)), sym_values, color='steelblue', alpha=0.8)
            
            # 거래 횟수에 따라 색상 조절
            max_count = max(sym_counts)
            colors = plt.cm.Oranges(np.array(sym_counts) / max_count)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax1.set_xticks(range(len(symbols)))
            ax1.set_xticklabels(sym_labels, rotation=45, ha='right')
            ax1.set_title('심볼별 총 세탁 금액', fontsize=14, fontweight='bold')
            ax1.set_ylabel('총 세탁 금액 (USD)')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # 거래 횟수 표시
            for idx, sym in enumerate(symbols):
                ax1.text(idx, sym['total_laundered'] * 1.02,
                        f"{sym['trade_count']}건",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. 시간대별 패턴
        ax2 = axes[1]
        time_data = analysis['time_analysis']
        if time_data:
            time_labels = [t['time_bracket'] for t in time_data]
            time_values = [t['total_laundered'] for t in time_data]
            time_counts = [t['trade_count'] for t in time_data]
            
            colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90'][:len(time_data)]
            bars = ax2.bar(time_labels, time_values, color=colors, alpha=0.8,
                          edgecolor='black', linewidth=1.5)
            
            ax2.set_title('보너스 수령 후 시간대별 세탁 금액', fontsize=14, fontweight='bold')
            ax2.set_xlabel('보너스 수령 후 경과 시간')
            ax2.set_ylabel('총 세탁 금액 (USD)')
            ax2.grid(True, axis='y', alpha=0.3)
            
            # 거래 횟수 표시
            for idx, time in enumerate(time_data):
                ax2.text(idx, time['total_laundered'] * 1.02,
                        f"{time['trade_count']}건\n평균 ${time['avg_laundered']:,.0f}",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'laundering_patterns.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  패턴 분석 저장: {filepath}")


# ============================================================================
# 메인 실행 함수
# ============================================================================

def run_laundering_detection(
    data_filepath: str,
    config: Optional[DetectionConfig] = None,
    enable_visualization: bool = True
) -> Dict:
    """
    보너스 현금화 탐지 메인 함수
    
    Args:
        data_filepath: 데이터 파일 경로 (Excel)
        config: 탐지 설정 (None일 경우 기본값 사용)
        enable_visualization: 시각화 생성 여부
    
    Returns:
        탐지 결과 딕셔너리
    """
    # 설정 초기화
    if config is None:
        config = DetectionConfig()
    
    # 설정 저장
    config.save_config(str(Path(config.output_dir) / 'detection_config.json'))
    
    # 데이터 로드
    loader = DataLoader(data_filepath)
    loader.load_excel_data()
    
    # 탐지 실행
    detector = LaunderingDetector(loader, config)
    results = detector.detect()
    
    # 분석
    analyzer = LaunderingAnalyzer(config)
    analysis = analyzer.analyze_results(results)
    
    # 보고서 생성
    reporter = ReportGenerator(config)
    reporter.generate_reports(results, analysis)
    
    # 시각화 (옵션)
    if enable_visualization and results:
        visualizer = Visualizer(config)
        visualizer.create_visualizations(results, analysis)
    
    return {
        'config': config.to_dict(),
        'results': results,
        'analysis': analysis,
    }


# ============================================================================
# 실행 예제
# ============================================================================

if __name__ == "__main__":
    # 커스텀 설정 예제
    custom_config = DetectionConfig(
        # 시간 임계값 조정
        open_time_sync_threshold=180,  # 3분으로 완화
        close_time_sync_threshold=180,
        bonus_time_window=86400,  # 24시간으로 단축
        
        # 점수 임계값 조정
        score_threshold=75,  # 더 엄격하게
        
        # 거래량 허용 오차 조정
        amount_similarity_tolerance=0.03,  # 3%로 완화
        
        # Top N 개수
        top_n_pairs=15,
        top_n_accounts=15,
        
        # 출력 디렉토리
        output_dir="./bonus/output"
    )
    
    # 실행
    result = run_laundering_detection(
        data_filepath="problem_data_final.xlsx",
        config=custom_config,
        enable_visualization=True
    )
    
    print("\n" + "="*70)
    print("탐지 완료!")
    print(f"총 {len(result['results'])} 건의 의심 거래가 탐지되었습니다.")
    print(f"결과는 '{custom_config.output_dir}' 디렉토리에 저장되었습니다.")
    print("="*70)
