# -*- coding: utf-8 -*-
"""
암호화폐 거래소 부정거래 탐지 시스템
Fraud Detection System for Cryptocurrency Exchange

탐지 대상:
1. Funding Hunter - 펀딩비 악용
2. Wash Trading - 자전거래
3. Cooperative Trading - 공모거래

Author: Fraud Detection Team
Date: 2025-11-13
"""

import pandas as pd
import duckdb as dd
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 설정 및 초기화
# ============================================================================

DATA_FILE = "problem_data_final.xlsx"
OUTPUT_FILE = "fraud_detection_results.xlsx"

print("=" * 80)
print(" " * 20 + "부정거래 탐지 시스템 시작")
print("=" * 80)
print()

# ============================================================================
# 데이터 로딩
# ============================================================================

print("[1/6] 데이터 로딩 중...")
try:
    Trade = pd.read_excel(DATA_FILE, sheet_name="Trade")
    Funding = pd.read_excel(DATA_FILE, sheet_name="Funding")
    Reward = pd.read_excel(DATA_FILE, sheet_name="Reward")
    Ip = pd.read_excel(DATA_FILE, sheet_name="IP")
    Spec = pd.read_excel(DATA_FILE, sheet_name="Spec")

    print(f"  ✓ Trade: {Trade.shape[0]:,} rows")
    print(f"  ✓ Funding: {Funding.shape[0]:,} rows")
    print(f"  ✓ Reward: {Reward.shape[0]:,} rows")
    print(f"  ✓ IP: {Ip.shape[0]:,} rows")
    print(f"  ✓ Spec: {Spec.shape[0]:,} rows")
    print("  ✓ 데이터 로딩 완료\n")
except Exception as e:
    print(f"  ✗ 데이터 로딩 실패: {e}")
    exit(1)

# ============================================================================
# Part 1: Funding Hunter 탐지
# ============================================================================

print("[2/6] Funding Hunter 탐지 중...")
print("-" * 80)

funding_query = """
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
        CAST(min(ts) AS TIMESTAMP) as open_ts,
        CAST(max(ts) AS TIMESTAMP) as closing_ts,
        max(symbol) as symbol,
        max(side) as side,
        DATE(max(ts)) as closing_day,
        sum(if(openclose='OPEN',amount,0)) as amount
    from Trade
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
        (julian(ct.closing_ts) - julian(ct.open_ts)) * 24 * 60 AS holding_minutes,
        sc.fund_period_hr,
        sc.max_order_amount
    FROM position ct
    LEFT JOIN funding_agg fa ON ct.account_id = fa.account_id
    LEFT JOIN spec_clean sc
        ON ct.symbol = sc.symbol AND ct.closing_day = sc.spec_day
),
final AS (
    SELECT *,
        CASE
            WHEN
                total_funding > 0 AND
                leverage >= 5 AND
                amount > max_order_amount * 0.3 AND
                holding_minutes < 20 AND
                CAST(STRFTIME('%H', closing_ts) AS INTEGER) % fund_period_hr = 0 AND
                EXTRACT(HOUR FROM closing_ts) != EXTRACT(HOUR FROM open_ts)
            THEN 1 ELSE 0
        END AS b_funding_hunter
    FROM joined
)
SELECT DISTINCT *
FROM final
WHERE b_funding_hunter = 1
ORDER BY symbol, open_ts;
"""

try:
    fund_df = dd.query(funding_query).to_df()
    funding_hunters = list(set(fund_df[fund_df.b_funding_hunter == 1].account_id))

    print(f"탐지 결과:")
    print(f"  • 탐지된 Funding Hunter: {len(funding_hunters)}명")

    if len(funding_hunters) > 0:
        print(f"\n  탐지된 계정:")
        for i, acc in enumerate(funding_hunters, 1):
            print(f"    {i}. {acc}")

        # 수익 계산
        funding_profit = Funding[Funding.account_id.isin(funding_hunters)].groupby('account_id').sum()[
                             'funding_fee'] * -1
        total_profit = funding_profit.sum()

        print(f"\n  수익 분석:")
        print(f"    총 펀딩비 수익: ${total_profit:,.2f}")
        print(f"    평균 수익: ${total_profit / len(funding_hunters):,.2f}")
        print(f"\n  계정별 상세:")
        for acc in funding_hunters:
            if acc in funding_profit.index:
                print(f"    {acc}: ${funding_profit[acc]:,.2f}")
    else:
        print("  ⚠️  탐지된 계정 없음")

    print()
except Exception as e:
    print(f"  ✗ Funding Hunter 탐지 실패: {e}\n")
    funding_hunters = []
    fund_df = pd.DataFrame()

# ============================================================================
# Part 2: Wash Trading 탐지
# ============================================================================

print("[3/6] Wash Trading 탐지 중...")
print("-" * 80)

wash_query = """
WITH
position AS (
    SELECT
        account_id,
        position_id,
        MAX(leverage) AS leverage,
        CAST(min(ts) AS TIMESTAMP) as open_ts,
        CAST(max(ts) AS TIMESTAMP) as closing_ts,
        max(symbol) as symbol,
        max(side) as side,
        DATE(max(ts)) as closing_day,
        sum(if(openclose='OPEN',amount,0)) as amount,
        sum(if(openclose='OPEN',-amount,amount)*if(side='LONG',1,-1)) as rpnl
    from Trade
    GROUP BY account_id, position_id
),
joined AS (
    select
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
        t2.rpnl as rpnl2
    from
    position t1 inner join position t2
        on
            t1.symbol = t2.symbol
            and ABS(julian(t1.open_ts) - julian(t2.open_ts)) * 24 * 60 <= 2
            and ABS(julian(t1.closing_ts) - julian(t2.closing_ts)) * 24 * 60 <= 2
            and t1.leverage = t2.leverage
            and t1.open_ts < t2.open_ts
            and t1.amount <= 1.02 * t2.amount and t1.amount >= 0.98 * t2.amount
            and GREATEST(t1.open_ts, t2.open_ts) < LEAST(t1.closing_ts, t2.closing_ts)
            and t1.account_id != t2.account_id
            and t1.side != t2.side
            and (t1.rpnl > 0 or t2.rpnl > 0)
)
SELECT DISTINCT *
FROM joined
ORDER BY symbol, open_ts1;
"""

try:
    wash_df = dd.query(wash_query).to_df()

    if len(wash_df) > 0:
        unique_pairs = set(tuple(sorted([a1, a2])) for a1, a2 in zip(wash_df.account_id1, wash_df.account_id2))

        print(f"탐지 결과:")
        print(f"  • 의심 거래 패턴: {len(wash_df)}개")
        print(f"  • 고유 계정 페어: {len(unique_pairs)}개")

        # 1. 각 계정별 rpnl > 0 / < 0 합 계산
        rpnl_long = pd.concat([
            wash_df[['account_id1', 'rpnl1']].rename(columns={'account_id1': 'account_id', 'rpnl1': 'rpnl'}),
            wash_df[['account_id2', 'rpnl2']].rename(columns={'account_id2': 'account_id', 'rpnl2': 'rpnl'})
        ])

        rpnl_stats = (
            rpnl_long.groupby('account_id')['rpnl']
            .agg([
                ('rpnl_pos_sum', lambda x: x[x > 0].sum()),
                ('rpnl_neg_sum', lambda x: x[x < 0].sum())
            ])
            .fillna(0)
        )

        # 2. reward 합산
        reward_sum = Reward.groupby('account_id')['reward_amount'].sum().rename('reward_sum')

        # 3. 병합 후 계정별 net_pnl 계산
        account_stats = (
            rpnl_stats.join(reward_sum, how='left').fillna({'reward_sum': 0})
        )
        account_stats['net_pnl'] = account_stats['rpnl_pos_sum'] + (
                    account_stats['rpnl_neg_sum'] + account_stats['reward_sum']).clip(upper=0)


        # 4. pair별 합산
        def sorted_pair(a, b):
            return tuple(sorted([a, b]))


        wash_df['pair'] = wash_df.apply(lambda row: sorted_pair(row['account_id1'], row['account_id2']), axis=1)

        pair_net_pnl = (
            pd.DataFrame(wash_df['pair'].unique(), columns=['pair'])
            .assign(
                net_pnl1=lambda df: df['pair'].apply(
                    lambda p: account_stats.loc[p[0], 'net_pnl'] if p[0] in account_stats.index else 0),
                net_pnl2=lambda df: df['pair'].apply(
                    lambda p: account_stats.loc[p[1], 'net_pnl'] if p[1] in account_stats.index else 0)
            )
        )
        pair_net_pnl['pair_net_pnl'] = pair_net_pnl['net_pnl1'] + pair_net_pnl['net_pnl2']
        pair_net_pnl = pair_net_pnl.sort_values('pair_net_pnl', ascending=False).reset_index(drop=True)

        total_wash_profit = pair_net_pnl['pair_net_pnl'].sum()

        print(f"\n  수익 분석:")
        print(f"    총 순수익: ${total_wash_profit:,.2f}")
        print(f"    평균 페어 수익: ${total_wash_profit / len(pair_net_pnl):,.2f}")

        print(f"\n  탐지된 Wash Trading 페어 (상위 5개):")
        for idx, row in pair_net_pnl.head(5).iterrows():
            print(f"    {idx + 1}. {row['pair'][0]} ↔ {row['pair'][1]}: ${row['pair_net_pnl']:,.2f}")
    else:
        print("  ⚠️  탐지된 패턴 없음")
        unique_pairs = set()
        pair_net_pnl = pd.DataFrame()

    print()
except Exception as e:
    print(f"  ✗ Wash Trading 탐지 실패: {e}\n")
    wash_df = pd.DataFrame()
    pair_net_pnl = pd.DataFrame()
    unique_pairs = set()

# ============================================================================
# Part 3: Cooperative Trading 탐지
# ============================================================================

print("[4/6] Cooperative Trading 탐지 중...")
print("-" * 80)

cop_query = """
WITH
position AS (
    SELECT
        account_id,
        position_id,
        MAX(leverage) AS leverage,
        CAST(min(ts) AS TIMESTAMP) as open_ts,
        CAST(max(ts) AS TIMESTAMP) as closing_ts,
        max(symbol) as symbol,
        max(side) as side,
        DATE(max(ts)) as closing_day,
        sum(if(openclose='OPEN',amount,0)) as amount,
        sum(if(openclose='OPEN',-amount,amount)*if(side='LONG',1,-1)) as rpnl
    from Trade
    GROUP BY account_id, position_id
),
joined AS (
    select
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
        t2.rpnl as rpnl2
    from
    position t1 inner join position t2
        on
            t1.symbol = t2.symbol
            and ABS(julian(t1.open_ts) - julian(t2.open_ts)) * 24 * 60 <= 2
            and ABS(julian(t1.closing_ts) - julian(t2.closing_ts)) * 24 * 60 <= 2
            and t1.open_ts < t2.open_ts
            and GREATEST(t1.open_ts, t2.open_ts) < LEAST(t1.closing_ts, t2.closing_ts)
            and t1.account_id != t2.account_id
            and t1.side = t2.side
            and t1.symbol not in ('BTCUSDT.PERP','ETHUSDT.PERP','SOLUSDT.PERP','XRPUSDT.PERP','BNBUSDT.PERP','DOGEUSDT.PERP')
)
SELECT DISTINCT *
FROM joined
ORDER BY symbol, open_ts1;
"""

try:
    cop_df = dd.query(cop_query).to_df()

    if len(cop_df) > 0:
        unique_pairs_cop = set(tuple(sorted([a1, a2])) for a1, a2 in zip(cop_df.account_id1, cop_df.account_id2))

        # Union-Find 알고리즘으로 그룹 찾기
        groups = []

        for a, b in unique_pairs_cop:
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

        connected_groups = [sorted(list(g)) for g in groups]

        print(f"탐지 결과:")
        print(f"  • 의심 거래 패턴: {len(cop_df)}개")
        print(f"  • 고유 계정 페어: {len(unique_pairs_cop)}개")
        print(f"  • 연결된 그룹: {len(connected_groups)}개")

        # 그룹별 PnL 분석
        group_pnls = []

        for group in connected_groups:
            group_set = set(group)
            sub = cop_df[
                cop_df['account_id1'].isin(group_set) | cop_df['account_id2'].isin(group_set)
                ]

            total_pos = sub[['rpnl1', 'rpnl2']].clip(lower=0).sum().sum()
            total_neg = sub[['rpnl1', 'rpnl2']].clip(upper=0).sum().sum()
            total_pnl = total_pos + total_neg

            group_pnls.append({
                'group': group,
                'member_count': len(group),
                'trade_count': len(sub),
                'pnl_positive_sum': total_pos,
                'pnl_negative_sum': total_neg,
                'pnl_total': total_pnl
            })

        group_pnls_df = (
            pd.DataFrame(group_pnls)
            .sort_values('pnl_total', ascending=False)
            .reset_index(drop=True)
        )

        # IP 중복 계산
        cnts = []
        for group in group_pnls_df['group']:
            group_ips = Ip[Ip.account_id.isin(group)].ip
            cnt = len([kk for kk, vv in Counter(group_ips).items() if vv > 1])
            cnts.append(cnt)

        group_pnls_df['shared_ip_count'] = cnts

        total_coop_profit = group_pnls_df['pnl_total'].sum()
        groups_with_shared_ip = (group_pnls_df['shared_ip_count'] > 0).sum()

        print(f"\n  수익 분석:")
        print(f"    총 순수익: ${total_coop_profit:,.2f}")
        print(f"    평균 그룹 수익: ${total_coop_profit / len(group_pnls_df):,.2f}")
        print(f"    공유 IP 보유 그룹: {groups_with_shared_ip}개")

        print(f"\n  탐지된 Cooperative Trading 그룹 (상위 3개):")
        for idx, row in group_pnls_df.head(3).iterrows():
            print(f"\n    그룹 {idx + 1}:")
            print(f"      멤버: {', '.join(row['group'])}")
            print(f"      멤버 수: {row['member_count']}명")
            print(f"      거래 수: {row['trade_count']}개")
            print(f"      순수익: ${row['pnl_total']:,.2f}")
            print(f"      공유 IP: {row['shared_ip_count']}개")

            # 공유 IP 상세 정보
            if row['shared_ip_count'] > 0:
                group_ip_data = Ip[Ip.account_id.isin(row['group'])]
                ip_counter = Counter(group_ip_data['ip'])
                shared_ips = {ip: count for ip, count in ip_counter.items() if count > 1}
                print(f"      ⚠️  공유 IP 상세:")
                for ip, count in list(shared_ips.items())[:3]:  # 상위 3개만
                    accounts = group_ip_data[group_ip_data['ip'] == ip]['account_id'].tolist()
                    print(f"         {ip}: {count}명 ({', '.join(accounts[:3])})")
    else:
        print("  ⚠️  탐지된 패턴 없음")
        connected_groups = []
        group_pnls_df = pd.DataFrame()

    print()
except Exception as e:
    print(f"  ✗ Cooperative Trading 탐지 실패: {e}\n")
    cop_df = pd.DataFrame()
    connected_groups = []
    group_pnls_df = pd.DataFrame()

# ============================================================================
# 통합 분석 및 리포트
# ============================================================================

print("[5/6] 통합 분석 중...")
print("-" * 80)

# 전체 부정 거래 계정 집계
total_fraud_accounts = set()

if len(funding_hunters) > 0:
    total_fraud_accounts.update(funding_hunters)

if len(wash_df) > 0:
    total_fraud_accounts.update(wash_df['account_id1'].unique())
    total_fraud_accounts.update(wash_df['account_id2'].unique())

if len(connected_groups) > 0:
    for group in connected_groups:
        total_fraud_accounts.update(group)

# 전체 통계
total_accounts = Trade['account_id'].nunique()
fraud_rate = (len(total_fraud_accounts) / total_accounts * 100) if total_accounts > 0 else 0

# 총 수익 계산
total_profit = 0
if len(funding_hunters) > 0:
    total_profit += (
                Funding[Funding.account_id.isin(funding_hunters)].groupby('account_id').sum()['funding_fee'] * -1).sum()
if len(pair_net_pnl) > 0:
    total_profit += pair_net_pnl['pair_net_pnl'].sum()
if len(group_pnls_df) > 0:
    total_profit += group_pnls_df['pnl_total'].sum()

print("통합 부정거래 분석 결과:")
print(f"  • 전체 계정 수: {total_accounts:,}명")
print(f"  • 부정거래 연루 계정: {len(total_fraud_accounts):,}명")
print(f"  • 부정거래 비율: {fraud_rate:.2f}%")
print(f"  • 총 부정거래 수익 추정: ${total_profit:,.2f}")
print()

print("유형별 탐지 현황:")
print(f"  1. Funding Hunter: {len(funding_hunters)}명")
print(f"  2. Wash Trading: {len(unique_pairs) if 'unique_pairs' in locals() else 0}개 페어")
print(f"  3. Cooperative Trading: {len(connected_groups)}개 그룹")
print()

# ============================================================================
# 결과 저장
# ============================================================================

print("[6/6] 결과 저장 중...")
print("-" * 80)

try:
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # 1. Funding Hunters
        if len(funding_hunters) > 0:
            funding_result = fund_df[fund_df.b_funding_hunter == 1][
                ['account_id', 'symbol', 'side', 'leverage', 'amount', 'total_funding', 'holding_minutes']
            ]
            funding_result.to_excel(writer, sheet_name='Funding_Hunters', index=False)
            print(f"  ✓ Funding Hunters 저장 완료 ({len(funding_result)}행)")

        # 2. Wash Trading Pairs
        if len(pair_net_pnl) > 0:
            wash_result = pair_net_pnl.copy()
            wash_result['account1'] = wash_result['pair'].apply(lambda x: x[0])
            wash_result['account2'] = wash_result['pair'].apply(lambda x: x[1])
            wash_result = wash_result[['account1', 'account2', 'net_pnl1', 'net_pnl2', 'pair_net_pnl']]
            wash_result.to_excel(writer, sheet_name='Wash_Trading_Pairs', index=False)
            print(f"  ✓ Wash Trading Pairs 저장 완료 ({len(wash_result)}행)")

        # 3. Cooperative Trading Groups
        if len(group_pnls_df) > 0:
            coop_result = group_pnls_df.copy()
            coop_result['group_members'] = coop_result['group'].apply(lambda x: ', '.join(x))
            coop_result = coop_result[['group_members', 'member_count', 'trade_count',
                                       'pnl_positive_sum', 'pnl_negative_sum', 'pnl_total', 'shared_ip_count']]
            coop_result.to_excel(writer, sheet_name='Cooperative_Groups', index=False)
            print(f"  ✓ Cooperative Trading Groups 저장 완료 ({len(coop_result)}행)")

        # 4. 통합 요약
        summary_data = {
            '구분': ['전체 계정 수', '부정거래 계정 수', '부정거래 비율(%)',
                   'Funding Hunter', 'Wash Trading 페어', 'Cooperative Trading 그룹',
                   '총 부정거래 수익($)'],
            '값': [total_accounts, len(total_fraud_accounts), round(fraud_rate, 2),
                  len(funding_hunters), len(unique_pairs) if 'unique_pairs' in locals() else 0,
                  len(connected_groups), round(total_profit, 2)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"  ✓ 통합 요약 저장 완료")

        # 5. 전체 부정거래 계정 리스트
        if len(total_fraud_accounts) > 0:
            fraud_accounts_df = pd.DataFrame({
                'account_id': list(total_fraud_accounts),
                'fraud_types': [
                    ', '.join([
                        'Funding Hunter' if acc in funding_hunters else '',
                        'Wash Trading' if (len(wash_df) > 0 and (acc in wash_df['account_id1'].values or acc in wash_df[
                            'account_id2'].values)) else '',
                        'Cooperative Trading' if any(acc in g for g in connected_groups) else ''
                    ]).strip(', ').replace(', , ', ', ')
                    for acc in total_fraud_accounts
                ]
            })
            fraud_accounts_df = fraud_accounts_df[fraud_accounts_df['fraud_types'] != '']
            fraud_accounts_df.to_excel(writer, sheet_name='All_Fraud_Accounts', index=False)
            print(f"  ✓ 전체 부정거래 계정 리스트 저장 완료 ({len(fraud_accounts_df)}행)")

    print(f"\n  ✓ 결과 파일 저장 완료: {OUTPUT_FILE}")
except Exception as e:
    print(f"  ✗ 결과 저장 실패: {e}")

# ============================================================================
# 최종 요약 리포트
# ============================================================================

print("\n" + "=" * 80)
print(" " * 25 + "최종 탐지 리포트")
print("=" * 80)
print()

print("┌─────────────────────────────────────────────────────────────────────────────┐")
print("│ 부정거래 유형별 상세 분석                                                    │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│                                                                               │")
print("│ 1. Funding Hunter (펀딩비 악용)                                              │")
if len(funding_hunters) > 0:
    funding_profit_total = (
                Funding[Funding.account_id.isin(funding_hunters)].groupby('account_id').sum()['funding_fee'] * -1).sum()
    print(f"│    • 탐지 계정: {len(funding_hunters):3d}명                                                       │")
    print(f"│    • 총 수익: ${funding_profit_total:12,.2f}                                               │")
    print(
        f"│    • 평균 수익: ${funding_profit_total / len(funding_hunters):12,.2f}                                               │")
else:
    print("│    • 탐지된 계정 없음                                                        │")
print("│                                                                               │")
print("│ 2. Wash Trading (자전거래)                                                   │")
if len(pair_net_pnl) > 0:
    print(f"│    • 탐지 페어: {len(pair_net_pnl):3d}개                                                      │")
    print(f"│    • 총 수익: ${pair_net_pnl['pair_net_pnl'].sum():12,.2f}                                               │")
    print(
        f"│    • 평균 수익: ${pair_net_pnl['pair_net_pnl'].mean():12,.2f}                                               │")
else:
    print("│    • 탐지된 패턴 없음                                                        │")
print("│                                                                               │")
print("│ 3. Cooperative Trading (공모거래)                                            │")
if len(group_pnls_df) > 0:
    print(f"│    • 탐지 그룹: {len(group_pnls_df):3d}개                                                      │")
    print(f"│    • 총 수익: ${group_pnls_df['pnl_total'].sum():12,.2f}                                               │")
    print(
        f"│    • 공유 IP 그룹: {(group_pnls_df['shared_ip_count'] > 0).sum():3d}개                                                   │")
else:
    print("│    • 탐지된 그룹 없음                                                        │")
print("│                                                                               │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│ 종합 통계                                                                     │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print(f"│    • 전체 거래 계정: {total_accounts:6,}명                                                   │")
print(f"│    • 부정거래 연루: {len(total_fraud_accounts):6,}명 ({fraud_rate:5.2f}%)                                      │")
print(f"│    • 추정 부정 수익: ${total_profit:12,.2f}                                               │")
print("│                                                                               │")
print("└─────────────────────────────────────────────────────────────────────────────┘")
print()

print("=" * 80)
print(" " * 25 + "탐지 시스템 종료")
print("=" * 80)
print()
print(f"✓ 분석 완료")
print(f"✓ 결과 파일: {OUTPUT_FILE}")
print(f"✓ 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()