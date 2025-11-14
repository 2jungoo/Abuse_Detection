"""
통합 부정거래 탐지 시스템
All-in-One Fraud Detection System

모든 탐지 모듈을 한번에 실행합니다.
"""

import sys
from pathlib import Path
from datetime import datetime

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent))


def run_all_detections(data_filepath: str = "problem_data_final.xlsx"):
    """
    모든 탐지 모듈 실행
    
    Args:
        data_filepath: 데이터 파일 경로
    """
    
    print("\n" + "="*80)
    print(" " * 20 + "통합 부정거래 탐지 시스템 v2.0")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"데이터 파일: {data_filepath}")
    print("="*80 + "\n")
    
    results = {}

    # 1. Bonus Laundering 탐지
    print("\n" + "🎁 " + "="*76)
    print("1/3: Bonus Laundering (증정금 녹이기) 탐지 시작...")
    print("="*80)
    try:
        from wash_trading.wash_trading import run_detection as run_bonus_detection
        results['bonus'] = run_bonus_detection(data_filepath)
        print("✓ Bonus Laundering 탐지 완료")
    except Exception as e:
        print(f"✗ Bonus Laundering 탐지 실패: {e}")
        results['bonus'] = {'error': str(e)}
    
    # 2. Funding Hunter 탐지
    print("\n" + "💰 " + "="*76)
    print("2/3: Funding Hunter (펀딩비 악용) 탐지 시작...")
    print("="*80)
    try:
        from funding_fee.funding_hunter import run_detection as run_funding_detection
        results['funding'] = run_funding_detection(data_filepath)
        print("✓ Funding Hunter 탐지 완료")
    except Exception as e:
        print(f"✗ Funding Hunter 탐지 실패: {e}")
        results['funding'] = {'error': str(e)}
    
    # 3. Cooperative Trading 탐지
    print("\n" + "🤝 " + "="*76)
    print("3/3: Cooperative Trading (공모거래) 탐지 시작...")
    print("="*80)
    try:
        from abusing.cooperative_trading import run_detection as run_coop_detection
        results['cooperative'] = run_coop_detection(data_filepath)
        print("✓ Cooperative Trading 탐지 완료")
    except Exception as e:
        print(f"✗ Cooperative Trading 탐지 실패: {e}")
        results['cooperative'] = {'error': str(e)}
    
    # 최종 요약
    print("\n" + "="*80)
    print(" " * 30 + "최종 탐지 요약")
    print("="*80 + "\n")
    
    # Bonus Laundering 요약
    if 'error' not in results.get('bonus', {}):
        bonus = results['bonus']
        print("🎁 Bonus Laundering (증정금 녹이기)")
        print(f"   - 총 후보: {bonus.get('total_candidates', 0)}건")
        print(f"   - 필터 통과: {bonus.get('passed_filter', 0)}건")
        print(f"   - Bot 제재: {bonus.get('bot_sanctions', 0)}건")
        print(f"   - 네트워크 제재: {bonus.get('network_sanctions', 0)}건")
        print(f"   - 출력 위치: {bonus.get('output_directory', 'N/A')}")
    else:
        print("🎁 Bonus Laundering: 실패")
    
    print()
    
    # Funding Hunter 요약
    if 'error' not in results.get('funding', {}):
        funding = results['funding']
        print("💰 Funding Hunter (펀딩비 악용)")
        print(f"   - 총 후보: {funding.get('total_candidates', 0)}건")
        print(f"   - 필터 통과: {funding.get('passed_filter', 0)}건")
        print(f"   - 총 계정: {funding.get('total_accounts', 0)}개")
        print(f"   - 총 펀딩비 수익: ${funding.get('total_funding_profit', 0):,.2f}")
        print(f"   - 출력 위치: {funding.get('output_directory', 'N/A')}")
    else:
        print("💰 Funding Hunter: 실패")
    
    print()
    
    # Cooperative Trading 요약
    if 'error' not in results.get('cooperative', {}):
        coop = results['cooperative']
        print("🤝 Cooperative Trading (공모거래)")
        print(f"   - 총 후보: {coop.get('total_candidates', 0)}건")
        print(f"   - 필터 통과: {coop.get('passed_filter', 0)}건")
        print(f"   - 총 그룹: {coop.get('total_groups', 0)}개")
        print(f"   - 총 순수익: ${coop.get('total_pnl', 0):,.2f}")
        print(f"   - 출력 위치: {coop.get('output_directory', 'N/A')}")
    else:
        print("🤝 Cooperative Trading: 실패")
    
    print("\n" + "="*80)
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="통합 부정거래 탐지 시스템")
    parser.add_argument(
        "data_file", 
        nargs='?', 
        default="problem_data_final.xlsx",
        help="데이터 파일 경로 (기본값: problem_data_final.xlsx)"
    )
    
    args = parser.parse_args()
    
    run_all_detections(args.data_file)
