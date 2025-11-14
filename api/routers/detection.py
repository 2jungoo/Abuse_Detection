"""
Detection API Router
탐지 시스템 관련 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from api.data_aggregator import get_aggregator

router = APIRouter()


@router.get("/stats")
async def get_stats():
    """
    전체 탐지 통계
    
    Returns:
        - total_detections: 전체 탐지 건수
        - wash_trading: 증정금 녹이기 탐지 건수
        - funding_fee: 펀딩비 악용 탐지 건수
        - cooperative: 공모거래 탐지 건수
        - total_sanctions: 전체 제재 건수
        - bonus_details: Bonus 상세 (bot_tier, manual_tier, suspicious)
        - funding_details: Funding 상세 (critical, high, medium)
        - cooperative_details: Cooperative 상세 (critical, high, medium)
    """
    try:
        aggregator = get_aggregator()
        stats = aggregator.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.get("/detections")
async def get_detections(
    model: Optional[str] = Query(None, description="Filter by model: wash, funding, cooperative"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
):
    """
    모든 탐지 케이스 통합 리스트 (제재 여부 포함)
    
    Parameters:
        - model: 필터링할 모델 (wash, funding, cooperative)
        - limit: 반환할 최대 개수
    
    Returns:
        List of detections with:
        - id: 탐지 케이스 ID
        - model: 모델 유형 (wash, funding, cooperative)
        - timestamp: 탐지 시간
        - type: 탐지 유형
        - accounts: 관련 계정 리스트
        - score: 점수
        - is_sanctioned: 제재 여부
        - sanction_id: 제재 케이스 ID (제재된 경우)
        - sanction_type: 제재 유형 (제재된 경우)
        - details: 상세 설명
        - raw: 원본 데이터
    """
    try:
        aggregator = get_aggregator()
        detections = aggregator.get_detections()
        
        # 모델 필터링
        if model:
            detections = [d for d in detections if d['model'] == model]
        
        # 점수 기준 정렬 (높은 순)
        detections = sorted(detections, key=lambda x: x.get('score', 0), reverse=True)
        
        # 제한
        if limit:
            detections = detections[:limit]
        
        return detections
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting detections: {str(e)}")


@router.get("/sanctions")
async def get_sanctions(
    model: Optional[str] = Query(None, description="Filter by model: wash, funding, cooperative"),
    limit: Optional[int] = Query(None, description="Limit number of results"),
):
    """
    모든 제재 케이스 통합 리스트
    
    Parameters:
        - model: 필터링할 모델 (wash, funding, cooperative)
        - limit: 반환할 최대 개수
    
    Returns:
        List of sanctions with:
        - id: 제재 케이스 ID
        - model: 모델 유형 (wash, funding, cooperative)
        - timestamp: 탐지 시간
        - type: 제재 유형
        - accounts: 관련 계정 리스트
        - score: 점수
        - details: 상세 설명
        - raw: 원본 데이터
    """
    try:
        aggregator = get_aggregator()
        sanctions = aggregator.get_sanctions()
        
        # 모델 필터링
        if model:
            sanctions = [s for s in sanctions if s['model'] == model]
        
        # 점수 기준 정렬 (높은 순)
        sanctions = sorted(sanctions, key=lambda x: x.get('score', 0), reverse=True)
        
        # 제한
        if limit:
            sanctions = sanctions[:limit]
        
        return sanctions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sanctions: {str(e)}")


@router.get("/timeseries")
async def get_timeseries(
    interval: str = Query("1h", description="Time interval: 1h, 1d"),
):
    """
    시간별 탐지 추이 데이터
    
    Parameters:
        - interval: 시간 간격 (1h=1시간, 1d=1일)
    
    Returns:
        List of time series data:
        - time: Unix timestamp
        - WASH_TRADING: 증정금 녹이기 건수
        - FUNDING_FEE: 펀딩비 악용 건수
        - COOPERATIVE: 공모거래 건수
    """
    try:
        aggregator = get_aggregator()
        timeseries = aggregator.get_timeseries_data()
        
        # TODO: interval에 따라 집계 변경 (현재는 1시간 단위)
        
        return timeseries
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting timeseries: {str(e)}")


@router.get("/top-accounts")
async def get_top_accounts(
    limit: int = Query(10, description="Number of top accounts to return", ge=1, le=100),
):
    """
    상위 위반 계정 리스트
    
    Parameters:
        - limit: 반환할 계정 수 (기본 10개)
    
    Returns:
        List of top accounts:
        - account_id: 계정 ID
        - total_cases: 총 탐지 건수
        - total_profit_loss: 총 손익
        - profits: 모델별 수익 (funding, wash, cooperative)
        - avg_score: 평균 점수
        - max_score: 최고 점수
        - critical_count: Critical 건수
        - high_count: High 건수
    """
    try:
        aggregator = get_aggregator()
        top_accounts = aggregator.get_top_accounts(limit=limit)
        return top_accounts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting top accounts: {str(e)}")


@router.get("/hourly-distribution")
async def get_hourly_distribution():
    """
    시간대별 탐지 분포 (0~23시)
    
    Returns:
        Dict with hour (0-23) as key and count as value
    """
    try:
        aggregator = get_aggregator()
        hourly = aggregator.get_hourly_distribution()
        return hourly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting hourly distribution: {str(e)}")


@router.get("/visualization")
async def get_visualization_data(
    model: Optional[str] = Query(None, description="Filter by model: bonus, funding, cooperative"),
):
    """
    시각화 데이터 (네트워크 그래프, 히트맵 등)
    
    Parameters:
        - model: 특정 모델의 시각화 데이터만 반환 (bonus, funding, cooperative)
    
    Returns:
        Visualization data for each model
    """
    try:
        aggregator = get_aggregator()
        vis_data = aggregator.get_visualization_data()
        
        if model:
            if model in vis_data:
                return {model: vis_data[model]}
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        
        return vis_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting visualization data: {str(e)}")


@router.post("/reload")
async def reload_data():
    """
    데이터 강제 리로드
    
    캐시를 무시하고 output 파일들을 다시 읽어옵니다.
    """
    try:
        aggregator = get_aggregator()
        aggregator.get_all_data(force_reload=True)
        return {"status": "success", "message": "Data reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading data: {str(e)}")


@router.get("/raw/{model}")
async def get_raw_data(
    model: str,
):
    """
    특정 모델의 원본 데이터 반환
    
    Parameters:
        - model: 모델 이름 (bonus, funding, cooperative)
    
    Returns:
        Raw data including config, visualization, sanctions, etc.
    """
    try:
        aggregator = get_aggregator()
        all_data = aggregator.get_all_data()
        
        if model not in all_data:
            raise HTTPException(status_code=404, detail=f"Model '{model}' not found")
        
        return all_data[model]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting raw data: {str(e)}")


@router.get("/trade-pairs/{model}")
async def get_trade_pairs(
    model: str,
):
    """
    특정 모델의 trade pairs 데이터 반환
    
    Parameters:
        - model: 모델 이름 (wash, cooperative, funding)
    
    Returns:
        List of trade pairs for the specified model
    """
    try:
        if model not in ['wash', 'cooperative', 'funding']:
            raise HTTPException(status_code=400, detail="Model must be one of: wash, cooperative, funding")
        
        aggregator = get_aggregator()
        trade_pairs = aggregator.get_trade_pairs(model)
        return trade_pairs
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trade pairs: {str(e)}")


@router.get("/cooperative-groups")
async def get_cooperative_groups():
    """
    Cooperative groups 데이터 반환
    
    Returns:
        List of cooperative groups
    """
    try:
        aggregator = get_aggregator()
        groups = aggregator.get_cooperative_groups()
        return groups
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting cooperative groups: {str(e)}")


@router.get("/account/{account_id}/trades")
async def get_account_trades(
    account_id: str,
):
    """
    특정 계정의 거래 이력 반환
    
    Parameters:
        - account_id: 계정 ID
    
    Returns:
        List of trades for the account
    """
    try:
        aggregator = get_aggregator()
        trades = aggregator.get_account_trade_history(account_id)
        return trades
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting account trades: {str(e)}")


@router.get("/case/{model}/{case_id}")
async def get_case_detail(
    model: str,
    case_id: str,
):
    """
    특정 케이스의 상세 정보 반환
    
    Parameters:
        - model: 모델 이름 (wash, funding, cooperative)
        - case_id: 케이스 ID
    
    Returns:
        Case detail data
    """
    try:
        if model not in ['wash', 'funding', 'cooperative']:
            raise HTTPException(status_code=400, detail="Model must be one of: wash, funding, cooperative")
        
        aggregator = get_aggregator()
        case_detail = aggregator.get_case_detail(model, case_id)
        
        if case_detail is None:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found for model {model}")
        
        return case_detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting case detail: {str(e)}")

