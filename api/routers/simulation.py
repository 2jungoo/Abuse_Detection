"""
Simulation API Router
시뮬레이션 관련 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
from pathlib import Path
from common.data_manager import get_data_manager
from api.data_aggregator import get_aggregator

router = APIRouter()


class AdvanceRequest(BaseModel):
    days: int = 7
    hours: int = 0


@router.get("/status")
async def get_simulation_status():
    """
    시뮬레이션 현재 상태 조회
    
    Returns:
        - current_time: 현재 시뮬레이션 시간
        - status: 시뮬레이션 상태 (running, not_initialized, error)
    """
    try:
        aggregator = get_aggregator()
        status = aggregator.get_simulation_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting simulation status: {str(e)}")


@router.post("/advance")
async def advance_simulation(request: AdvanceRequest):
    """
    시뮬레이션을 지정된 일수만큼 진행
    
    Parameters:
        - days: 진행할 일수 (기본 7일)
        - hours: 진행할 시간수 (기본 0시간)
    
    Process:
        1. data_manager의 advance_model_by_days 호출
        2. run_all_detections.py 실행하여 새 데이터 기반 탐지
        3. 완료 후 새로운 상태 반환
    
    Returns:
        - status: success or error
        - current_time: 진행 후 시뮬레이션 시간
        - message: 진행 결과 메시지
    """
    try:
        print(f"=== Simulation Advance Started ===")
        print(f"Advancing by {request.days} days and {request.hours} hours")
        
        # 1. DataManager를 통해 데이터 진행
        dm = get_data_manager()
        dm.advance_model_by_days(days=request.days, hours=request.hours)
        
        # 2. 탐지 모델 재실행
        print("Running detection models...")
        script_path = Path(__file__).parent.parent.parent / "run_all_detections.py"

        dm.close_connection()
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent)
        )

        dm.reopen_connection()
        
        if result.returncode != 0:
            print(f"Detection failed: {result.stderr}")
            raise Exception(f"Detection execution failed: {result.stderr}")
        
        print("Detection completed successfully")
        
        # 3. 캐시 강제 리로드
        aggregator = get_aggregator()
        aggregator.get_all_data(force_reload=True)
        
        # 4. 새로운 상태 반환
        status = aggregator.get_simulation_status()
        
        print(f"=== Simulation Advance Completed ===")
        
        return {
            'status': 'success',
            'current_time': status.get('current_time'),
            'days_advanced': request.days,
            'hours_advanced': request.hours,
            'message': f'Simulation advanced by {request.days} days and {request.hours} hours'
        }
        
    except Exception as e:
        print(f"Error advancing simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error advancing simulation: {str(e)}")


@router.post("/reset")
async def reset_simulation():
    """
    시뮬레이션을 초기 상태로 리셋 (2025-02-01)
    
    Process:
        1. data_manager의 seed_full_and_model 재호출
        2. run_all_detections.py 실행
        3. 캐시 리로드
    
    Returns:
        - status: success or error
        - current_time: 리셋 후 시뮬레이션 시간
        - message: 리셋 결과 메시지
    """
    try:
        print(f"=== Simulation Reset Started ===")
        
        # 1. DataManager를 통해 초기 상태로 리셋
        dm = get_data_manager()
        dm.seed_full_and_model(year=2025, month=2)
        
        # 2. 탐지 모델 재실행
        print("Running detection models...")
        script_path = Path(__file__).parent.parent.parent / "run_all_detections.py"

        dm.close_connection()
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(script_path.parent)
        )

        dm.reopen_connection()
        
        if result.returncode != 0:
            print(f"Detection failed: {result.stderr}")
            raise Exception(f"Detection execution failed: {result.stderr}")
        
        print("Detection completed successfully")
        
        # 3. 캐시 강제 리로드
        aggregator = get_aggregator()
        aggregator.get_all_data(force_reload=True)
        
        # 4. 새로운 상태 반환
        status = aggregator.get_simulation_status()
        
        print(f"=== Simulation Reset Completed ===")
        
        return {
            'status': 'success',
            'current_time': status.get('current_time'),
            'message': 'Simulation reset to 2025-02-01'
        }
        
    except Exception as e:
        print(f"Error resetting simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting simulation: {str(e)}")
