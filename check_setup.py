"""
Quick Start Guide for Wash Trading Detection Dashboard
실행 전 체크리스트와 가이드
"""

# ============================================================================
# 1. 의존성 설치 확인
# ============================================================================

print("="*70)
print("Step 1: Checking Dependencies...")
print("="*70)

import sys

# 필수 패키지 확인
required_packages = {
    'dash': 'pip install dash>=2.14.0',
    'dash_bootstrap_components': 'pip install dash-bootstrap-components>=1.5.0',
    'plotly': 'pip install plotly>=5.17.0',
    'pandas': 'pip install pandas',
    'duckdb': 'pip install duckdb',
    'networkx': 'pip install networkx>=3.2',
    'openpyxl': 'pip install openpyxl',
}

missing_packages = []

for package, install_cmd in required_packages.items():
    try:
        __import__(package)
        print(f"✓ {package:30s} ... OK")
    except ImportError:
        print(f"✗ {package:30s} ... MISSING")
        missing_packages.append((package, install_cmd))

if missing_packages:
    print("\n" + "="*70)
    print("Missing packages detected! Please install:")
    print("="*70)
    for pkg, cmd in missing_packages:
        print(f"  {cmd}")
    print("\nOr install all at once:")
    print("  pip install -r requirements.txt")
    print("="*70)
    sys.exit(1)

print("\n✓ All dependencies installed!\n")

# ============================================================================
# 2. 파일 구조 확인
# ============================================================================

print("="*70)
print("Step 2: Checking File Structure...")
print("="*70)

from pathlib import Path

base_dir = Path(__file__).parent

required_files = {
    'bonus/washTrading.py': 'Core detection module',
    'bonus/washTrading_api.py': 'API wrapper',
    'dashboard/app.py': 'Main dashboard application',
    'dashboard/components/overview.py': 'Overview tab',
    'dashboard/components/timeseries.py': 'Time series tab',
    'dashboard/components/network.py': 'Network tab',
    'dashboard/components/results_table.py': 'Results table tab',
    'dashboard/components/profiles.py': 'Profiles tab',
    'dashboard/assets/custom.css': 'Custom CSS styles',
}

missing_files = []

for file_path, description in required_files.items():
    full_path = base_dir / file_path
    if full_path.exists():
        print(f"✓ {file_path:50s} ... OK")
    else:
        print(f"✗ {file_path:50s} ... MISSING")
        missing_files.append(file_path)

if missing_files:
    print("\n" + "="*70)
    print("Missing files detected!")
    print("="*70)
    for f in missing_files:
        print(f"  {f}")
    print("="*70)
    sys.exit(1)

print("\n✓ All files present!\n")

# ============================================================================
# 3. 샘플 데이터 확인
# ============================================================================

print("="*70)
print("Step 3: Checking for Sample Data...")
print("="*70)

sample_data = base_dir / 'problem_data_final.xlsx'

if sample_data.exists():
    print(f"✓ Sample data found: {sample_data}")
    print(f"  File size: {sample_data.stat().st_size / 1024 / 1024:.2f} MB")
else:
    print("⚠ Sample data not found!")
    print(f"  Expected location: {sample_data}")
    print("  You can still run the dashboard and upload your own data.")

print()

# ============================================================================
# 4. 실행 정보
# ============================================================================

print("="*70)
print("Ready to Launch!")
print("="*70)
print()
print("To start the dashboard, run:")
print()
print("  python dashboard/app.py")
print()
print("Then open your browser and navigate to:")
print()
print("  http://localhost:8050")
print()
print("="*70)
print()
print("Quick Tips:")
print("  1. Upload your Excel file with trade data")
print("  2. Adjust detection parameters as needed")
print("  3. Click 'Run Detection' to analyze")
print("  4. Explore results in different tabs")
print("  5. Export findings for further analysis")
print()
print("="*70)
print()
print("For detailed usage instructions, see:")
print("  DASHBOARD_README.md")
print()
print("="*70)
