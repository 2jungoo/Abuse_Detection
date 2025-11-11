# Python 프로젝트 시작 가이드

이 문서는 Python 프로젝트의 기본적인 개발 환경 세팅 및 필수 패키지 설치 방법을 안내합니다.

## 1. Python 가상환경(venv) 생성

```bash
python3 -m venv venv
```

-   위 명령어로 `venv`라는 이름의 가상환경이 생성됩니다.

## 2. 가상환경 활성화

-   **macOS/Linux**:
    ```bash
    source venv/bin/activate
    ```
-   **Windows**:
    ```bash
    .\venv\Scripts\activate
    ```

## 3. 필수 패키지 설치

이 프로젝트는 다음 패키지들을 필요로 합니다:

-   pandas
-   duckdb

아래 명령어로 설치하세요:

```bash
pip install pandas duckdb
```

## 4. requirements.txt 생성 (선택)

설치된 패키지를 `requirements.txt`로 저장하려면:

```bash
pip freeze > requirements.txt
```

다른 환경에서 동일한 패키지를 설치하려면:

```bash
pip install -r requirements.txt
```
