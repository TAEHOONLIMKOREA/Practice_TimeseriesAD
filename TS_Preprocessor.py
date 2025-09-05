import pandas as pd
import os


def preprocess_timeseries(csv_path: str, save_path: str = None) -> pd.DataFrame:
    """
    시계열 CSV 데이터를 불러와서 다음과 같이 전처리합니다:
    1. 처음부터 끝까지 값이 0인 칼럼 제거
    2. bool 타입 칼럼 제거
    """
    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 1. 처음부터 끝까지 값이 모두 같은 칼럼 제거
    const_cols = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=const_cols)

    # 2. bool 타입 칼럼 제거
    bool_cols = df.select_dtypes(include=['bool']).columns
    df = df.drop(columns=bool_cols)

    # 3. 저장
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"[INFO] 전처리된 데이터 저장 완료 → {save_path}")

    return df

# 사용 예시
if __name__ == "__main__":
    
    print("현재 작업 디렉토리:", os.getcwd())

    processed_df = preprocess_timeseries("./3PDX_TimeseriesAD/data/MachineDataLog/20221027_1045_Environment.csv", "./3PDX_TimeseriesAD/data/MachineDataLog/preprocessing.csv")
    print(processed_df.head())
