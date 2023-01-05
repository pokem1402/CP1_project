from typing import List, Iterator
import pandas as pd
import numpy as np

# 1. 데이터를 불러오는 함수
def load_data(src : str) -> List[pd.DataFrame]:

    data : pd.DataFrame = pd.read_csv(src)
    x : pd.DataFrame = data.iloc[:, :-1]
    y : pd.DataFrame = data.iloc[:, -1]

    return x, y

# 3. 데이터의 row를 셔플하는 함수
def data_shuffle(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:

    idx = np.random.permutation(x.index)
    x = x.reindex(idx)
    y = y.reindex(idx)
    return x, y


# 4. 학습 데이터와 테스트 데이터를 분리하는 함수
def train_test_split(x: pd.DataFrame, y: pd.DataFrame,
                     test_size: float = 0.2) -> List[pd.DataFrame]:

    # 데이터의 row 수
    n: int = x.shape[0]

    test_x: pd.DataFrame = x.iloc[:int(n*test_size)]
    test_y: pd.DataFrame = y.iloc[:int(n*test_size)]
    train_x: pd.DataFrame = x.iloc[int(n*test_size):]
    train_y: pd.DataFrame = y.iloc[int(n*test_size):]
    return train_x, train_y, test_x, test_y


# 5. mini-batch 생성기
def mini_batch_generator(x: pd.DataFrame, y: pd.DataFrame,
                         batch_size: int = 4) -> Iterator[pd.DataFrame]:

    n: int = x.shape[0]

    x, y = data_shuffle(x, y)

    for j in range(n//batch_size):  # 데이터 크기를 배치 사이즈로 나눈만큼 반복한다.
        yield x.iloc[j*batch_size:j*batch_size+batch_size], \
            y.iloc[j*batch_size:j*batch_size+batch_size]
