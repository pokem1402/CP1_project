import numpy as np
from typing import Union, List
import pandas as pd
from data.data import mini_batch_generator

class NN:
    
    def __init__(self):
        
        self.w = None
        self.b = None
     

    # 2. 가중치와 편향을 초기화하는 함수
    # 초기화에 관한 부분에 대한 reference : https://reniew.github.io/13/
    def initialize_parameters(self, x: np.ndarray or pd.DataFrame, d_out: int,
                            init: str = "xavier_normal",
                            use_bias: bool = True) -> None:

        d_in: int = x.shape[1]  # 입력 변수의 차원

        # 편향을 사용하는 경우
        if use_bias:
            # 초기화를 수월하게 진행할 수 있도록
            # 가중치의 크기에 편향의 크기를 넣어서 계산한다.
            w_size: tuple = (d_in+1, d_out)
        # 편향을 사용하지 않는 경우
        else:
            w_size: tuple = (d_in, d_out)

        # he 초기화의 경우
        if init == "he":
            # 분산의 값이 (2/입력)의 root 값
            v: float = (2/d_in)**0.5
            w: np.array = np.random.normal(0, v, w_size)
        # Xavier 초기화 중 정규 분포를 통한 초기화인 경우
        elif init == "xavier_normal":
            v: float = (2/(d_in+d_out))**0.5
            w: np.array = np.random.normal(0, v, w_size)
        # Xavier 초기화 중 균일 분포를 통한 초기화인 경우
        elif init == "xavier_uniform":
            u: float = (6/(d_in+d_out))**0.5
            w: np.array = np.random.uniform(-u, u, w_size)
        else:
            raise ValueError(
             f"There is no implementation for [{init}] initialization")

        # 편향을 사용하는 경우
        if use_bias:
            # 가중치와 편향을 분리
            self.w = w[:-1]
            self.b = w[-1]
        # 편향을 사용하지 않는 경우
        else:
            # 가중치만 저장
            self.w = w

    # 6. sigmoid 함수
    def sigmoid(self, x: np.ndarray or pd.DataFrame):
        return 1./(1.+np.exp(-x))
    
    # 시그모이드를 미분한 함수
    def sigmoid_derivative(self, x: np.ndarray or pd.DataFrame) \
                        -> Union[np.ndarray, pd.DataFrame]:
        s = self.sigmoid(x)
        return s * (1-s)

    # 7. Binary Cross entropy
    def binaryCrossEntropy(self, y_true: np.ndarray or pd.DataFrame,
                                y_pred: np.ndarray or pd.DataFrame,
                                    c: float = 1e-6) -> float:

        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        # y의 실제값과 y의 예측값의 shape가 같지 않다면 ValueError
        if y_true.shape != y_pred.shape:
            raise ValueError(
                "Shape of true value of y and prediction of y must be equal.")

        return - np.mean(y_true * np.log(y_pred+c)+ (1.-y_true)*np.log(1.-y_pred+c))


    # 8. 정확도 연산 기능
    def get_accuracy(self, y_true: np.ndarray or pd.DataFrame,
                    y_pred: np.ndarray or pd.DataFrame):

        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        # y의 실제값과 y의 예측값의 shape가 같지 않다면 ValueError
        if y_true.shape != y_pred.shape:
            raise ValueError(
                "Shape of true value of y and prediction of y must be equal.")

        return np.mean(y_pred == y_true)
    
    # 8. 훈련 데이터를 통해 정확도를 계산하는 함수
    def compute_network_train_acc(self, x: np.ndarray or pd.DataFrame,
                                y: np.ndarray or pd.DataFrame,
                                batch_size: int = 4) -> float:
        acc = []
        # generator로부터 x와 y 값을 받아오는 generator
        for batch_x, batch_y in mini_batch_generator(x, y, batch_size=batch_size):
            # 행렬 연산
            logit = self.sigmoid(np.matmul(batch_x, self.w) + (self.b if self.b is not None else 0))
            # logit 값이 0.5보다 크다면 class 1 아니라면 0
            y_pred = (logit >= 0.5).applymap(int)
            acc.append(self.get_accuracy(batch_y, y_pred))
        return np.mean(acc)

    # 역전파를 수행하는 함수
    # 역전파를 위한 미분 계산 : https://smwgood.tistory.com/6
    def backprop(self, x: np.ndarray or pd.DataFrame,
                y: np.ndarray or pd.DataFrame,
                logit: np.ndarray or pd.DataFrame,
                learning_rate: float = 1e-4) -> None:

        # pandas 형식을 numpy array로 형식을 바꿔주고 차원 축을 동일하게 고정해둔다.
        x = np.asarray(x)
        y = np.asarray(np.squeeze(y))[:, np.newaxis]
        logit = np.asarray(np.squeeze(logit))[:, np.newaxis]

        # sigmoid를 activation으로 사용하는 cross entropy의 미분 값 계산
        target_error: np.ndarray = y - logit

        # 이전 항으로 미분 값 이전을 위한 체인 룰 계산
        target_error_derivative: np.ndarray = target_error * \
            self.sigmoid_derivative(logit)

        # 가중치 업데이트
        self.w += learning_rate * \
            x.T.dot(target_error_derivative)

        if self.b is not None:
            # 편향 업데이트
            if len(self.b.shape) == 1:
                b_delta = learning_rate * np.mean(target_error_derivative, axis=0)
            else:
                b_delta = learning_rate * \
                    np.mean(target_error_derivative, axis=1)[:, np.newaxis]
            self.b += b_delta

    # 9. 훈련 데이터를 통해 신경망 연산을 수행하는 함수
    def compute_network_train(self, x: np.ndarray or pd.DataFrame,
                            y: np.ndarray or pd.DataFrame,
                            batch_size: int = 4, epochs: int = 1) -> None:

        # 손실값을 연산하는 함수
        for epoch in range(epochs):
            errors: list = []
            for batch_x, batch_y in mini_batch_generator(x, y, batch_size=batch_size):
                logit = self.sigmoid(np.matmul(batch_x, self.w) + (self.b if self.b is not None else 0))
                error = self.binaryCrossEntropy(batch_y, logit)
                self.backprop(batch_x, batch_y, logit)  # 역전파
                errors.append(error)
            errors = np.mean(errors)

            acc = self.compute_network_train_acc(x, y, batch_size)

            print(f"[Epoch {epoch+1}] TrainData - Loss = {np.round(errors, 4)}, Accuracy = {np.round(acc,4)}")

    # 10. 테스트 데이터를 통해 정확도를 연산하는 함수
    def compute_network_test(self, x: np.ndarray or pd.DataFrame,
                            y: np.ndarray or pd.DataFrame):

        logit = self.sigmoid(np.matmul(x, self.w) + (self.b if self.b is not None else 0))
        y_pred = (logit >= 0.5).applymap(int)
        return self.get_accuracy(y, y_pred)
