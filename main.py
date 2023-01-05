
from data.data import load_data, data_shuffle, train_test_split
from net.network import NN
import numpy as np


if __name__ == "__main__":
    # file 위치
    DATA_FILE_PATH = "binary_dataset.csv"

    # (1) file load
    x, y = load_data(DATA_FILE_PATH)

    nn = NN()
    # (2) 파라미터와 편향 생성
    nn.initialize_parameters(x, 1, init="xavier_normal")
    # (3) 데이터 셔플링
    x, y = data_shuffle(x, y)
    # (4) 훈련 데이터와 테스트 데이터 분리
    train_x, train_y, test_x, test_y = train_test_split(x, y)
    # (5) 훈련 데이터를 통한 손실값 및 정확도 계산
    nn.compute_network_train(train_x, train_y, epochs=10)
    # (6) 테스트 데이터 정확도 연산
    test_acc = nn.compute_network_test(test_x, test_y)
    print(f"TestData - Accuracy = {np.round(test_acc,4)}")
