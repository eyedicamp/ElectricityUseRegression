import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from tensorflow import keras

#from MyMetrics import mape, smape, rmse, mae, coeff_determination
from MyModels import *

def test_predict_mergedInput_mergedOutput_forcingMax(model, Xmerged, ymerged, title, _ylim=None):
    # X 데이터 분리
    X = Xmerged[0]
    conditionalX = Xmerged[1]
    # y 데이터 분리
    y = ymerged[0]
    maxy = ymerged[1]

    ts_pred = [] # 시계열 예측값을 저장할 리스트
    actual = y.reshape(y.shape[0] * y.shape[1]) # 시계열 정답을 저장할 리스트
    
    ts_max_pred = [] # 24시간 예측 단위로, 예측으로 생성한 시계열 데이터의 max 저장할 리스트
    max_y_values = [] # 24시간 예측 단위로, 정답에 해당하는 시계열 데이터의 max 저장할 리스트
    
    for i in range(X.shape[0]):
        x_sample1 = X[i].reshape(1, len(X[i]), 1)
        x_sample2 = conditionalX[i].reshape(1, len(conditionalX[i]), 1)
        
        [y_hat, y_hat_max] = model.predict([x_sample1, x_sample2])
        
        if False:
            print("1] hourly forecast : ", y_hat)
            print("2] my calc sum 1 : ", max(max(y_hat)))
            print("3] Keras calc sum 2 : ", y_hat_max)
            print("4] conditioning on X : ", conditionalX[i][0])
            print("5] expecting y : ", maxy[i][0], end='\n\n')
            
        y_hat_values = y_hat[0].reshape(len(y_hat[0]),).tolist()

        if len(ts_pred) == 0:
            ts_pred = y_hat_values
        else:
            ts_pred = ts_pred + y_hat_values
            
        ts_max_pred.append(y_hat_max)
        max_y_values.append(maxy[i][0])
            
    # 시계열 데이터 예측값을 plot
    print("TimeSeries MAE : %d"%(int(mean_absolute_error(actual, ts_pred))))
    #plt.figure(figsize=(20,10))
    plt.figure()
    plt.plot(actual, label='actual')
    plt.plot(ts_pred, label='forecast')
    plt.title(title) 
    plt.legend()
    plt.xlabel('hours')
    if _ylim is not None:
        plt.ylim(_ylim)
    plt.show()
        
    # 24시간 예측 단위로, 시계열 데이터의 총합을 plot
    print("TimeSeriesSum MAE : %d"%(int(mean_absolute_error(np.array(max_y_values), np.array(ts_max_pred)))))
    #print(sum_y_values)
    #print(ts_sum_pred)
    plt.figure()
    plt.bar(np.arange(len(max_y_values))-0.1, max_y_values, width=0.3, label='actual')
    plt.bar(np.arange(len(ts_max_pred))+0.1, ts_max_pred, width=0.3, label='forecast')
    plt.legend()
    plt.title("Comparison: daily power max")
    plt.xlabel('days')
    plt.show()
    
def build_transformer_conditionalInput_maxOutput_model(
    # 전력 PEAK를 조건부 입력으로 제공 + 출력층에서 시계열 데이터의 PEAK를 출력
    input_shape, conditional_input_shape, num_outputs,
    head_size, num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0, mlp_dropout=0,
    ):
    
    LayerNormEps = 1e-6
    """
    인코더 
    """
    inputs = keras.Input(shape=input_shape, name="ts_input") # 시계열 학습 데이터
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    # PEAK는 최대값을 예측하는 문제니까, max pooling이 더 좋으려나?
    x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
    
    # 조건부 입력(일간 총 사용량 예측치) 데이터를 추가로 입력 받음
    conditional_inputs = keras.Input(shape=conditional_input_shape, name="ts_max_input")
    # 조건부 입력(일간 총 사용량)의 단위가 크고, 다른 특징들은 이미 정규화가 되어있어서
    # 조건부 입력을 그대로 사용하면 이로 인한 업데이트가 너무나 과도해 질 수 있음
    # 따라서, 조건부 입력에 대해서도 정규화를 실시함
    normalized_conditional_inputs \
    = layers.LayerNormalization(epsilon=LayerNormEps)(conditional_inputs)
    # 인코더의 최종 출력 + 조건부 입력을 인코더 계층의 최종 출력으로 하고, 디코더로 전달
    x = x + normalized_conditional_inputs
    
    """
    디코더
    """
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",
                         kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dropout(mlp_dropout)(x)
    
    # 양수만 나와야 하니까, linear 대신 relu
    timeseries_outputs = layers.Dense(num_outputs, activation="relu",
                                      kernel_initializer='random_normal',
                                      bias_initializer='zeros',
                                      name='ts_output')(x) 
    
    #시계열 출력을 다시 입력으로 받아서, 출력의 element-wise max을 계산
    max_outputs = layers.Lambda(lambda v: K.max(v), 
                                output_shape=(1,1),
                                name="ts_max")(timeseries_outputs)
    # 최종 모델을 리턴
    return keras.Model(inputs = [inputs, conditional_inputs], 
                       outputs = [timeseries_outputs, max_outputs])