import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
#from MyMetrics import mape, smape, rmse, mae, coeff_determination


#from numpy import split
#from numpy import array
# --------------------------------------------------------------------
def test_predict(model, X, y, title, _ylim=None):
    """
    학습된 모델, 그리고 test/validation 데이터셋을 입력으로
    예측 결과를 추출하고 시각화 하는 함수
    """
    pred = []
    actual = y.reshape(y.shape[0] * y.shape[1])
    for i in range(X.shape[0]):
        x_sample = X[i].reshape(1, len(X[i]), 1)
        y_hat = model.predict(x_sample)

        y_hat_values = y_hat[0].reshape(len(y_hat[0]),).tolist()

        if len(pred) == 0:
            pred = y_hat_values
        else:
            pred = pred + y_hat_values

    print("RMSE : ", int(np.sqrt(mean_squared_error(actual, pred))))
    
    plt.figure(figsize=(20,10))
    plt.plot(actual, label='actual')
    plt.plot(pred, label='forecast')
    plt.title(title) 
    plt.legend()
    if _ylim is not None:
        plt.ylim(_ylim)
    plt.show()
    
def test_predict_mergedInput(model, Xmerged, y, title, _ylim=None):
    X = Xmerged[0]
    conditionalX = Xmerged[1]

    pred = []
    actual = y.reshape(y.shape[0] * y.shape[1])
    for i in range(X.shape[0]):
        x_sample1 = X[i].reshape(1, len(X[i]), 1)
        x_sample2 = conditionalX[i].reshape(1, len(conditionalX[i]), 1)
        
        y_hat = model.predict([x_sample1, x_sample2])

        y_hat_values = y_hat[0].reshape(len(y_hat[0]),).tolist()

        if len(pred) == 0:
            pred = y_hat_values
        else:
            pred = pred + y_hat_values

    print("RMSE : ", int(np.sqrt(mean_squared_error(actual, pred))))
    
    plt.figure(figsize=(20,10))
    plt.plot(actual, label='actual')
    plt.plot(pred, label='forecast')
    plt.title(title) 
    plt.legend()
    if _ylim is not None:
        plt.ylim(_ylim)
    plt.show()
      
# --------------------------------------------------------------------    
def plotTrainingProgress(training_history, title, _ylim=None):
    """
    학습 성능(수렴)을 시각화 하는 프로그램
    """
    history_dict = training_history.history
    losses = history_dict["loss"]
    val_losses = history_dict["val_loss"]

    # 학습 결과 시각화
    plt.plot(losses, label='loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    if _ylim is not None:
        plt.ylim(_ylim)
    plt.title(title)
    plt.show()
    
# --------------------------------------------------------------------    
def to_supervisedDaily(data, nLookBackDays=2, nForecastDays=1):
    """
    하나의 학습 샘플에 하루치 데이터를 넣을지, 이틀지 데이터를 넣을지를 결정하는 것
    모든 데이터는 1시~24시 단위로 구분되며
    이렇게 구분된 데이터를 몇 일 간의 분량을 학습 데이터로 사용할지만 결정하는 것
    - nLookBackDays : 과거 몇일간의 데이터를 입력으로 할 것인지?
    - nForecastDays : 미래 몇일간의 데이터를 예측 할 것인지?
    """
    data = np.array(data)
    X, y = [], []
    for i in range(len(data)):
        if i+nLookBackDays+nForecastDays > len(data):
            # 데이터가 충분하지 않음
            break
            
        #dataCombined = np.concatenate((data[i],data[i+1]),axis=0)
        XDataCombined = np.concatenate((data[i:i+nLookBackDays]),axis=0)
        X.append(XDataCombined.reshape((len(XDataCombined),1)))
        """
        주의: y값 각각도 3차원으로 변형해야함
        """
        yDataCombined = np.concatenate((data[i+nLookBackDays:i+nLookBackDays+nForecastDays]),
                                       axis=0)
        y.append(yDataCombined.reshape((len(yDataCombined),1))) # 3차원 변형이 필요한 경우의 코드
    return np.array(X), np.array(y)
    
# --------------------------------------------------------------------  
def to_supervisedContinuousHours(data, nLookBackDays=2, nForecastDays=1):
    """
    하나의 학습 샘플에 하루치 데이터를 넣을지, 이틀지 데이터를 넣을지를 결정하는 것
    모든 데이터는 몇시부터 시작하는지에 관계 없이 24시간 동안의 연속적으로 구성됨
    이렇게 구분된 데이터를 몇 일 간의 분량을 학습 데이터로 사용할지만 결정하는 것
    - nLookBackDays : 과거 몇일간의 데이터를 입력으로 할 것인지?
    - nForecastDays : 미래 몇일간의 데이터를 예측 할 것인지?
    """
    hoursPerDay = 24
    
    data_continuous = np.array(data)
    data_continuous = np.concatenate(data_continuous)
    data_continuous = data_continuous.reshape(len(data_continuous),1)
    
    n_input = hoursPerDay * nLookBackDays
    n_out = hoursPerDay * nForecastDays
    
    X, y = [], []
    in_start = 0
    for _ in range(len(data_continuous)):
        in_end = in_start + n_input
        out_end = in_end + n_out
        if( out_end < len(data_continuous) ):
            x_input = data_continuous[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(x_input)
            
            y_value = data_continuous[in_end:out_end, 0]
            y_value = y_value.reshape((len(y_value), 1)) # 주의: y값 각각도 3차원으로 변형해야함
            y.append(y_value)
        in_start += 1
    return np.array(X), np.array(y)
    
# --------------------------------------------------------------------      
def split_dataset(data, howManyDaysToTrain, numItemsPerDay):
    """
    - split a univariate dataset into train/test sets
    - 데이터가 1시부터 시작하는지, 24시에서 끝나는지를 확인하기 
      => 엑셀 파일을 직접 보고 확인함 (프로그래밍으로 확인하면 더 좋은디...)
    """

    """
    - 데이터를 하루 단위로 나눌거야
    - 예를 들어 24개의 데이터를 하루 데이터로 그루핑하고 사용을 하는거지
    - 그리고, 하루 단위로 나눈 다시 데이터를 학습용, 테스트 용으로 다시 나눌거야
    - 이 형태 그대로 사용할지, 아니면 다른 형태로 사용할지는 나중에 다시 정할거야
    """
    train_end_index = howManyDaysToTrain * numItemsPerDay
    
    # split
    train, test = data[:train_end_index], data[train_end_index:]
    
    # restructure into windows of daily data
    train = np.array(np.split(train, len(train)/numItemsPerDay))
    test = np.array(np.split(test, len(test)/numItemsPerDay))
    
    return train, test
    
# --------------------------------------------------------------------          
def test_predict_mergedInput_mergedOutput_forcingMax_multiplemodels(model1, model2, model3, Xmerged, ymerged, title, _ylim=None):
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
        
        [y_hat1, y_hat_max1] = model1.predict([x_sample1, x_sample2])
        [y_hat2, y_hat_max2] = model2.predict([x_sample1, x_sample2])
        [y_hat3, y_hat_max3] = model3.predict([x_sample1, x_sample2])
            
        y_hat_values1 = y_hat1[0].reshape(len(y_hat1[0]),).tolist()
        y_hat_values2 = y_hat2[0].reshape(len(y_hat2[0]),).tolist()
        y_hat_values3 = y_hat3[0].reshape(len(y_hat3[0]),).tolist()
        
        y_hat_values = []
        for j in range(len(y_hat_values1)):
            y_hat_values.append(max(y_hat_values1[j],y_hat_values2[j],y_hat_values3[j]))

        if len(ts_pred) == 0:
            ts_pred = y_hat_values
        else:
            ts_pred = ts_pred + y_hat_values
            
        ts_max_pred.append(max(y_hat_max1,y_hat_max2,y_hat_max3))
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