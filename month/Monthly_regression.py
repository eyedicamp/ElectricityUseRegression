import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def monthly_regression():
    print('다음달 예정 작업량을 실수로 입력해주세요.')
    gagong_weight = float(input('가공 > 중량 : '))
    gagong_length = float(input('가공 > 절단장 : '))
    dojang_pyomeyun = float(input('선행도장 > 표면처리 : '))
    dojang_dojang = float(input('선행도장 > 도장 : '))
    jorip_weight = float(input('조립 > 중량 : '))
    jorip_length = float(input('조립 > 용접장 : '))
    jorip_yongchack = float(input('조립 > 용착량 : '))
    
    model_xgb_peak = xgb.Booster()
    model_xgb_peak.load_model("./month/Peak_monthly_model.txt")

    model_xgb_use = xgb.Booster()
    model_xgb_use.load_model("./month/Use_monthly_model.txt")

    xtest = [[gagong_weight, gagong_length, dojang_pyomeyun, dojang_dojang, jorip_weight, jorip_length, jorip_yongchack]]
    dm_xtest = xgb.DMatrix(xtest)

    ypred_use = model_xgb_use.predict(dm_xtest)
    print('사용량(kWh) :', ypred_use[0])

    ypred_peak = model_xgb_peak.predict(dm_xtest)
    print('최대수요(kW) :', ypred_peak[0])

# 24723.36166563704
# 225436.85803112213
# 529411.9779059449
# 4561639.032918035
# 47656.60404844131
# 247618.36407040557
# 30886.657370067416

# 37804804.800000004
# 121296.0