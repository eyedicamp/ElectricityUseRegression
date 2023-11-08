import pandas as pd
import os 

file_name = './xlsx/한전 일별 전력 사용현황_16_22.xlsx'

output_filename = 'daily_elec.xlsx'

final_dataset=pd.DataFrame(columns=['사용량\n(kWh)', '최대수요\n(kW)', 'date'])

sheets = [i for i in range(68, 80)]

# 엑셀 파일 읽어서 판다스 데이터프레임에 저장
df = pd.read_excel(file_name, skiprows=[0,1,2,3,5], usecols=[9,10,13,14], sheet_name=sheets)

print(df)

result_df = pd.DataFrame()

month = 4
year = 2016
for i in range(68, 80):
    
    if month == 12:
        month = 1
        year += 1
    else:
        month += 1
    
    if month < 10:
        month_str = '0' + str(month)
    else:
        month_str = str(month)
    date = str(year) + '.' + month_str
    
    df1 = pd.concat([df[i]['일자.2'], df[i]['일자.3']])
    df2 = pd.concat([df[i]['최대수요(kW)'], df[i]['최대수요(kW).1']])

    df3 = pd.concat([df1, df2], axis=1)

    result_df = pd.concat([result_df, df3])
    
    print('page ok : ', i)





xlxs_dir = './result/result2.xlsx'

result_df.to_excel(xlxs_dir,sheet_name = 'Sheet1', na_rep = 'NaN', float_format = "%.2f", header = True, index = False, startrow = 0, startcol = 0,)

