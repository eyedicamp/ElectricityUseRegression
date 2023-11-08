import pandas as pd

srcDirectory = '../4.1 일단위로 작업량 계산된 엑셀 파일/'


dstDirectory = '../5.1 병합된 엑셀 파일/'
dstFilename = '가공-도장-조립 취합_전체.xlsx'

src1Filename = '1_가공실적_전체_2016.xlsx'
src2AFilename = '2A_선행도장_전체_2016.xlsx'
src2BFilename = '2B_선행도장_전체_2016.xlsx'
src3Filename = '3_조립실적_전체_2016.xlsx'

src1df  = pd.read_excel(srcDirectory + src1Filename)
print(src1df.head())
#print(src1df.columns)

# Unnamed: 0 라는 이름의 열이 있는데, 그것을 제거하기
src1df = src1df.drop(src1df.columns[0], axis=1)
print(src1df.head())

src2Adf = pd.read_excel(srcDirectory + src2AFilename)
print(src2Adf.head())

# Unnamed: 0 라는 이름의 열 및 date 열을 제거하기
src2Adf = src2Adf.drop(src2Adf.columns[[0,1]], axis=1)
print(src2Adf.head())

src2Bdf = pd.read_excel(srcDirectory + src2BFilename)
print(src2Bdf.head())

# Unnamed: 0 라는 이름의 열 및 date 열을 제거하기
src2Bdf = src2Bdf.drop(src2Bdf.columns[[0,1]], axis=1)
print(src2Bdf.head())

src3df  = pd.read_excel(srcDirectory + src3Filename)
print(src3df.head())

# Unnamed: 0 라는 이름의 열 및 date 열을 제거하기
src3df = src3df.drop(src3df.columns[[0,1]], axis=1)
print(src3df.head())


for year in range(2017, 2024):
    src1Filename = '1_가공실적_전체_' + str(year) + '.xlsx'
    src2AFilename = '2A_선행도장_전체_' + str(year) + '.xlsx'
    src2BFilename = '2B_선행도장_전체_' + str(year) + '.xlsx'
    src3Filename = '3_조립실적_전체_' + str(year) + '.xlsx'

    temp_src1df = pd.read_excel(srcDirectory + src1Filename)
    temp_src1df = temp_src1df.drop(temp_src1df.columns[0], axis=1)
    src1df['1.가공>중량'] = src1df['1.가공>중량'] + temp_src1df['1.가공>중량']
    src1df['1.가공>절단장'] = src1df['1.가공>절단장'] + temp_src1df['1.가공>절단장']

    temp_src2Adf = pd.read_excel(srcDirectory + src2AFilename)
    temp_src2Adf = temp_src2Adf.drop(temp_src2Adf.columns[0], axis=1)
    src2Adf['2A.선행도장>표면처리'] = src2Adf['2A.선행도장>표면처리'] + temp_src2Adf['2A.선행도장>표면처리']

    temp_src2Bdf = pd.read_excel(srcDirectory + src2BFilename)
    temp_src2Bdf = temp_src2Bdf.drop(temp_src2Bdf.columns[0], axis=1)
    src2Bdf['2B.선행도장>도장'] = src2Bdf['2B.선행도장>도장'] + temp_src2Bdf['2B.선행도장>도장']
    
    temp_src3df = pd.read_excel(srcDirectory + src3Filename)
    temp_src3df = temp_src3df.drop(temp_src3df.columns[0], axis=1)
    src3df['3.조립>중량'] = src3df['3.조립>중량'] + temp_src3df['3.조립>중량']
    src3df['3.조립>용접장'] = src3df['3.조립>용접장'] + temp_src3df['3.조립>용접장']
    src3df['3.조립>용착량'] = src3df['3.조립>용착량'] + temp_src3df['3.조립>용착량']


dfCombined = pd.concat([src1df,src2Adf,src2Bdf,src3df], axis=1)
print(dfCombined.head())

if True:
    print("save to : ", dstDirectory + dstFilename)
    dfCombined.to_excel(dstDirectory + dstFilename)