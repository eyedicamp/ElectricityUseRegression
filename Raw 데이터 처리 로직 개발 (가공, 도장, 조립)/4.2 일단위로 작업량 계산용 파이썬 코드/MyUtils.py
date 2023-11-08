import pandas as pd

daysOfWeek = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
weekendList = ['Sat','Sun']
dtOneDayChange = pd.DateOffset(1)

def readExcelFile(src, _usecols=None):
    if _usecols is None:
        df = pd.read_excel(src)
    else:
        df = pd.read_excel(src, usecols=_usecols)
    # NaN 값을 0으로 채우기 # print(dfExcel.isna().sum())
    df = df.fillna(0)
    return df
    
def printDataFrameInfo(df):
    # 읽어들인 파일의 정보를 출력
    nRows = len(df)
    nCols = int(df.size / nRows)
    print("* 데이터 프레임 크기 : %d rows, %d cols"%(nRows, nCols))
    print("* 필드별 데이터 유형: ")
    print(df.dtypes)
    print("* 최초 5개 행 출력: ")
    print(df.head())    

def countDaysBetweenStringDate(strStartDate, strEndDate):
    # 두 날짜 사이의 날 수를 계산하기
    # 시작 및 종료날짜를 포함하여 처리함
    # 날 수 = 1(시작일) + 1(종료일) + N(그 사이 날)
    dt1 = pd.to_datetime(strStartDate, format='%Y-%m-%d')
    dt2 = pd.to_datetime(strEndDate, format='%Y-%m-%d')
    return countDaysBetweenDateTime(dt1, dt2)

def countDaysBetweenDateTime(dtStart, dtEnd):
    # 두 날짜 사이의 날 수를 계산하기
    # 시작 및 종료날짜를 포함하여 처리함
    # 날 수 = 1(시작일) + 1(종료일) + N(그 사이 날)
    if dtStart > dtEnd: # 잘못된 경우
        assert False
    return (dtEnd-dtStart).days + 1 # 하루를 더한다

def getDayOfWeek(strDate):
    # 날짜를 입력으로, 요일 구하기
    # 테스트; print(getDayOfWeek('2023-07-20'))
    dt = pd.to_datetime(strDate, format='%Y-%m-%d')
    return daysOfWeek[dt.dayofweek]

def isWeekendFromDatetime(date):
    # 주어진 날이 주말인지 아닌지를 판단
    if daysOfWeek[date.dayofweek] in weekendList:
        return True
    else:
        return False 
    
def isWeekendFromDateString(datestring):
    # 주어진 날이 주말인지 아닌지를 판단
    dt = pd.to_datetime(dateString, format='%Y-%m-%d')
    return isWeekendFromDatetime(dt)

def getWorkingDaysFromStringDates(strStartDate, strEndDate):
    # 두 날짜 사이의 business day (= working day) 계산하기
    # 시작 및 종료날짜를 포함하여 처리함
    # 즉, 주말(토,일)을 제외한, 근무일이 며칠인지를 계산
    dtFrom = pd.to_datetime(strStartDate, format='%Y-%m-%d')
    dtTo = pd.to_datetime(strEndDate, format='%Y-%m-%d')
    return getWorkingDaysFromDateTime(dtFrom,dtTo)

def getWorkingDaysFromDateTime(dtFrom, dtTo):
    # 두 날짜 사이의 business day (= working day) 계산하기
    # 시작 및 종료날짜를 포함하여 처리함 
    # 즉, 주말(토,일)을 제외한, 근무일이 며칠인지를 계산
    if dtFrom > dtTo:
        assert False
    total_count = countDaysBetweenDateTime(dtFrom, dtTo)
    
    dtNow = dtFrom
    while dtNow <= dtTo:
        if isWeekendFromDatetime(dtNow) == True:
            total_count -= 1
        dtNow += dtOneDayChange
            
    return total_count

def genEmptyDataFrame(dtFrom, dtTo, newCols, DEBUG_PRINT = True):
    """
    - 데이터 프레임을 생성하고 0으로 초기화
    - 단, date 라는 열을 만들때, 거기는 0이 아니라 yyyy-mm-dd 값으로 초기화하고
      기간은 stFrom에서 dtTo 까지로 설정함
    - 그리고, 특정 날짜가 주말이면 데이터프레임에 추가하지 않음
    """
    assert newCols[0] == 'date'
    nColsExceptDate = len(newCols) - 1 # 날짜 필드를 제외한 필드명의 수
    
    newdf = pd.DataFrame(columns = newCols)
    
    # 필요한 행의 수를 계산하고, 초기화된 데이터를 데이터프레임에 추가하기
    dtCurrDate = dtFrom
    while dtCurrDate <= dtTo:
        dateString = dtCurrDate.strftime('%Y-%m-%d') # 문자열 날짜 획득
        if isWeekendFromDatetime(dtCurrDate) == False: # 평일
            new_data = [dateString] # 첫 필드는 날짜 문자열 값을 입력으로...
            for iter in range(nColsExceptDate):
                # 나머지 필드 수 만큼 0으로 채워넣기 (초기화)
                new_data.append(0)
            newdf.loc[len(newdf)] = new_data
        else: # 주말
            if DEBUG_PRINT:
                print("* %s : 주말 (데이터프레임에 추가하지 않음)"%(dateString))
            else:
                pass

        dtCurrDate += dtOneDayChange # 하루를 증가
        
    return newdf

def getRowIndexFromDateTime(df, dt):
    """
    특정 날짜가, 데이트프레임에서 몇 번째 행에 해당하는지를 리턴
    """
    stringDate = dt.strftime('%Y-%m-%d')
    return getRowIndexFromDateString(df, stringDate)

def getRowIndexFromDateString(df, stringDate):
    """
    특정 날짜가, 데이트프레임에서 몇 번째 행에 해당하는지를 리턴
    """
    ret = df.index[df['date'] == stringDate].tolist()
    assert len(ret) == 1, '%s : len = %d'%(stringDate, len(ret))
    return ret[0]

def addDailyWorkload(df,
                     singleColumnName, # 하나의 필드에 대해서만 처리하는 코드임
                     dailyWorkloadAmount,
                     dtStart, dtEnd, # 시작 및 종료날짜를 포함하여 처리함
                     dtEntireStart, dtEntireEnd, # 전체 유효한 기간 정보
                     DEBUG_PRINT = True):
    """
    - 예를 들어, 원본 엑셀 파일에서 2020-01-01~2020-01-10까지 100만큼의 작업을 했다는
      기록이 있다고 하면, 우리는 이를 처리하기 위해
      1) 해당 기간 내에 (주말을 제외한) 작업일 수(N)를 계산
      2) 전체 작업량 100 / 작업일수(N) = dailyWorkloadAmount 를 계산
      3) 해당 기간 중 작업일에 대해서 dailyWorkloadAmount 를 더해줌
      을 수행하는데, addDailyWorkload 함수는 3)번 작업을 처리하는 함수이다.
    """    
    update_count = 0 # 디버깅 용으로 사용하는 변수(총 몇개의 날에 반영이 되었는지)
    dt = dtStart # 시작일자로 설정하고, 종료일자까지 모든 평일에 같은 작업량을 더함
    while dt <= dtEnd:
        # 평일이고, 유효한 기간 범위 내에 있으면 업데이트
        if (isWeekendFromDatetime(dt) == False) \
            and (dt >= dtEntireStart and dt <= dtEntireEnd):
            
            if DEBUG_PRINT:
                assert dt >= dtStart
                assert dt <= dtEnd
                print('calling getRowIndexFromDateTime for ', dt)
                
            row_index = getRowIndexFromDateTime(df, dt)
            if DEBUG_PRINT:
                print("Before : ", df.at[row_index, singleColumnName])
                
            df.at[row_index, singleColumnName] += dailyWorkloadAmount # 작업량 추가            
            if DEBUG_PRINT:
                print("After : ", df.at[row_index, singleColumnName])
            
            update_count += 1
        else:
            # 작업량을 더할 수 없는 경우, 그 원인을 설명
            if isWeekendFromDatetime(dt):
                if DEBUG_PRINT:
                    print("not updating for the date : ", dt , end=" for reason : weekend\n")
                else:
                    pass
            elif dt > dtEntireEnd:
                if DEBUG_PRINT:
                    print("not updating for the date : ", dt , end=" for reason : future data\n")
                else:
                    pass
            elif dt < dtEntireStart:
                if DEBUG_PRINT:
                    print("not updating for the date : ", dt , end=" for reason : past data\n")
                else:
                    pass
            else:
                assert False, "unknown reason"
        dt += dtOneDayChange
        
    if update_count == 0:
        # 모든 기간이 휴일이거나,,, 등등의 경우에 이런 상황이 발생 가능
        if DEBUG_PRINT:
            print("작업량이 반영되지 않습니다")
        else:
            pass
        
    return df