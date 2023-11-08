from year.Yearly_regression import yearly_regression
from month.Monthly_regression import monthly_regression
from week.Weekly_regression import weekly_regression
from day.Daily_regression import daily_regression

print("----기간을 선택하세요----")
print("|       1. 년 단위      |")
print("|       2. 월 단위      |")
print("|       3. 주 단위      |")
print("|       4. 일 단위      |")
print("|       5. 시 단위      |")
print("-------------------------")

time_select = input("1~5 숫자로 분석할 기간을 입력해주세요 : ")

if time_select == "1":
    yearly_regression()
elif time_select == "2":
    monthly_regression()
elif time_select == "3":
    weekly_regression()
elif time_select == "4":
    daily_regression()
elif time_select == "5":
    print("미구현")