---
layout: single
title:  "미니프로젝트 - 네이버 증권 페이지 스크래핑"
---
# 라이브러리 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import koreanize_matplotlib
from statsmodels.formula.api import ols
import warnings 
warnings.filterwarnings("ignore")
import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
# 그래프가 노트북에 표시되지 않을 때
from plotly.offline import iplot, init_notebook_mode
from plotly.subplots import make_subplots
init_notebook_mode()
# 그래프에 retina display 적용
%config InlineBackend.figure_format = 'retina'
# 노트북 안에서 그래프를 디스플레이 하겠다는 설정입니다.
%matplotlib inline
```

# 데이터셋 가져오기

- Github에서 raw데이터를 가져오는 방식으로 진행 

## 도입부 데이터셋


```python
# life_expectancy : 기대수명 데이터
# aging_index : 노령화지수 데이터
# sel_est_pop : 서울시 추계 인구 데이터
# url_health : 노인 건강 만족도 데이터
# url_age : 노인 연령 인지 데이터
life_expectancy = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8_%E1%84%80%E1%85%B5%E1%84%83%E1%85%A2%E1%84%89%E1%85%AE%E1%84%86%E1%85%A7%E1%86%BC.csv'
aging_index = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%8C%E1%85%A5%E1%86%AB%E1%84%80%E1%85%AE%E1%86%A8_%E1%84%82%E1%85%A9%E1%84%85%E1%85%A7%E1%86%BC%E1%84%92%E1%85%AA%E1%84%8C%E1%85%B5%E1%84%89%E1%85%AE.csv'
sel_est_pop = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%89%E1%85%B5_%E1%84%8E%E1%85%AE%E1%84%80%E1%85%A8%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE.csv'
url_health = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EB%85%B8%EC%9D%B8%EC%9D%98_%EA%B1%B4%EA%B0%95%EC%83%81%ED%83%9C_%EB%A7%8C%EC%A1%B1%EB%8F%84.csv'
url_age = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EB%85%B8%EC%9D%B8%EC%9D%98_%EB%85%B8%EC%9D%B8%EC%97%B0%EB%A0%B9%EC%97%90_%EB%8C%80%ED%95%9C_%EC%9D%B8%EC%A7%80.csv'
```

## 지하철 운영적자와 노인의 관계

### 지하철 운영 적자와 고령 인구 수의 증가, 정말 연관이 있을까?


```python
# sel_est_pop : 서울시 추계 인구 데이터
# sel_fr_cost : 무임비용 데이터
# sel_sub_opm : 영업손익 데이터
# sel_sub_expenses : 경비 데이터
# sel_sub_col : 인건비 데이터
sel_est_pop = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%89%E1%85%B5_%E1%84%8E%E1%85%AE%E1%84%80%E1%85%A8%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE.csv'
sel_fr_cost = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%8C%E1%85%B5%E1%84%92%E1%85%A1%E1%84%8E%E1%85%A5%E1%86%AF_%E1%84%86%E1%85%AE%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%87%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC.csv'
sel_sub_opm = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%8C%E1%85%B5%E1%84%92%E1%85%A1%E1%84%8E%E1%85%A5%E1%86%AF_%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A5%E1%86%B8%E1%84%89%E1%85%A9%E1%86%AB%E1%84%8B%E1%85%B5%E1%86%A8.csv'
sel_sub_expenses = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%8C%E1%85%B5%E1%84%92%E1%85%A1%E1%84%8E%E1%85%A5%E1%86%AF_%E1%84%80%E1%85%A7%E1%86%BC%E1%84%87%E1%85%B5.csv'
sel_sub_col = 'https://raw.githubusercontent.com/BDDKID/AIS8/main/used_csv/%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%8C%E1%85%B5%E1%84%92%E1%85%A1%E1%84%8E%E1%85%A5%E1%86%AF_%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A5%E1%86%AB%E1%84%87%E1%85%B5.csv'
```

### 고령층 무임수송에 따른 손실이 17~19년에 비해 20년 되레 줄었다… 왜?


```python
# fr_sub: 무임승차 대상별 현황 데이터
# fr_cost : 무임승차 및 무임비용 데이터
# fr_finstate : 운영기관별 손익계산서 데이터
sub_url = "https://raw.githubusercontent.com/gogo-yubari/AI-S8/main/%E1%84%86%E1%85%AE%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%89%E1%85%B3%E1%86%BC%E1%84%8E%E1%85%A1_%E1%84%83%E1%85%A2%E1%84%89%E1%85%A1%E1%86%BC%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%92%E1%85%A7%E1%86%AB%E1%84%92%E1%85%AA%E1%86%BC_20230221145752.csv"
cost_url = "https://raw.githubusercontent.com/gogo-yubari/AI-S8/main/%E1%84%86%E1%85%AE%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%89%E1%85%B3%E1%86%BC%E1%84%8E%E1%85%A1_%E1%84%86%E1%85%B5%E1%86%BE_%E1%84%86%E1%85%AE%E1%84%8B%E1%85%B5%E1%86%B7%E1%84%87%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC_20230221165228.csv"
finstate_url = "https://raw.githubusercontent.com/gogo-yubari/AI-S8/main/%E1%84%8B%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A7%E1%86%BC%E1%84%80%E1%85%B5%E1%84%80%E1%85%AA%E1%86%AB%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%89%E1%85%A9%E1%86%AB%E1%84%8B%E1%85%B5%E1%86%A8%E1%84%80%E1%85%A8%E1%84%89%E1%85%A1%E1%86%AB%E1%84%89%E1%85%A5_20230221145639.csv"
```

## 노인 연령을 70세로 상향한다면 정말 지하철 적자를 유의미하게 메꿀 수 있을까?

### 노인 연령을 70세로 상향 시 발생하는 이익 계산


```python
# url_pop : 서울시 연도별 연령대별 인구 데이터
# url_pnl : 지하철 운영기관별 손익계산서 데이터
# url_nomoney : 무임승차 대상별 현황 데이터 불러오기
# url_deficit : 승객 1인당 운임손실 현황 데이터 불러오기
# url_carry :승차 및 수송인원 데이터 불러오기
url_pop = "https://raw.githubusercontent.com/everyshayday/AS8/main/seoul_age_population.csv"
url_pnl = "https://raw.githubusercontent.com/everyshayday/AS8/main/%EC%9A%B4%EC%98%81%EA%B8%B0%EA%B4%80%EB%B3%84_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C.csv"
url_nomoney = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EB%AC%B4%EC%9E%84%EC%8A%B9%EC%B0%A8_%EB%8C%80%EC%83%81%EB%B3%84_%ED%98%84%ED%99%A9.csv'
url_deficit = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EC%8A%B9%EA%B0%9D_1%EC%9D%B8%EB%8B%B9_%EC%9A%B4%EC%9E%84%EC%86%90%EC%8B%A4_%ED%98%84%ED%99%A9.csv'
url_carry = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EC%8A%B9%EC%B0%A8_%EB%B0%8F_%EC%88%98%EC%86%A1%EC%9D%B8%EC%9B%90.csv'
```

### 향후 5년간(22-26년) 무임승차 연령을 상향한다면 얼마나 수익이 생길까? 과연 유의미할까?


```python
# url_cpi : 물가상승지수 데이터
# url_pop : 노인 인구수 변화 데이터 
# url : 인구수 데이터
# url_carry : 승차인원 데이터
url_cpi = 'https://raw.githubusercontent.com/JounKK/AIS8_task/main/mid_%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%83%E1%85%A9%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%8C%E1%85%B5%E1%84%8E%E1%85%AE%E1%86%AF%E1%84%86%E1%85%A9%E1%86%A8%E1%84%8C%E1%85%A5%E1%86%A8%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%89%E1%85%A9%E1%84%87%E1%85%B5%E1%84%8C%E1%85%A1%E1%84%86%E1%85%AE%E1%86%AF%E1%84%80%E1%85%A1%E1%84%8C%E1%85%B5%E1%84%89%E1%85%AE_%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%90%E1%85%B3%E1%86%A8%E1%84%87%E1%85%A7%E1%86%AF%E1%84%89%E1%85%B5.csv'
url_pop1 = 'https://raw.githubusercontent.com/JounKK/AIS8_task/main/mid_%E1%84%89%E1%85%A5%E1%86%BC_%E1%84%86%E1%85%B5%E1%86%BE_%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%85%E1%85%A7%E1%86%BC%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%8E%E1%85%AE%E1%84%80%E1%85%A8%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE_%E1%84%89%E1%85%A5%E1%84%8B%E1%85%AE%E1%86%AF%E1%84%89%E1%85%B5.csv'
url = "https://raw.githubusercontent.com/JounKK/AIS8_task/main/mid_%E1%84%89%E1%85%A5%E1%86%BC_%E1%84%86%E1%85%B5%E1%86%BE_%E1%84%8B%E1%85%A7%E1%86%AB%E1%84%85%E1%85%A7%E1%86%BC%E1%84%87%E1%85%A7%E1%86%AF_%E1%84%8E%E1%85%AE%E1%84%80%E1%85%A8%E1%84%8B%E1%85%B5%E1%86%AB%E1%84%80%E1%85%AE_1722.csv"
url_carry1 = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EC%8A%B9%EC%B0%A8_%EB%B0%8F_%EC%88%98%EC%86%A1%EC%9D%B8%EC%9B%90.csv'
url_nomoney = 'https://raw.githubusercontent.com/everyshayday/AS8/main/%EB%AC%B4%EC%9E%84%EC%8A%B9%EC%B0%A8_%EB%8C%80%EC%83%81%EB%B3%84_%ED%98%84%ED%99%A9.csv'
```

## 지하철 운영의 실질적인 문제

### 지하철 운영 적자의 진짜 원인은?


```python
# url_2017 ~ 2021 : 서울 교통공사 손익계산서 데이터
url_2021 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2021.csv"
url_2020 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2020.csv"
url_2019 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2019.csv"
url_2018 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2018.csv"
url_2017 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2017.csv"
url_cal = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%AC%B4%EC%9E%84%EC%8A%B9%EC%B0%A8_%EB%8C%80%EC%83%81%EB%B3%84_%ED%98%84%ED%99%A9_20230222141349.csv"
```

## 노인 무임승차의 긍정적인 측면

### 노인 무임승차 연령 상향의 불이익


```python
# url_sub : 지역별 65세 인구수
# url_sui : 자살을 생각하는 이유 데이터
# url_suicide : 지역별 실제 자살자 수 데이터
# url_dep : 우울 증세 경험률 데이터
url_sub = 'https://raw.githubusercontent.com/Bae-Sangbin/AISCHOOL8/main/subway_population.csv'
url_sui = 'https://raw.githubusercontent.com/Bae-Sangbin/AISCHOOL8/main/suicide.csv'
url_suicide = 'https://raw.githubusercontent.com/Bae-Sangbin/AISCHOOL8/main/suicide_population.csv'
url_dep = 'https://raw.githubusercontent.com/Bae-Sangbin/AISCHOOL8/main/depression.csv'
```

# 도입부 파트
- 고령인구와 4세이하 인구 변화추이
- 기대수명과 노령화 지수 추이
- 연도별 노인 건강 만족도 추이
- 노인연령에 대한 인지(2020년)

## 고령인구와 4세이하 인구 변화추이



```python
# 서울인구 추계데이터 불러오기
kid_elder = pd.read_csv(sel_est_pop,encoding='cp949')
kid_elder = kid_elder.drop(columns=['시나리오별(1)','성별(1)','시도별(1)'])

# 고령인구와 유아인구로 데이터셋 분리 하기  
elder = kid_elder.drop(columns='0 - 4세')
kid = kid_elder.drop(columns=['65 - 69세','70 - 74세','75 - 79세','80세이상'])
```


```python
# 고령인구 데이터프레임 데이터 형식과 컬럼 변경하기
# 65세 이상 인구 합계를 알기위해 새로운 컬럼 생성(65세이상 인구 합계)
elder = elder.astype('int64')
elder['시점'] = elder['시점'].astype('string')
elder['합계'] = elder.sum(axis=1)
elder = elder.drop(columns=['65 - 69세','70 - 74세','75 - 79세','80세이상'])
elder.columns = ['연도','65세이상 인구 합계']
elder
```


```python
# 유아인구 데이터 프레임 데이터 형식과 컬럼 변경하기
kid.columns = ['연도', '4세 이하 인구 합계']
kid['연도'] = kid['연도'].astype('string')
```


```python
# 분리했던 고령인구와 유아인구 데이터프레임을 연도를 기준으로 병합
total_kid_elder = pd.merge(kid,elder,how='outer',on='연도')
total_kid_elder['연도'] =total_kid_elder['연도'].astype('int64')
total_kid_elder
```


```python
# 고령인구와 4세 이하 인구 추이 시각화 하기
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=total_kid_elder['연도'], y=total_kid_elder['4세 이하 인구 합계'], name="4세 이하 인구 합계"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=total_kid_elder['연도'],y=total_kid_elder['65세이상 인구 합계'], name="65세이상 인구 합계"),
    secondary_y=True,
)

fig.update_layout(
    title_text="고령인구와 4세 이하 인구 변화 추이",
    width=900,
    height=500
)

fig.update_xaxes(title_text="연도")

fig.update_yaxes(title_text="<b>4세 이하</b> 인구 합계", secondary_y=False)
fig.update_yaxes(title_text="<b>65세이상</b> 인구 합계", secondary_y=True)

fig.show()
```

## 기대수명과 노령화 지수 추이



```python
# 기대수명과 노령화 지수 데이터셋 불러오기
life_expect = pd.read_csv(life_expectancy, encoding='cp949')
aging_idx = pd.read_csv(aging_index, encoding='cp949')
```


```python
# 기대수명 데이터프레임 컬럼 변경
life_expect = life_expect.drop(columns='가정별')
life_expect.columns = ['연도','기대수명']
life_expect
```


```python
# 노령화 지수 데이터 프레임 컬럼 변경
aging_idx = aging_idx.drop(columns='가정별')
aging_idx.columns = ['연도','노령화지수']
```


```python
exlife_ai = pd.merge(aging_idx,life_expect,how='outer',on='연도')
exlife_ai
```


```python
# 기대수명과 노령화지수 추이 시각화 하기
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=exlife_ai['연도'], y=exlife_ai['기대수명'], name="기대수명"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=exlife_ai['연도'],y=exlife_ai['노령화지수'], name="노령화지수"),
    secondary_y=True,
)

fig.update_layout(
    title_text="기대수명과 노령화지수의 추이",
    width=900,
    height=500
)

fig.update_xaxes(title_text="연도")

fig.update_yaxes(title_text="<b>기대수명</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>노령화지수</b>", secondary_y=True)

fig.show()
```

## 연도별 노인 건강 만족도 추이



```python
# 노인 건강 만족도 데이터셋 불러오기
hlt = pd.read_csv(url_health, encoding='cp949')
hlt.head()
```


```python
# 노인 건강 만족도 데이터의 구조와 형식 확인하기
hlt.shape
hlt.info()
```


```python
# 불필요한 컬럼 제외
hlt = hlt.drop(columns=["특성별(1)"])
```


```python
# 만족한다고 응답한 항목만 합계
hlt["만족 응답 합계(%)"] = hlt["매우 만족 (%)"] + hlt["만족 (%)"]
hlt.head(2)
```


```python
# 컬럼명 변경

hlt = hlt.rename(columns={"시점":"연도", "특성별(2)":"연령대"})
hlt.head()
```


```python
# 노인 건강 만족도 추이 시각화
px.line(hlt, x="연도", y="만족 응답 합계(%)", color="연령대", height=500, width=800,
       title="연도별 노인 건강 만족도 추이", markers=True)
```

## 노인연령에 대한 인지(2020년)


```python
age = pd.read_csv(url_age, encoding='cp949')
age.head()
```


```python
# 데이터의 구조와 '시점' 컬럼의 요소 확인하기
age.shape
age["시점"].unique()
```


```python
#가장 최근 자료만 가져오기

age_2020 = age[age["시점"]==2020]
age_2020
```


```python
# 필요한 컬럼만 가져오기
age_2020_2 = age_2020.iloc[:, [0, 2, 7, 8, 9, 10]]
age_2020_2
```


```python
# 컬럼명 변경
age_2020_2 = age_2020_2.rename(columns={"시점":"연도", "특성별(2)":"연령대"})
age_2020_2
```


```python
# 데이터 타입 확인하기
age_2020_2.dtypes
```


```python
# float형식으로 변환
age_2020_2["69세 이하 (%)"] = age_2020_2["69세 이하 (%)"].astype("float")
```


```python
# 각 항목의 평균 구하기

age_df = age_2020_2.mean().to_frame()
age_t = age_df.T
```


```python
# melt를 이용하여 데이터 프레임 구조 변환
age_t = age_t.melt(id_vars=['연도'], var_name='항목', value_name='응답비율')
age_t
```


```python
# 2020년 노인연령에 대한 인지 시각화
px.bar(age_t, x='항목', y='응답비율', height=500, width=800,
       title="노인연령에 대한 인지 (2020년)", labels={"항목":"노인이라고 인지하는 연령"})
```

# 지하철 운영적자와 노인의 관계

## 지하철 운영 적자와 고령 인구 수의 증가, 정말 연관이 있을까?

### 연도별 지하철 운영 적자 추이와 고령인구의 상관관계 분석


```python
# 서울시 무임비용 데이터 불러오기
# 필요없는 행과 열 삭제하기
fr_cost= pd.read_csv(sel_fr_cost,encoding = 'cp949')
fr_cost = fr_cost.drop(columns='대상별(1)')
fr_cost = fr_cost.loc[1:]
fr_cost
```


```python
# 무임비용 데이터 프레임 컬럼명과 데이터 형식 변환
col1  = ['연도', '서울교통공사(무임비용(백만원))', '서울메트로 9호선(주)(무임비용(백만원))',
 '서울교통공사9호선운영부문(무임비용(백만원))', '우이 신설경전철(주)(무임비용(백만원))']
fr_cost.columns = col1
fr_cost

fr_cost[['서울교통공사(무임비용(백만원))', '서울메트로 9호선(주)(무임비용(백만원))',
 '서울교통공사9호선운영부문(무임비용(백만원))', '우이 신설경전철(주)(무임비용(백만원))']]=fr_cost[['서울교통공사(무임비용(백만원))',
'서울메트로 9호선(주)(무임비용(백만원))','서울교통공사9호선운영부문(무임비용(백만원))', '우이 신설경전철(주)(무임비용(백만원))']].astype('int64')
fr_cost['연도'] = pd.to_datetime(fr_cost['연도'])
fr_cost
```


```python
# 무임비용 합계를 구하기 위해 새로운 컬럼 생성
# 인덱스를 초기화 하고 필요없는 컬럼 삭제
fr_cost['전체무임비용(백만원)'] =fr_cost.sum(axis = 1)
fr_cost.reset_index(inplace=True)
fr_cost = fr_cost.drop(columns=['index'])
fr_cost
```


```python
# 영업손익 데이터 불러오기
# 필요없는 행 삭제하기
opm = pd.read_csv(sel_sub_opm, encoding='cp949')
opm = opm.loc[1:]
opm = opm.astype('int64')
```


```python
opm['시점'] = ['2017-12-31','2018-12-31','2019-12-31','2020-12-31','2021-12-31']
opm['시점'] = pd.to_datetime(opm['시점'])
opm['합계(억원)'] = opm.sum(axis=1)
opm['손실 합계(억원)'] = abs(opm['합계(억원)'])
```


```python
# 인덱스 초기화 및 필요없는 컬럼 삭제
opm.reset_index(inplace=True)
opm = opm.drop(columns=['index','서울교통공사','서울메트로 9호선㈜','서울교통공사9호선운영부문','우이 신설경전철㈜','합계(억원)'])
opm
```


```python
# 영업비용 : 경비 데이터 불러오기(expenses : 경비)
expenses = pd.read_csv(sel_sub_expenses,encoding='cp949')
expenses = expenses.loc[2:]
```


```python
# 경비데이터셋 데이터 형식 변경
expenses = expenses.astype('int64')
expenses['시점'] = expenses['시점'].astype('string')

```


```python
# 전체 경비를 구하기 위해 새로운 컬럼 생성
# 인덱스 초기화 및 필요 없는 컬럼 삭제
expenses['전체 경비(억원)'] = expenses.sum(axis=1)
expenses.reset_index(inplace=True)
expenses = expenses.drop(columns=['index','서울교통공사','서울메트로 9호선㈜','서울교통공사9호선운영부문','우이 신설경전철㈜'])

expenses
```


```python
# 영업비용 : 인건비 데이터 불러오기(col : cost of labor)
# 필요없는 행 삭제
col = pd.read_csv(sel_sub_col,encoding='cp949')
col = col.loc[2:]
```


```python
# 데이터프레임 형식 변환
col = col.astype('int64')
col['시점'] = col['시점'].astype('string')
```


```python
# 전체 인건비를 구하기 위한 새로운 컬럼생성
# 인덱스 초기화 및 필요 없는 컬럼 삭제

col['전체 인건비(억원)'] = col.sum(axis=1)
col.reset_index(inplace=True)
col = col.drop(columns=['index','서울교통공사','서울메트로 9호선㈜','서울교통공사9호선운영부문','우이 신설경전철㈜'])

col
```


```python
# 상관꼐수를 구하기위해 데이터를 병합(손실합계, 65세 인구 합계, 전체무임비용, 전체 경비, 전체 인건비)
year = col['시점']
total_opm = opm['손실 합계(억원)']
total_fr_old = total_kid_elder['65세이상 인구 합계'].loc[:4]
total_fr_cost = fr_cost['전체무임비용(백만원)']
total_expenses = expenses['전체 경비(억원)']
total_col = col['전체 인건비(억원)']

total = pd.concat([year,total_opm, total_fr_old,total_fr_cost,total_expenses,total_col], axis=1)
total
```


```python
# 단위를 맞춰주기 위해 컬럼명과 요소 변경
total.columns = ['연도','손실 합계(백만원)','65세이상 인구 합계','전체 무임비용(백만원)',
              '전체 경비(백만원)','전체 인건비(백만원)']

total['손실 합계(백만원)'] = [551100,569400,
                   563800,1141400,989600]
total['전체 무임비용(백만원)'] = [298115,308084,
                     308084,328873,250677]
total['전체 경비(백만원)'] = [1347100,1412300,1434800,
                        1434200,1487500]
total['전체 인건비(백만원)'] = [1300500,1286200,1292600,
                            1405700,1281500]
```


```python
total.corr()
```


```python
# 손십할계에 대한 각각의 독립변수에 대해 상관관계 구하기
import pingouin as pg
```


```python
# 손십할계와 65세이상 인구 합계와의 상관관계
# 귀무가설 기각할 수 없음.
pg.corr(total['손실 합계(백만원)'], total['65세이상 인구 합계'])
```


```python
# 손십할계와 전체 무임비용간의 상관관계
# 귀무가설 기각할 수 없음.
pg.corr(total['손실 합계(백만원)'], total['전체 무임비용(백만원)'])
```


```python
# 손십할계와 전체 경비와의 상관관계
# 귀무가설 기각할 수 없음.

pg.corr(total['손실 합계(백만원)'], total['전체 경비(백만원)'])
```


```python
# 손십할계와 전체 인건비와의 상관관계
# 귀무가설 기각할 수 없음.
pg.corr(total['손실 합계(백만원)'], total['전체 인건비(백만원)'])
```

### 손실합계에 대한 독립변수들의 영향을 보기 위한 시각화


```python
fig = make_subplots(rows=2, cols=2,
                    specs=[[{"secondary_y": True}, {"secondary_y": True}],
                           [{"secondary_y": True}, {"secondary_y": True}]],
                   subplot_titles=("손실합계와 노인인구추이 ", "손실합계와 전체 무임비용",
                                   "손실합계와 전체 경비", "손실합계와 전체 인건비"))

# Top left
fig.add_trace(
    go.Scatter(x=total['연도'], y=total['손실 합계(백만원)'], name="손실합계(백만원)"),
    row=1, col=1, secondary_y=False)

fig.add_trace(
    go.Scatter(x=total['연도'], y=total['65세이상 인구 합계'], name="65세이상 인구 합계"),
    row=1, col=1, secondary_y=True,
)

# Top right
fig.add_trace(
    go.Scatter(x=total['연도'], y=total['손실 합계(백만원)'], name="손실합계(백만원)"),
    row=1, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=total['연도'], y=total['전체 무임비용(백만원)'], name="전체 무임비용(백만원)"),
    row=1, col=2, secondary_y=True,
)

# Bottom left
fig.add_trace(
    go.Scatter(x=total['연도'], y=total['손실 합계(백만원)'], name="손실합계(백만원)"),
    row=2, col=1, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=total['연도'], y=total['전체 경비(백만원)'], name="전체 경비(백만원)"),
    row=2, col=1, secondary_y=True,
)

# Bottom right
fig.add_trace(
    go.Scatter(x=total['연도'], y=total['손실 합계(백만원)'], name="손실합계(백만원)"),
    row=2, col=2, secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=total['연도'], y=total['전체 인건비(백만원)'], name="전체 인건비(백만원)"),
    row=2, col=2, secondary_y=True,
)

fig.update_layout(
    title_text="전체 손실액과 독립변수들간의 변화 추이",
    width=1200,
    height=800
)
fig.show()
```

## 고령층 무임수송에 따른 손실이 17~19년에 비해 20년 되레 줄었다. 왜?

### 데이터 전처리 수행


```python
fr_sub = pd.read_csv(sub_url, encoding="cp949")
display(fr_sub.head(10))

```


```python
fr_sub = fr_sub.replace('-', 0)
display(fr_sub.info())

```


```python
fr_sub.iloc[:,3:] = fr_sub.iloc[:,3:].apply(pd.to_numeric)
fr_sub = fr_sub.drop(columns=['합계'])
fr_sub.head(10)
```


```python
fr_sub['합계'] = fr_sub.iloc[:,3:].sum(axis=1)
display(fr_sub.head(10))
```


```python
# 회사별 컬럼, 대상별에서 '계' 삭제
fr_sub_t = fr_sub.drop(fr_sub.columns[3:7], axis = 1)
fr_sub_t = fr_sub_t[~fr_sub_t["대상별(1)"].str.contains("계")]
fr_sub_t.rename(columns = {"대상별(1)" : "대상"}, inplace = True)
display(fr_sub_t.head())
```


```python
# 원하는 컬럼값을 가져오는 함수
# df_name : 데이터 프레임 이름
# col_name : var가 포함된 컬럼
# var : 기준 단어
# col_name2 : 값을 추출할 컬럼
def sort_df(df_name, col_name, var, col_name2):
    cols = df_name[df_name[col_name].str.contains(var)].loc[:,col_name2].reset_index(drop=True)
    return cols

# 함수 동작 테스트
# fr_sub_t 프레임의 '항목'컬럼이 승차라는 단어를 포함한 '합계'컬럼의 값
sort_df(fr_sub_t,"항목","승차","합계")
#sort_df(fr_sub_t,"항목","만원","합계")
```


```python
# 인원과 비용으로 분리 후 이름변경
# 무임비용 단위 '억원'으로 통일
# fr_sub4 과 동일
fr_sub_all = pd.DataFrame({"연도":sort_df(fr_sub_t,"항목","승차","시점"),
                 "대상": sort_df(fr_sub_t,"항목","승차","대상"),
                "무임승차(천명)":sort_df(fr_sub_t,"항목","승차","합계"),
                "무임비용(억원)":sort_df(fr_sub_t,"항목","만원","합계").div(100)})

fr_sub_all
```


```python
fr_cost = pd.read_csv(cost_url, encoding="cp949")
display(fr_cost.info())

```


```python
fr_cost.iloc[:,2:] = fr_cost.iloc[:,2:].apply(pd.to_numeric)
fr_cost.iloc[:,2:] = fr_cost.iloc[:,2:].astype('int')

```


```python
fr_cost['합계'] = fr_cost.iloc[:,2:].sum(axis=1)
display(fr_cost.head(10))

```


```python
# 회사 지우고 컬럼 이름 변경
fr_cost_t = fr_cost.drop(fr_cost.columns[2:6], axis = 1)
display(fr_cost_t.head())
```


```python
fr_cost_all = pd.DataFrame({"연도":sort_df(fr_cost_t,"항목","연간","시점"),
              "연간승차(천명)": sort_df(fr_cost_t,"항목","연간","합계"),
              "무임승차(천명)": sort_df(fr_cost_t,"항목","무임승차","합계"),
              "영업손실(억원)":sort_df(fr_cost_t,"항목","영업","합계"),
              "무임비용(억원)":sort_df(fr_cost_t,"항목","비용","합계")})

fr_cost_all
```


```python
fr_finstate = pd.read_csv(finstate_url, encoding="cp949")
display(fr_finstate.head(10))
```


```python
fr_finstate = fr_finstate.replace('-', 0)
display(fr_finstate.info())
```


```python
fr_finstate.iloc[:,3:] = fr_finstate.iloc[:,3:].apply(pd.to_numeric)
fr_finstate['합계'] = fr_finstate.iloc[:,3:].sum(axis=1)
display(fr_finstate.head(10))
# 회사 지우기
fr_finstate_t = fr_finstate.drop(fr_finstate.columns[3:7], axis =1)
```


```python
# 회사 지우기
fr_finstate_t = fr_finstate.drop(fr_finstate.columns[3:7], axis =1)
```


```python
fr_finstate_t.head(10)

```


```python
# 손익계산서의 손실액이 더 정확함
sort_df(fr_finstate_t,"손익계정별(1)","영업손익","합계").mul(-1)
```


```python
# 새로운 데이터 프레임 만들기
display(fr_sub_all.head())
display(fr_cost_all.head())

```


```python
df = []
df = pd.DataFrame({"연도":sort_df(fr_cost_t,"항목","연간","시점"),
                   "연간승차(천명)":sort_df(fr_cost_t,"항목","연간","합계"),
                   "무임승차(천명)":sort_df(fr_cost_t,"항목","무임승차","합계"),
                   "영업손실(억원)":sort_df(fr_finstate_t,"손익계정별(1)","영업손익","합계").mul(-1),
                   "무임비용(억원)":sort_df(fr_cost_t,"항목","비용","합계")})
df["연간승차 대비 무임승차"] = (df["무임승차(천명)"]/df["연간승차(천명)"])*100
df["손실액 대비 무임비용"] = (df["무임비용(억원)"]/df["영업손실(억원)"])*100
df

```


```python
display(fr_sub_t.head())
display(fr_sub_all.head())
display(fr_cost_all.head())
```


```python
# 노인 무임승차표
df_senior = []
df_senior = pd.DataFrame({"연도":sort_df(fr_cost_t,"항목","연간","시점"),
                          "연간승차(천명)":sort_df(fr_cost_t,"항목","연간","합계"),
                          "노인무임승차(천명)":sort_df(fr_sub_all,"대상","노인","무임승차(천명)"),
                          "영업손실(억원)":sort_df(fr_finstate_t,"손익계정별(1)","영업손익","합계").mul(-1),
                          "노인무임비용(억원)":sort_df(fr_sub_all,"대상","노인","무임비용(억원)")
                          })
df_senior["노인무임승차비율"] = df_senior["노인무임승차(천명)"]/df_senior["연간승차(천명)"]*100
df_senior["노인무임비용비율"] = df_senior["노인무임비용(억원)"]/df_senior["영업손실(억원)"]*100
df_senior

```


```python
# 대상별 무임승차인원과 비용 비율 
df_sub_mean = fr_sub_all.groupby('대상').mean().sort_values(by='무임승차(천명)', ascending=False)
df_sub_mean 
```


```python
# 왜 해놨는지 기억이 안나요
fr_sub_all[fr_sub_all["대상"].str.contains("노인")].iloc[:,2].reset_index(drop=True)
```


```python
# 증감율 테이블 
df_senior_pct = df_senior.pct_change().fillna(0).mul(100).round(2)
df_senior_pct.columns = ["연도","승차인원증감율","노인무임승차인원증감율","영업손실증감율","노인무임비용증감율","노인무임승차비율증감율","손실액대비노인무임비용증감율"]
df_senior_pct["연도"] = [2017, 2018, 2019, 2020, 2021]
df_senior_pct
```


```python
# 연도별 무임승차인원수 대비 노인무임승차자수 비율
fr_ratio = df_senior["노인무임승차(천명)"]/ df["무임승차(천명)"]*100
fr_ratio
```


```python
a = pd.DataFrame({"연도" :df["연도"],
                  "연간승차(천명)":df["연간승차(천명)"],
                  "무임승차(천명)":df["무임승차(천명)"],
                  "노인무임승차(천명)":df_senior["노인무임승차(천명)"],
                  "노인무임승차비율":df_senior["노인무임승차비율"],
                  "무임승차대상중노인비율":fr_ratio
                 })
a
```


```python
# 다중 막대 그래프 생성
#fig = go.Figure()
fig = make_subplots(specs=[[ {"secondary_y": True}]])
#specs=[[{}, {"secondary_y": True}]

fig.add_trace(go.Bar(x=df_senior['연도'], y=df_senior['연간승차(천명)'], name='연간승차(천명)'),secondary_y=False)
fig.add_trace(go.Bar(x=df_senior['연도'], y=df['무임승차(천명)'], name='무임승차(천명)'),secondary_y=False)
fig.add_trace(go.Bar(x=df_senior['연도'], y=df_senior['노인무임승차(천명)'], name='노인무임승차(천명)'),secondary_y=False)
fig.add_trace(go.Scatter(x=df_senior['연도'], y=fr_ratio, name='무임승차대상 중 노인비율'),secondary_y=True)
fig.add_trace(go.Scatter(x=df_senior['연도'], y=df_senior['노인무임승차비율'], name='노인무임승차비율'),secondary_y=True)

fig.update_yaxes(title_text="인원(천명)", secondary_y=False)
fig.update_yaxes(title_text="비율(%)",range=[10,84], secondary_y=True)
fig.update_yaxes(title_text="비율(%)",range=[10,15], secondary_y=True)

# 레이아웃 설정
fig.update_layout(
    title='연간 승차자수와 무임승차자수')

# 그래프 출력
fig.show()
```


```python
# df_sub_mean 사용

import matplotlib.pyplot as plt

# 몇개그릴지
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 인원수 비율
values1 = df_sub_mean['무임승차(천명)'].tolist()
labels1 = df_sub_mean.index.tolist()
colors1 = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
ax1.pie(values1, colors=colors1, wedgeprops=dict(width=0.7), startangle=-40, autopct='%1.1f%%')
ax1.set_title('대상별 무임승차 비율', fontsize=16)

# 비용 비율
values2 = df_sub_mean['무임비용(억원)'].tolist()
labels2 = df_sub_mean.index.tolist()
colors2 = ['lightskyblue', 'lightcoral', 'gold', 'yellowgreen']
ax2.pie(values2, colors=colors2, wedgeprops=dict(width=0.7), startangle=-40, autopct='%1.1f%%')
ax2.set_title('대상별 무임비용 비율', fontsize=16)

# 도넛차트에 레이블 추가
ax1.legend(labels1, loc='best')
ax2.legend(labels2, loc='best')

plt.show()

```


```python
# 퍼센트 표시 성공
fig = make_subplots(specs=[[{"secondary_y": True}]])


# 노인무임비용비율
fig.add_trace(go.Bar(x=df_senior['연도'], y=df_senior['노인무임비용비율']/100, name='손실액 대비 노인무임승차비용', marker=dict(color='orange')), secondary_y=False)

# 노인무임승차비율
fig.add_trace(go.Scatter(x=df_senior['연도'], y=df_senior['노인무임승차비율']/100, name='승객 수 대비 노인무임승차 수', line=dict(color='blue')), secondary_y=True)


#y축 
fig.update_yaxes(title_text='비율비율(%)', range=[0, 1], tickformat=".0%", secondary_y=False)
fig.update_yaxes(title_text="승차비율(%)", range=[0.1,0.2], tickformat=".0%",secondary_y=True)

fig.update_layout(title_text='노인 무임승차수 비율과 손실액 대비 무임비용 비율',
                  xaxis_title='연도')

fig.show()
```


```python
# 페이지 16

df_senior_pct
```


```python
# 페이지 16
df
```


```python
# 다양하게
# df_senior

df_senior["연도"] = df_senior["연도"].astype(int)


import plotly.graph_objs as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=2,
                    specs=[[{}, {"secondary_y": True}],
                           [{}, {"secondary_y": True}],
                           [{}, {"secondary_y": True}]],
                    subplot_titles=("연간 승차인원", "노인 무임승차 비용", "노인 무임승차인원 비율", "노인무임비용 대비 손실액비율", "노인무임승차비율과 증감율","영업손실액"))


# 승차인원 선그래프
fig.add_trace(
    go.Scatter(x=df_senior["연도"], y=df_senior["연간승차(천명)"], mode="lines+markers", name="연간승차(천명)"),
    row=1, col=1)
fig.add_trace(
    go.Scatter(x=df_senior["연도"], y=df_senior["노인무임승차(천명)"], mode="lines+markers", name="노인무임승차(천명)"),
    row=1, col=1)
fig.update_xaxes(title_text="연도", row=1, col=1)
fig.update_yaxes(title_text="승차인원(천명)", row=1, col=1)

# 노인무임승차비용 막대그래프
fig.add_trace(
    go.Bar(x=df_senior["연도"], y=df_senior["노인무임비용(억원)"], name="노인무임비용(억원)"),
    row=1, col=2)
fig.update_xaxes(title_text="연도", row=1, col=2)
fig.update_yaxes(title_text="금액(억원)", range=[0, 12000], row=1, col=2)

# 승차인원 비율 선그래프
fig.add_trace(
    go.Scatter(x=df_senior["연도"], y=df_senior["노인무임승차비율"], mode="lines+markers", name="노인무임승차비율"),
    row=2, col=1)
fig.update_xaxes(title_text="연도", row=2, col=1)
fig.update_yaxes(title_text="증감율(%)", range=[10,15], row=2, col=1)

# 노인무임승차비용 비율 막대
fig.add_trace(
    go.Bar(x=df_senior["연도"], y=df_senior["노인무임비용비율"], name="노인무임비용/손실액"),
    row=2, col=2)
fig.update_xaxes(title_text="연도", row=2, col=2)
fig.update_yaxes(title_text="비율(%)", range=[0, 100], row=2, col=2)


# 노인무임승차비용, 증감율
fig.add_trace(
    go.Bar(x=df_senior["연도"], y=df_senior["노인무임승차비율"], name="노인무임승차비율"),
    row=3, col=1)
fig.add_trace(
    go.Scatter(x=df_senior["연도"], y=df_senior_pct["노인무임승차비율증감율"], mode="lines+markers", name="비율증감율"),
    row=3, col=1)
fig.update_xaxes(title_text="연도", row=3, col=1)
fig.update_yaxes(title_text="비율/증감율(%)", range=[-10, 25], row=3, col=1)

# 영업손실액
fig.add_trace(
    go.Bar(x=df_senior["연도"], y=df_senior["영업손실(억원)"], name="영업손실(억원)"),
    row=3, col=2, secondary_y=False)
fig.add_trace(
    go.Scatter(x=df_senior["연도"], y=df_senior_pct["영업손실증감율"], mode="lines+markers", name="영업손실증감율"),
    row=3, col=2,  secondary_y=True )
fig.update_xaxes(title_text="연도", row=3, col=2)
fig.update_yaxes(title_text="금액(억원)", range=[0, 12000], row=3, col=2, secondary_y=False)
fig.update_yaxes(title_text="증감율(%)", range=[-50,150], row=3, col=2, secondary_y=True)
 

fig.update_layout(height=1000, width=1000,title_text="여러가지 시각화")
fig.show()

```


```python
# df_senior
# 20년도 팍 떨어지는거... 
# 영업손실(승차인원수가 떨어져서) 하지만 노인무임비용은 줄었다
# 페이지 16


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(x=df_senior['연도'], y=df_senior['영업손실(억원)'], name='영업손실액'))
fig.add_trace(go.Bar(x=df_senior['연도'], y=df_senior['노인무임비용(억원)'], name='노인무임비용'))

fig.update_layout(title='연도별 노인무임/승차인원과 노인무임비용/손실액',
                  xaxis_title='연도',
                  yaxis=dict(title='금액(억원)', side='left'),
                  yaxis2=dict(title='인원(천명)', side='right', overlaying='y'),
                  legend=dict(x=1.2, y=0.9)) # 범례 위치 조정

fig.add_trace(go.Scatter(x=df_senior['연도'], y=df_senior['연간승차(천명)'], name='연간승차자수', 
                         yaxis='y2'))
fig.add_trace(go.Scatter(x=df_senior['연도'], y=df_senior['노인무임승차(천명)'], name='노인무임승차자수', 
                         yaxis='y2'))

fig.show()

```


```python
# 페이지 16

# 연도 컬럼을 시각화하지 않기위해 
# 깊은복사 후 '연도'컬럼 인덱스 설정
df_c = df_senior_pct.copy()
df_c.set_index('연도', inplace=True)

# 다중 막대그래프 시각화
ax = df_c.plot(kind='bar', figsize=(10,6), width=0.7)
ax.set_title('항목 별 증감율', fontsize=20)
ax.set_xlabel('연도', fontsize=16)
ax.set_ylabel('증감율', fontsize=16)
plt.xticks(fontsize=12, rotation=0)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
```

# 만약 노인연령을 상향한다면…

## 노인 연령을 70세로 상향한다면 정말 지하철 적자를 유의미하게 메꿀 수 있을까?

### 17~21년도 70세로 연령 상향 시 발생하는 이익

#### 데이터 불러오기

##### 서울시 연령대별 인구 데이터


```python
# 서울시 연령대별 인구 데이터 데이터 가져오기

pop = pd.read_csv(url_pop, encoding='cp949')
pop.head()
```


```python
# 데이터의 형식과 구조 확인하기

pop.shape
pop.info()
```

##### 지하철 운영기관별 손익계산서 데이터


```python
#지하철 운영기관별 손익계산서 데이터 가져오기

pnl = pd.read_csv(url_pnl, encoding='cp949')
pnl.head()
```


```python
# 데이터의 형식과 구조 확인하기
# 0행 때문에 모두 object로 처리됨. -> 수정필요

pnl.shape
pnl.info()


```

#### 데이터 전처리

##### 서울시 연령대별 인구 데이터 전처리


```python
# 서울시 연령대별 인구 데이터 전처리
pop.head()
```


```python
# 불필요한 컬럼 제거 및 컬럼명 바꾸기
pop = pop.drop(columns=['시나리오별(1)', '성별(1)'], axis=1)
pop = pop.rename(columns = {'시도별(1)':'시도', '연령별(1)':'연령대', '연령별(2)':'나이'})
pop.head(2)
```


```python
# tidy data 만들기

pop_m = pd.melt(pop, id_vars=['시도', '연령대', '나이'], var_name='연도', value_name='인구수')
pop_m
```


```python
#컬럼별 data type 확인
# melt를 하는 과정에서 '연도'가 셀로 내려옴에 따라 object 타입으로 바뀜

# 연도 타입 object -> int

pop_m.dtypes
pop_m["연도"] = pop_m["연도"].astype("int")
```

##### 영업손익 데이터 전처리



```python
pnl.head()
```


```python
# 컬럼명 변경

pnl_cols = ["운영기관", "연도", "영업수입 합계", "운수사업수익", "기타사업수익",
           "영업비용", "인건비", "경비", "영업손익", "영업외수익", "영업외비용", "경상손익"]
pnl.columns = pnl_cols
pnl.head()
```


```python
# 불필요한 0행 제거

pnl = pnl.drop([0], axis=0)
```


```python
# 필요한 컬럼만 가져오기
# '운영기관' 컬림 요소 확인하기

pnl2 = pnl[["운영기관", "연도", "영업손익"]]
pnl2["운영기관"].unique()
```


```python
# 서울 지하철만 가져오기
seoul_subway = ['서울교통공사', '서울메트로 9호선㈜', '서울교통공사9호선운영부문', '우이 신설경전철㈜']

pnl2 = pnl2[pnl2["운영기관"].isin(seoul_subway)]
pnl2
```


```python
pnl2.dtypes
```


```python
# 연도, 영업손익 데이터 타입 바꾸기
pnl2["연도"] = pnl2["연도"].astype(int)
pnl2["영업손익"] = pnl2["영업손익"].astype(int)
pnl2.dtypes
```


```python
# 연도별 영업손익 합계 구하기
# 우선 2017년도로 생각해보기

pnl2[pnl2["연도"] == 2017]["영업손익"].sum()
```


```python
# 연도별 영업손익 합계 구하는 함수
def yr_pnl(yr):
    
    # 해당연도의 영업손익 합계 가져오기
    pnl2_yr_sum = pnl2[pnl2["연도"] == yr]["영업손익"].sum()
    
    return pnl2_yr_sum

# 반복문으로 모든 연도의 영업손익 합계 가져오기
yr_pnl_list = []

for i in range(2017, 2022):
    yr_pnl_list.append(yr_pnl(i))
    
yr_pnl_list
```


```python
# 연도별 영업손익

yr = [2017, 2018, 2019, 2020, 2021]
yr_pnl = pd.DataFrame({'연도':yr,
                      '영업손익(억원)':yr_pnl_list})
yr_pnl
```

#### 연도별 서울시 만 65세 ~ 69세 인구수 구하기
- 우선 17년으로 작업 후 21년까지 반복문 돌리기


```python
# 2017년 자료만 가져오기
pop_17 = pop_m[pop_m["연도"] == 2017]
pop_17.tail(2)
```


```python
# 2017년 서울시 만 65세 ~ 69세 인구 수 구하기 ('소계' 이용)

pop_17_6569sum = (pop_17[pop_17["연령대"] == "65 - 69세"]).iloc[0, -1]
display(pop_17[pop_17["연령대"] == "65 - 69세"])
display(pop_17_6569sum)
```

- **17 ~ 21년 서울시 만 65세 ~ 69세 인구 수만 추출하기**


```python
# 연도별 해당연도의 65-69세 인구 수 합계를 구하는 함수

def pop_yr_6569sum(yr):
    
    # 해당연도 데이터프레임만 가져오기
    pop_yr = pop_m[pop_m["연도"] == yr]
    
    # 해당연도의 65-69세 인구 수 합계 가져오기
    pop_yr_6569sum = (pop_yr[pop_yr["연령대"] == "65 - 69세"]).iloc[0, -1]
    
    return pop_yr_6569sum
```


```python
# 반복문 돌리기 -> 17 ~ 21년 서울시 만 65세 ~ 69세 인구 수만 추출하기

yr_6569sum_list = []

for i in range(2017, 2022):
    yr_6569sum_list.append(pop_yr_6569sum(i))
    
yr_6569sum_list
```


```python
# 데이터 프레임으로 만들기

yr_6569 = pd.DataFrame({'연도':yr,
                      '만 65세 ~ 69세 인구 수':yr_6569sum_list})
yr_6569
```

####  연도별 "대중교통을 이용하는" 서울시 만 65세 ~ 69세 인구수 구하기

- (2018년도 기준)
- 서울시 만 65세 이상 인구 수 (1,341,836) - 출처: KOSIS
- 대중교통을 이용하는 서울시 만 65세 이상 인구 수 (83만명) - 출처: 뉴스기사(http://segyelocalnews.com/news/newsview.php?ncode=1065625174648440#:~:text=%EB%B6%84%EC%84%9D%20%EA%B2%B0%EA%B3%BC%20%EC%84%9C%EC%9A%B8%20%EB%8C%80%EC%A4%91%EA%B5%90%ED%86%B5,%EC%A3%BC%EC%9D%BC%EA%B0%84%208%EB%A7%8C%EB%AA%85%EC%9D%B4%20%EB%84%98%EC%97%88%EB%8B%A4)
- => 따라서 만 65세 이상 노인 인구 중 대중교통을 이용하는 인구 수 비율은 약 61.9%

#### 연도별 "대중교통을 이용하는" 서울시 만 65세 ~ 69세 인구수 구하기

- (2018년도 기준)
- 서울시 만 65세 이상 인구 수 (1,341,836) - 출처: KOSIS
- 대중교통을 이용하는 서울시 만 65세 이상 인구 수 (83만명) - 출처: 뉴스기사
- (http://segyelocalnews.com/news/newsview.php?ncode=1065625174648440#:~:text=%EB%B6%84%EC%84%9D%20%EA%B2%B0%EA%B3%BC%20%EC%84%9C%EC%9A%B8%20%EB%8C%80%EC%A4%91%EA%B5%90%ED%86%B5,%EC%A3%BC%EC%9D%BC%EA%B0%84%208%EB%A7%8C%EB%AA%85%EC%9D%B4%20%EB%84%98%EC%97%88%EB%8B%A4)
- => 따라서 만 65세 이상 노인 인구 중 대중교통을 이용하는 인구 수 비율은 약 61.9%


```python
yr_6569["대중교통을 이용하는 서울시 만 65세 ~ 69세 인구 수"] = yr_6569["만 65세 ~ 69세 인구 수"]*0.691
yr_6569
```

#### 연도별 "지하철을 이용하는" 서울시 만 65세 ~ 69세 인구수 구하기


```python
yr_6569["지하철을 이용하는 서울시 만 65세 ~ 69세 인구 수"] = yr_6569["대중교통을 이용하는 서울시 만 65세 ~ 69세 인구 수"]*0.8
yr_6569
```

#### 최종) 연도별 노인연령을 70세로 상향 시 발생하는 이익
- (http://segyelocalnews.com/news/newsview.php?ncode=1065625174648440#:~:text=%EB%B6%84%EC%84%9D%20%EA%B2%B0%EA%B3%BC%20%EC%84%9C%EC%9A%B8%20%EB%8C%80%EC%A4%91%EA%B5%90%ED%86%B5,%EC%A3%BC%EC%9D%BC%EA%B0%84%208%EB%A7%8C%EB%AA%85%EC%9D%B4%20%EB%84%98%EC%97%88%EB%8B%A4)
- 하루 평균 노인들의 대중교통 이용 횟수는 2.4회
- 기본요금 1250원


```python
yr_6569["70세로 연령 상향 시 하루에 발생하는 이익"] = yr_6569["지하철을 이용하는 서울시 만 65세 ~ 69세 인구 수"]*2.4*1250
yr_6569
```


```python
yr_6569["노인연령상향 시 발생하는 연간 이익"] = yr_6569["70세로 연령 상향 시 하루에 발생하는 이익"] * 365
yr_6569
```


```python
# 천단위 콤마와 단위 붙이기
yr_6569["노인연령상향 시 발생하는 연간 이익"] = yr_6569["노인연령상향 시 발생하는 연간 이익"].map('{:,.0f}'.format) + "원"
```


```python
yr_6569
```


```python
yr_70up_pft = yr_6569[["연도","노인연령상향 시 발생하는 연간 이익"]]
yr_70up_pft
```


```python
# pandas로 그려보기
# 참고 : https://jimmy-ai.tistory.com/24
# 단위가 너무 차이나서 제대로 그래프가 그려지지 않음...

x = yr_6569["연도"]
y = yr_6569["노인연령상향 시 발생하는 연간 이익"]

bar = plt.bar(x, y)

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 8)

plt.title("연도별 70세로 연령 상향 시 발생하는 연간 이익")
plt.show()
```


```python
# plotly로 그려보기
# seaborn, matplotlib에 비해 훨씬 느림

px.bar(yr_6569, x="연도", y="노인연령상향 시 발생하는 연간 이익", height=400)
```


```python
# 연도별 영업손익과 노인연령 상향 시 발생하는 수익 비교

display(yr_pnl)
display(yr_70up_pft)
```


```python
# 영업손실 단위 변경: 억원 -> 원

yr_pnl["영업손익(억원)"] = yr_pnl["영업손익(억원)"].astype(int)
yr_pnl["영업손익(원)"] = yr_pnl["영업손익(억원)"]*(1E+8)

yr_pnl
```


```python
yr_pnl["영업손익"] = yr_pnl["영업손익(원)"].map('{:,.0f}'.format) + "원"

yr_pnl
```


```python
yr_pnl2 = pd.merge(yr_pnl, yr_70up_pft, on="연도")
yr_lp = yr_pnl2[["연도", "영업손익", "노인연령상향 시 발생하는 연간 이익"]]
yr_lp
```


```python
# 시각화를 나타내기 위해 숫자형으로 다시 바꾸기

yr_6569["노인연령상향 시 발생하는 연간 이익(원)"] = yr_6569["70세로 연령 상향 시 하루에 발생하는 이익"] * 365
```


```python
yr_6569
```


```python
display(yr_pnl[["연도", "영업손익(원)"]])
display(yr_6569[["연도", "노인연령상향 시 발생하는 연간 이익(원)"]])

yr_loss = yr_pnl[["연도", "영업손익(원)"]]
yr_70up_pft2 = yr_6569[["연도", "노인연령상향 시 발생하는 연간 이익(원)"]]
```


```python
# 영업손익 데이터 프레임과 65세~69세 인구 합계 데이터프레임 연도를 기준으로 병합
yr_lnp = pd.merge(yr_loss, yr_70up_pft2, on="연도")
yr_lnp
```


```python
yr_lnp["영업손실"] = yr_lnp["영업손익(원)"]*(-1)
yr_lnp
```


```python
yr_lnp_rate = yr_lnp["노인연령상향 시 발생하는 연간 이익(원)"]/yr_lnp["영업손실"]
yr_lnp["영업손실대비 연령상한시 이익비율(%)"] = round(yr_lnp_rate, 4)
yr_lnp
```


```python
yr_lnp[["연도", "영업손실", "노인연령상향 시 발생하는 연간 이익(원)", "영업손실대비 연령상한시 이익비율(%)"]]
```


```python
yr_lnp.plot.bar(x="연도", y="영업손실대비 연령상한시 이익비율(%)", rot=0
               , title="연도별 영업손실대비 연령상한시 이익비율")
```

### 17~21년도 지하철 1인당 평균운임과 70세로 연령 상향시 발생했을 1인당 평균운임 비교

#### 데이터 불러오기


```python
# 무임승차 대상별 현황 데이터 불러오기
# 승객 1인당 운임손실 현황 데이터 불러오기
# 승차 및 수송인원 데이터 불러오기
nm = pd.read_csv(url_nomoney, encoding='cp949')
dft = pd.read_csv(url_deficit, encoding='cp949')
cry = pd.read_csv(url_carry, encoding='cp949')
```

#### 데이터 전처리


```python
# 무임승차 대상별 현황 데이터 살펴보기
nm.head()
nm.tail()
```


```python
# '무임비용' 컬럼이 object인 것을 확인 -> 숫자형으로 수정필요
nm.shape
nm.info()
```


```python
# 컬럼명 변경
nm = nm.rename(columns = {"운영기관별(1)":"운영기관", "시점":"연도", "대상별(1)":"대상"})
nm.head()
```


```python
# # '무임비용 (백만원)' 컬럼의 타입을 int로 변경하기
# # '-'값 발견
# nm['무임비용 (백만원)'] = nm['무임비용 (백만원)'].astype(int)
```


```python
# '-'를 null값으로 변경
nm['무임비용 (백만원)'] = nm['무임비용 (백만원)'].replace('-', np.nan)
nm['무임비용 (백만원)'].isnull().sum()
```


```python
# ['무임비용 (백만원)'] 컬럼의 타입을 float로 변경하기
nm['무임비용 (백만원)'] = nm['무임비용 (백만원)'].astype(float)
nm.dtypes
```


```python
# '대상' 컬럼에서 '노인'만 가져오기
nm_elderly = nm[nm["대상"] == "노인"].reset_index(drop=True)
nm_elderly
```


```python
# '운영기관'에서 서울 지하철만 가져오기
nm_elderly["운영기관"].unique()
```


```python
seoul_subway = ['서울교통공사', '서울메트로 9호선(주)', '서울교통공사9호선운영부문', '우이 신설경전철(주)']
nm_elderly = nm_elderly[nm_elderly["운영기관"].isin(seoul_subway)]
nm_elderly
```


```python
# 연도별 서울 전체 지하철의 무임비용 구하기
# 일단 2017년부터

nm_elderly[nm_elderly["연도"] == 2017]
```


```python
# 연도를 입력하면 노인 무임비용의 합계를 구하는 함수

def yr_nm(yr):
    nm_yr = nm_elderly[nm_elderly["연도"] == yr]
    nm_sum = nm_yr["무임비용 (백만원)"].sum()
    
    return nm_sum
```


```python
yr_nm(2017)
```


```python
# 17~21년 노인 무임비용의 합계 구하기

yr_nm_list = []

for i in range(2017, 2022):
    yr_nm_list.append(yr_nm(i))

yr_nm_list
```


```python
# 데이터프레임 만들기

yr_nm = pd.DataFrame({'연도':yr, '무임비용(백만원)': yr_nm_list})
yr_nm
```

##### 승객 1인당 운임손실 현황 살펴보기



```python
dft.shape
```


```python
dft.head()
```


```python
dft.tail()
```


```python
# 운임손실, 수송원가 컬럼이 object인 것을 확인 -> 숫자형으로 수정 필요

dft.info()
```


```python
# # 데이터 타입 변경
# # '-'값 발견
# dft['운임손실 (원)'] = dft['운임손실 (원)'].astype('int')
# dft['수송원가 (원)'] = dft['수송원가 (원)'].astype('int')
```


```python
# '-'값을 null값으로 변경

dft['운임손실 (원)'] = dft['운임손실 (원)'].replace('-', np.nan)
dft['수송원가 (원)'] = dft['수송원가 (원)'].replace('-', np.nan)
```


```python
display(dft['운임손실 (원)'].isnull().sum())
display(dft['수송원가 (원)'].isnull().sum())
```


```python
# float형식으로 변환
dft['운임손실 (원)'] = dft['운임손실 (원)'].astype('float')
dft['수송원가 (원)'] = dft['수송원가 (원)'].astype('float')
```


```python
# 컬럼명 변경
dft = dft.rename(columns = {"운영기관별(1)":"운영기관", "시점":"연도"})
dft.head()
```


```python
# '운영기관'에서 서울 지하철만 가져오기
dft["운영기관"].unique()

seoul_subway = ['서울교통공사', '서울메트로 9호선㈜', '서울교통공사9호선운영부문', '우이 신설경전철(주)']
dft = dft[dft["운영기관"].isin(seoul_subway)]
dft.tail()
```


```python
# 연도별 서울의 모든 지하철의 평균 운임료 구하기
# 우선 2017년 평균 운임료 구하기

dft_17 = dft[dft["연도"] == 2017]
dft_17
```


```python
# 2017년도의 서울 지하철 기관들의 평균운임의 평균
dft_17["평균운임 (원)"].mean()
```


```python
# 연도를 입력하면 평균운임이 출력되는 함수

def yr_mean(yr):
    dft_yr = dft[dft["연도"] == yr]
    yr_mean = dft_yr["평균운임 (원)"].mean()
    
    return yr_mean
```


```python
# 함수가 잘 작동하는지 확인
yr_mean(2017)
```


```python
# 17~21년 평균운임 구하기

yr_mean_list = []

for i in range(2017, 2022):
    yr_mean_list.append(yr_mean(i))

yr_mean_list
```


```python
# 데이터프레임 만들기

yr_fare = pd.DataFrame({'연도':yr, '평균운임(원)': yr_mean_list})
yr_fare
```

##### 승차 및 수송인원 데이터 살펴보기



```python
# 승차 및 수송인원 데이터 살펴보기
cry.head()
```


```python
cry.isnull().sum()
```


```python
cry.info()
```


```python
cry["운영기관별(1)"].unique()
```


```python
cry_seoul = cry[cry["운영기관별(1)"] == '서울']
cry_seoul
```


```python
cry_seoul = cry_seoul.rename(columns = {"운영기관별(1)":"지역"})
cry_seoul.head(2)
```


```python
# 서울의 연도별 수송인원 구하기

yr_seoul_carry = pd.DataFrame({"연도":[2017, 2018, 2019, 2020, 2021], "수송인원(천명)":cry_seoul["수송인원 (천명/년)"].to_list()})
yr_seoul_carry
```

##### 전국 연령대별 인구수 데이터


```python
# 17~21년도 70세로 연령 상향 시 발생하는 이익에서 생성한 pop_m 테이블로
# 연도별 만 65세 이상 인구수 구하기

pop_m.head()
```


```python
pop_m[(pop_m["연도"]==2017) & (pop_m["나이"]=="소계")]
```


```python
pop_m[(pop_m["연도"]==2017) & (pop_m["나이"]=="소계")]["인구수"].sum()
```


```python
yr_pop_65 = []

for i in range(2017, 2022):
    sum_65 = pop_m[(pop_m["연도"]==i) & (pop_m["나이"]=="소계")]["인구수"].sum()
    yr_pop_65.append(sum_65)
    
yr_pop_65
```


```python
yr_pop_65 = pd.DataFrame({"연도":yr, "65세 이상 인구수": yr_pop_65})
yr_pop_65
```


```python
# 연도별 만 65세 ~ 69세 인구수 구하기

yr_pop_6569 = yr_6569[["연도", "만 65세 ~ 69세 인구 수"]]
yr_pop_6569
```

#### 5개 데이터프레임 합치기


```python
df_list = [yr_fare, yr_nm, yr_seoul_carry, yr_pop_65, yr_pop_6569]
```


```python
from functools import reduce

df_merge = reduce(lambda left, right: pd.merge(left, right, on='연도'), df_list)
df_merge
```

참고 : https://stopsilver.tistory.com/entry/python-pandas-%EC%97%AC%EB%9F%AC%EA%B0%9C%EC%9D%98-dataframe%EC%9D%84-merge

### 분석3. 17~21년도 지하철 운임손실(영업손실)과 70세로 연령 상향했다면 발생하는 운임손실 비교

#### 데이터 불러오기


```python
df_merge
```

### 전처리

![image.png](attachment:image.png)


```python
# 현재 지하철이 버는 돈(노인 연령 상향 X)
subway_money = df_merge['평균운임(원)'] * (df_merge['수송인원(천명)']*(1E+3))
subway_money
```


```python
# 노인 연령을 70세로 높이면 지하철이 버는 돈
subway_70up_money = df_merge['무임비용(백만원)']*(1E+6) * (df_merge['만 65세 ~ 69세 인구 수']/df_merge['65세 이상 인구수'])
subway_70up_money
```


```python
# 서울지하철이 해당연도에 수송한 총 인원
carry_num = df_merge["수송인원(천명)"]*(1E+3)
carry_num
```


```python
# 노인연령을 70세로 상향 시 평균 운임료

sbw_70up_fare = (subway_money + subway_70up_money) / carry_num
sbw_70up_fare
```


```python
mean_fare_70up = pd.DataFrame({"연도":yr, "노인연령을 70세로 상향 시 평균 운임(원)":sbw_70up_fare})
mean_fare_70up
```


```python
df_merge.iloc[:, :2]
```


```python
# 기존 평균운임과 노인연령을 70세로 상향했을 때 평균 운임 비교

sub_fare = pd.merge(df_merge.iloc[:, :2], mean_fare_70up, on="연도")
sub_fare["평균운임(원)"] = round(sub_fare["평균운임(원)"], 1)
sub_fare["노인연령을 70세로 상향 시 평균 운임(원)"] = round(sub_fare["노인연령을 70세로 상향 시 평균 운임(원)"], 1)

sub_fare
```


```python
# 평균운임 변화율 컬럼 추가하기

icrm = (sub_fare["노인연령을 70세로 상향 시 평균 운임(원)"] - sub_fare["평균운임(원)"])/sub_fare["평균운임(원)"]
sub_fare["평균운임 변화율(%)"] = round(icrm*100, 2)
sub_fare
```


```python
sub_fare["노인연령을 70세로 상향 시 평균 운임(원)"] - sub_fare["평균운임(원)"]
```


```python
# 시각화
# 참고 : https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

x = np.arange(len(sub_fare["연도"]))
width=0.25
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in sub_fare.iloc[:,1:3].items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
ax.set_ylabel('평균운임 (원)')
ax.set_title('기존 평균운임과 노인연령을 70세로 상향했을 때 평균 운임 비교')
ax.set_xticks(x + width, sub_fare["연도"])
ax.legend(loc='upper left')
ax.set_ylim(0, 1500)

plt.show()
```


```python
# matplotlib으로 평균임금 증가량 나타내기

sub_fare2 = sub_fare.set_index("연도")
```


```python
sub_fare2
```


```python
# matplotlib으로 평균운임 증가량 시각화

sub_fare2.iloc[:, -1].plot.line(figsize=(10, 5), lw=0.9, rot=0)
```


```python
# seaborn으로 같은 시각화 해보기

fare_icrs = sns.lineplot(x="연도", y="평균운임 변화율(%)", data=sub_fare)
fare_icrs.set_title("연도별 노인연령상향 시 평균운임 변화율")
```

### 17~21년도 지하철 운임손실과 70세로 연령 상향했다면 발생하는 운임손실 비교

#### 데이터 불러오기


```python
# 필요한 컬럼 : 평균운임(원), 노인연령을 70세로 상향 시 평균 운임(원)

sub_fare
```


```python
# 필요한 컬럼 : (연도별) 운임손실 -> 전처리 필요

dft.head(10)
```

#### 전처리

![image.png](attachment:image.png)


```python
dft.info()
dft.head(2)
```


```python
dft["운영기관"].unique()
```


```python
# 연도별 평균 수송원가 구하기

def yr_mean_carry(yr):
    dft_yr = dft[dft["연도"] == yr]
    yr_mean = dft_yr["수송원가 (원)"].mean()
    
    return yr_mean


# 17~21년 평균 수송원가 구하기

yr_mean_carry_list = []

for i in range(2017, 2022):
    yr_mean_carry_list.append(yr_mean_carry(i))

yr_mean_carry_list


# 데이터프레임 만들기

yr_carry2 = pd.DataFrame({'연도':yr, '수송원가(원)': yr_mean_carry_list})
yr_carry2
```


```python
yr_mean_fare = sub_fare.iloc[:, :-1]
yr_mean_fare
```


```python
# yr_carry2와 yr_mean_fare 데이터 병합

loss = pd.merge(yr_carry2, yr_mean_fare, on="연도")
loss
```

##### 기존 운임손실 구하기


```python
loss["기존운임손실(원)"] = loss["수송원가(원)"] - loss["평균운임(원)"]
loss
```

##### 노인연령 상향시 운임손실 구하기



```python
loss["노인연령 상향시 운임손실(원)"] = loss["수송원가(원)"] - loss["노인연령을 70세로 상향 시 평균 운임(원)"]
loss
```


```python
# 운임손실 변화량 컬럼 추가하기

loss2 = (loss["노인연령 상향시 운임손실(원)"] - loss["기존운임손실(원)"]) / loss["기존운임손실(원)"]
loss["운임손실 변화율(%)"] = round(loss2*100, 2)
loss
```


```python
# 시각화
# 참고 : https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

x = np.arange(len(loss["연도"]))
width=0.25
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in loss.iloc[:,4:6].items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
ax.set_ylabel('운임손실(원)')
ax.set_title('승객 1인당 기존 운임손실과 노인연령을 70세로 상향했을 때 운임손실 비교')
ax.set_xticks(x + width, loss["연도"])
ax.legend(loc='upper left')
ax.set_ylim(0, 1500)

plt.show()
```


```python
# matplotlib으로 운임손실 변화량 나타내기
loss2 = loss.set_index("연도")
```


```python
loss2.iloc[:, -1].plot.line(figsize=(10, 5), lw=0.9, rot=0)
```


```python
# seaborn으로 같은 시각화 해보기

loss_dcrs = sns.lineplot(x="연도", y="운임손실 변화율(%)", data=loss)
loss_dcrs.set_title("연도별 노인연령상향 시 운임손실 변화율")
```

##### 연간 운임 손실액 구하기


```python
loss
```


```python
df_merge
```


```python
# loss와 df_merge에서 필요한 컬럼만 추출

loss2 = loss[["연도", "기존운임손실(원)", "노인연령 상향시 운임손실(원)"]]
carry2 = df_merge[["연도", "수송인원(천명)"]]
display(loss2)
display(carry2)
```


```python
df_ls_cry = pd.merge(loss2, carry2, on="연도")
df_ls_cry
```


```python
df_ls_cry["수송인원(명)"] = df_ls_cry["수송인원(천명)"] * 1000
df_ls_cry
```


```python
# 운임손실*수송인원 => 해당 연도의 전체 운임손실액
# 연도별 운임손실*수송인원 구하기

# 기존 연간 운임손실액
df_ls_cry["기존 연간 운임손실액"] = df_ls_cry["수송인원(명)"] * df_ls_cry["기존운임손실(원)"]

# 노인연령상향 시 연간 운임손실액
df_ls_cry["노인연령상향 시 연간 운임손실액"] = df_ls_cry["수송인원(명)"] * df_ls_cry["노인연령 상향시 운임손실(원)"]

df_ls_cry
```


```python
df_ls_cry2 = df_ls_cry[["연도", "기존 연간 운임손실액", "노인연령상향 시 연간 운임손실액"]]
df_ls_cry2
```


```python
df_ls_cry2["기존 연간 운임손실액"] = df_ls_cry2["기존 연간 운임손실액"].map('{:,.0f}'.format) + "원"
df_ls_cry2["노인연령상향 시 연간 운임손실액"] = df_ls_cry2["노인연령상향 시 연간 운임손실액"].map('{:,.0f}'.format) + "원"
df_ls_cry2
```


```python
# 운임 손실액 변화율 구하기

ls_cry = (df_ls_cry["노인연령상향 시 연간 운임손실액"] - df_ls_cry["기존 연간 운임손실액"])/df_ls_cry["기존 연간 운임손실액"]
df_ls_cry2["연간 운임손실액 변화율(%)"] = round(ls_cry*100, 2)
df_ls_cry2
```


```python
# 시각화를 위해 단위 변경

df_ls_cry
```


```python
df_ls_cry3 = df_ls_cry[["연도", "기존 연간 운임손실액", "노인연령상향 시 연간 운임손실액"]]
df_ls_cry3
```


```python
# 조 단위로 변경

df_ls_cry3["기존 연간 운임손실액(조원)"] = round(df_ls_cry3["기존 연간 운임손실액"]/1E+12, 2)
df_ls_cry3["노인연령상향 시 연간 운임손실액(조원)"] = round(df_ls_cry3["노인연령상향 시 연간 운임손실액"]/1E+12, 2)
df_ls_cry3
```


```python
# 시각화
# 참고 : https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

x = np.arange(len(df_ls_cry3["연도"]))
width=0.25
multiplier = 0

fig, ax = plt.subplots(constrained_layout=True)

for attribute, measurement in df_ls_cry3.iloc[:,3:].items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
ax.set_ylabel('연간운임손실액(조원)')
ax.set_title('기존 연간 운임손실액과 노인연령상향 시 연간 운임손실액 비교')
ax.set_xticks(x + width, df_ls_cry3["연도"])
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1))
plt.figure(figsize=(13, 7))

plt.show()
```


```python
# 연간 손실액 차이 구하기

df_ls_cry3["연간 손실액 변화량"] = df_ls_cry3["노인연령상향 시 연간 운임손실액"] - df_ls_cry3["기존 연간 운임손실액"]
df_ls_cry3
```


```python
df_ls_cry2
```


```python
df_ls_cry2["연간 손실액 변화량"] = df_ls_cry3["연간 손실액 변화량"].map('{:,.0f}'.format) + "원"
df_ls_cry2
```


```python
df_ls_cry2.drop(columns="연간 운임손실액 변화율(%)")
```

## 향후 5년간(22-26년) 무임승차 연령을 상향한다면 얼마나 수익이 생길까? 과연 유의미할까?

### 무임승차 연령을 올렸을 때 향후 5년간(22-26년) 1인당 운임비용 구하기

#### 85년~21년까지 평균 물가상승률 구하기

##### 물가상승지수 데이터 불러오기


```python
# 물가상승지수
cpi = pd.read_csv(url_cpi, encoding="cp949")
```

##### 전처리


```python
cpi.shape
cpi.info()
```


```python
cpi.tail()
```


```python
# '지출목적별' 컬럼에서 '07 교통'만 가져오기
cpi_trans = cpi[cpi["지출목적별"] == "07 교통"].reset_index(drop=True)
cpi_trans
```


```python
# tidy data 만들기

cpi_melt = pd.melt(cpi_trans, id_vars=['도시별', '지출목적별'], var_name='연도', value_name='물가지수')
cpi_melt
```

##### 물가상승률 구하기


```python
# 물가상승률 계산
cpi_melt["교통물가상승률"] = 0
for i in range(1, len(cpi_melt)):
    cpi_melt["교통물가상승률"].iloc[i] = ((cpi_melt["물가지수"].iloc[i] / cpi_melt["물가지수"].iloc[i-1])-1)*100

cpi_melt

```

##### 물가상승률 시각화


```python
cpi_melt.plot(x="연도", y = "교통물가상승률")
```


```python
# 평균 물가 상승률 구하기
# 예측이 어려우므로 . ..  평균을 곱해주겠음
cpi_up = cpi_melt["교통물가상승률"].mean()
cpi_up
```

#### 22-26년 승객 1인당 운임손실 예측하기


```python
# 승객 1인당 운임손실 현황
df_fare = pd.read_csv(url_deficit, encoding='cp949')
df_fare
```

##### 전처리


```python
# 컬럼명 변경
df_fare = df_fare.rename(columns = {"운영기관별(1)":"운영기관", "시점":"연도"})
df_fare.head()
```


```python
# '운영기관'에서 서울 지하철만 가져오기
df_fare["운영기관"].unique()
```


```python
seoul_subway = ['서울교통공사', '서울메트로 9호선㈜', '서울교통공사9호선운영부문', '우이 신설경전철(주)']
df_fare = df_fare[df_fare["운영기관"].isin(seoul_subway)]
df_fare
```


```python
# 데이터타입 변경
df_fare[["연도", "운임손실 (원)", "수송원가 (원)"]] = df_fare[["연도", "운임손실 (원)", "수송원가 (원)"]].astype("int")
df_fare.dtypes
```

##### 기존 수송원가, 평균운임 비용 변동률 구해보기


```python
# 데이터프레임 만들기
df_fare_seoul = pd.DataFrame({"연도" : [2017,2018,2019,2020,2021],
                       "운임손실(원)" : 0,
                       "수송원가(원)" : 0,
                        "평균운임(원)": 0})
df_fare_seoul.set_index("연도", inplace=True)

# 4개 지하철 합산 구하기
for yr in range(2017,2022):
    for col in range(len(df_fare)):
        if df_fare["연도"].iloc[col]==yr:
            df_fare_seoul["운임손실(원)"][yr] += df_fare["운임손실 (원)"].iloc[col]
            df_fare_seoul["수송원가(원)"][yr] += df_fare["수송원가 (원)"].iloc[col]
            df_fare_seoul["평균운임(원)"][yr] += df_fare["평균운임 (원)"].iloc[col]
    df_fare_seoul["운임손실(원)"][yr] /= 4
    df_fare_seoul["수송원가(원)"][yr] /= 4
    df_fare_seoul["평균운임(원)"][yr] /= 4

df_fare_seoul.round(2)
```

##### 향후 5년 운임비용 예측

2021년 데이터 * 물가상승률


```python
# 데이터프레임 만들기
f_fare = pd.DataFrame({"연도":[2022,2023,2024,2025,2026],
         "운임손실(원)" : 0,
         "수송원가(원)" : 0,
         "평균운임(원)" : 0,
         "기본운임(원)" : 1250,
         "운임손실 변화율(%)" : 0})

f_fare
```


```python
# 변동률의 평균 곱하여 상승폭 확인(기본 2022년 값 설정 : 21년값으로 초기화된 데이터에 물가상승률 고려 상향)
f_fare["수송원가(원)"].iloc[0] = df_fare_seoul["수송원가(원)"].iloc[0] * (1+cpi_up/100)
f_fare["평균운임(원)"] = df_fare_seoul["평균운임(원)"].iloc[0]
f_fare["운임손실(원)"].iloc[0] = f_fare["수송원가(원)"].iloc[0] - f_fare["평균운임(원)"].iloc[0]
f_fare
```


```python
# 23년도 이후 변동 구하기
for col in range(1, len(f_fare)):
    f_fare["수송원가(원)"].iloc[col] = f_fare["수송원가(원)"].iloc[col-1] * (1+cpi_up/100)
    f_fare["운임손실(원)"].iloc[col] = f_fare["수송원가(원)"].iloc[col] - f_fare["평균운임(원)"].iloc[col]

    
# 운임손실 변화율 구하기
for col in range(1,len(f_fare)):
    f_fare["운임손실 변화율(%)"].iloc[col] = (f_fare["운임손실(원)"].iloc[col] - f_fare["운임손실(원)"].iloc[col-1]) / f_fare["운임손실(원)"].iloc[col-1]
```


```python
f_fare
```

##### 노인인구수 변화 구하기



```python
pop = pd.read_csv(url_pop1, encoding='cp949')
```


```python
pop.columns
```


```python
# 불필요 항목 제외
pop = pop.drop(columns=['시나리오별(1)', '시도별(1)', '성별(1)'])
```


```python
# 컬럼명 변경
pop = pop.rename(columns = {'연령별(1)':'연령대'})
pop.tail(10)
```


```python
# tidy data 만들기
pop_melt = pd.melt(pop, id_vars = "연령대", var_name ="연도", value_name="인구수")
pop_melt.tail(10)
```


```python
# 타입 확인 및 변경
pop_melt.dtypes
pop_melt["연도"] = pop_melt["연도"].astype("int")
```


```python
# 노인 연령 인구수 데이터프레임 만들기

el_pop = pd.DataFrame({"연도" : [2022,2023,2024,2025,2026], 
                      "노인인구수" : 0})
el_pop = el_pop.set_index(keys=["연도"], drop=True)
```


```python
el_pop
```


```python
for col in range(len(pop_melt)):
    if (pop_melt["연령대"].iloc[col] == '65 - 69세') | (pop_melt["연령대"].iloc[col] == '70 - 74세') | (pop_melt["연령대"].iloc[col] == '75 - 79세') | (pop_melt["연령대"].iloc[col] == '80세이상'):
        yr = pop_melt["연도"].iloc[col]
        el_pop.loc[yr]["노인인구수"] += pop_melt["인구수"].iloc[col]
```


```python
el_pop["전체인구수"] = 0
el_pop
```


```python
for col in range(len(pop_melt)):
    if pop_melt["연령대"].iloc[col] == "계":
        yr = pop_melt["연도"].iloc[col]
        el_pop["전체인구수"].loc[yr] = pop_melt["인구수"].iloc[col]
```


```python
el_pop
```


```python
# 노인 비율
el_pop["노인비율"] = 0
for col in range(len(el_pop)):
    el_pop["노인비율"].iloc[col] = el_pop["노인인구수"].iloc[col] / el_pop["전체인구수"].iloc[col]
    
el_pop
```


```python
# 65-70세 인구수 구하기
el_pop["65-70세 노인인구수"] = [1,2,3,4,5]
```


```python
for col in range(len(pop_melt)):
    if pop_melt["연령대"].iloc[col] == '65 - 69세':
        yr = pop_melt["연도"].iloc[col]
        el_pop["65-70세 노인인구수"].loc[yr] = float(pop_melt["인구수"].iloc[col])
        
el_pop
```

##### 평균운임에 노인 비율 적용하기


```python
f_fare_ch = f_fare.copy()
```


```python
for col in range(len(f_fare_ch)):
    f_fare_ch["평균운임(원)"].iloc[col] = f_fare_ch["평균운임(원)"].iloc[col] * (1 - el_pop["노인비율"].iloc[col])
    f_fare_ch["운임손실(원)"].iloc[col] = f_fare_ch["수송원가(원)"].iloc[col] - f_fare_ch["평균운임(원)"].iloc[col]
    
# 운임손실 변화율 구하기
for col in range(1,len(f_fare)):
    f_fare_ch["운임손실 변화율(%)"].iloc[col] = (f_fare_ch["운임손실(원)"].iloc[col] - f_fare_ch["운임손실(원)"].iloc[col-1]) / f_fare_ch["운임손실(원)"].iloc[col-1]
    
f_fare_ch["운임손실 변화율(%)"] = round(f_fare_ch["운임손실 변화율(%)"]*100, 2)
```


```python
f_fare_ch
```


```python
f_fare_ch.plot(x="연도", y=["운임손실(원)", "평균운임(원)"])
plt.xticks(f_fare_ch["연도"]);
```

#### 22-26년 70세 상향시 운임손실 예측하기


```python
# 65-70세 노인인구 수가 유임승차시 노인 비율에 변화 발생
# (노인인구수 - 65-70세 노인인구수) / 전체인구수 = 변화된 노인비율 
el_pop
```

##### 변화된 노인(70세이상)비율 구하기


```python
# 변화된 노인 비율
el_pop["변화된 노인비율"] = 0
for col in range(len(el_pop)):
    el_pop["변화된 노인비율"].iloc[col] = (el_pop["노인인구수"].iloc[col] - el_pop["65-70세 노인인구수"].iloc[col]) / el_pop["전체인구수"].iloc[col]
    
el_pop
```

##### 변화된 노인비율이 적용된 운임비용 구하기


```python
f_fare_ch2 = f_fare.copy()
```


```python
for col in range(len(f_fare_ch2)):
    f_fare_ch2["평균운임(원)"].iloc[col] = f_fare_ch2["평균운임(원)"].iloc[col] * (1 - el_pop["변화된 노인비율"].iloc[col])
    f_fare_ch2["운임손실(원)"].iloc[col] = f_fare_ch2["수송원가(원)"].iloc[col] - f_fare_ch2["평균운임(원)"].iloc[col]
    
        
# 운임손실 변화율 구하기
for col in range(1,len(f_fare_ch2)):
    f_fare_ch2["운임손실 변화율(%)"].iloc[col] = (f_fare_ch2["운임손실(원)"].iloc[col] - f_fare_ch2["운임손실(원)"].iloc[col-1]) / f_fare_ch2["운임손실(원)"].iloc[col-1]
    
f_fare_ch2["운임손실 변화율(%)"] = round(f_fare_ch2["운임손실 변화율(%)"]*100, 2)
```


```python
f_fare_ch2
```


```python
f_fare_ch
```

##### 시각화



```python
# 70세 상향시 예측 그래프
f_fare_ch2.plot(x="연도", y=["운임손실(원)", "평균운임(원)"])
plt.xticks(f_fare["연도"]);

# 65세 이상 노인 인구수 변화율 적용 예측 그래프
f_fare_ch.plot(x="연도", y=["운임손실(원)", "평균운임(원)"])
plt.xticks(f_fare["연도"]);
```

### 기본운임을 올렸을 때 향후 5년간(22-26년) 1인당 운임비용 구하기

- 기본운임 300원 인상 가정

#### 기본운임료 인상(+300원)시 예측


```python
f_fare_chp = f_fare_ch.copy()
```


```python
f_fare_chp["변경된 기본운임(원)"] = f_fare_chp["기본운임(원)"]+300
```

##### 기본운임료 변화율이 적용된 운임비용 구하기


```python
# 기본운임 변화율 구하기
ch_ratio = f_fare_chp["변경된 기본운임(원)"][0] / f_fare_chp["기본운임(원)"][0]
ch_ratio
```


```python
# 기본운임 변화율 적용하기 (기존 평균운임 * 기본운임 변화율)
for col in range(len(f_fare_chp)):
    f_fare_chp["평균운임(원)"].iloc[col] = f_fare_chp["평균운임(원)"].iloc[col] * ch_ratio
    f_fare_chp["운임손실(원)"].iloc[col] = f_fare_chp["수송원가(원)"].iloc[col] - f_fare_chp["평균운임(원)"].iloc[col]

# 운임손실 변화율 구하기
for col in range(1,len(f_fare)):
    f_fare_chp["운임손실 변화율(%)"].iloc[col] = (f_fare_chp["운임손실(원)"].iloc[col] - f_fare_chp["운임손실(원)"].iloc[col-1]) / f_fare_chp["운임손실(원)"].iloc[col-1]
    
f_fare_chp["운임손실 변화율(%)"] = round(f_fare_chp["운임손실 변화율(%)"]*100, 2)
```


```python
f_fare_chp
```

##### 시각화


```python
f_fare_chp.plot(x="연도", y=["운임손실(원)", "평균운임(원)"])
plt.xticks(f_fare_chp["연도"]);
```

### 연령상향과 기본료 인상으로 인해 발생되는 수익예측

#### 향후 5년 승차수 구하기

##### 2017~2021년 승차인원비율 구하기

- 승차인원 비율 : 승차인원 / 전체 인구수 - 0~4세(운임측정불가)
- 해당년도 승차인원 : (해당년도 인구수 - 해당년도의 0~4세(운임측정불가)) * 승차인원 비율

###### 인구수 데이터 불러오기


```python
sl = pd.read_csv(url, encoding='cp949')
```


```python
# 불필요한 컬럼 제거
sl = sl.drop(columns=['시나리오별(1)', '시도별(1)', '성별(1)'], axis=1)
```


```python
# 컬럼명 바꾸기
sl = sl.rename(columns = {'연령별(1)':'연령대'})
sl
```


```python
# tidy data 만들기
sl_melt = pd.melt(sl, id_vars=['연령대'], var_name='연도', value_name='인구수')
sl_melt
```


```python
# 연도 타입 object -> int
sl_melt["연도"] = sl_melt["연도"].astype("int")
sl_melt
```

###### 연도별 인원 구하기
- 우선 17년으로 작업 후 21년까지 반복문 돌리기


```python
# 2017년 자료만 가져오기
sl_17 = sl_melt[sl_melt["연도"] == 2017]
sl_17
```


```python
# 2017년 인구 총 계
sl_17_total = (sl_17[sl_17["연령대"] == "계"]).iloc[0, -1]
display(sl_17[sl_17["연령대"] == "계"])
sl_17_total
```


```python
# 2017년 서울시 만 65세 ~ 69세 인구 수 구하기 ('소계' 이용)

sl_17_6569sum = (sl_17[sl_17["연령대"] == "65 - 69세"]).iloc[0, -1]
display(sl_17[sl_17["연령대"] == "65 - 69세"])
display(sl_17_6569sum)
```


```python
# 2017년 서울시 만 0-4세 인구 수 구하기 ('소계' 이용)

sl_17_6569sum = (sl_17[sl_17["연령대"] == "0 - 4세"]).iloc[0, -1]
display(sl_17[sl_17["연령대"] == "0 - 4세"])
display(sl_17_6569sum)
```


```python
# 총 계, 65-69세 인구수 구하는 함수

def sl_yr(yr):
    
    # 해당연도 데이터프레임만 가져오기
    sl_yr = sl_melt[sl_melt["연도"] == yr]
    
    # 해당연도의 전체 인구 수 가져오기
    sl_yr_total = (sl_yr[sl_yr["연령대"] == "계"]).iloc[0, -1]
    
    # 해당연도의 0-4세 인구 수 가져오기
    sl_yr_baby = (sl_yr[sl_yr["연령대"] == "0 - 4세"]).iloc[0, -1]
    
    # 해당연도의 65-69세 인구 수 가져오기
    sl_yr_elder = (sl_yr[sl_yr["연령대"] == "65 - 69세"]).iloc[0, -1]
    
    # 해당연도의 65세 이상 인구 수 가져오기
    sl_yr_elder_all = (sl_yr[sl_yr["연령대"] == "65 - 69세"]).iloc[0, -1]
    sl_yr_elder_all += (sl_yr[sl_yr["연령대"] == "70 - 74세"]).iloc[0, -1]
    sl_yr_elder_all += (sl_yr[sl_yr["연령대"] == "75 - 79세"]).iloc[0, -1]
    sl_yr_elder_all += (sl_yr[sl_yr["연령대"] == "80세이상"]).iloc[0, -1]
    
    return sl_yr_total, sl_yr_baby, sl_yr_elder, sl_yr_elder_all
```


```python
# 반복문 돌리기 -> 17 ~ 21년 데이터만 추출하기

yr_list = []

for i in range(2017, 2022):
    yr_list.append(sl_yr(i))
    
yr_list
```


```python
yr_list[0][0]
```


```python
# 데이터 프레임으로 만들기

yr_pop = pd.DataFrame({'연도':[2017, 2018, 2019, 2020, 2021],
                        '전체인구수': [yr_list[i][0] for i in range (5)],
                        '0-4세인구수': [yr_list[i][1] for i in range (5)],
                        '65-69세인구수' : [yr_list[i][2] for i in range (5)],
                        '노인인구수' : [yr_list[i][3] for i in range (5)],})
yr_pop
```

###### 승차인원 구하기


```python
# 승차인원
carry = pd.read_csv(url_carry1, encoding="cp949") 
```


```python
# 서울만 승차인원 확인하기
carry_seoul = carry[carry["운영기관별(1)"] == '서울']
carry_seoul
```


```python
# 서울의 연도별 승차인원 구하기
yr_seoul_carry = pd.DataFrame({"연도":[2017, 2018, 2019, 2020, 2021], "승차인원(천명)":carry_seoul["승차인원 (천명/년)"].to_list()})
yr_seoul_carry["승차인원(천명)"] = yr_seoul_carry["승차인원(천명)"]*1000
yr_seoul_carry = yr_seoul_carry.rename(columns = {"승차인원(천명)":"승차인원"})
```


```python
#17-21년 인구 수 데이터와 승차인원 데이터 합치기
pop_1721 = pd.merge(yr_pop, yr_seoul_carry, how='left',on='연도')
pop_1721
```

###### 승차인원 비율 구하기

* 전체인구수 에서 0-4세의 경우 승차인원으로 파악이 안되므로 제외
* carry_p_mn : 승차인원비율


```python
pop_1721["승차인원비율"] = pop_1721["승차인원"] / (pop_1721["전체인구수"] - pop_1721['0-4세인구수'])
pop_1721
```


```python
# 승차인원비율의 평균구하기
carry_p_mn = pop_1721["승차인원비율"].mean()
carry_p_mn
```

##### 2017~2021년의 무임승차비율 구하기

* 무임승차데이터프레임 : nomoney
* 노인 무임승차인원 비율 : 노인무임승차수 / 노인인구수
* 해당년도 노인 무임승차인원 : 해당년도 노인인구수 * 노인무임승차인원 비율


```python
# 대상별 무임승차, 무임비용
nomoney = pd.read_csv(url_nomoney, encoding="cp949")
```


```python
# 무임승차데이터에서 컬럼명 변경
nomoney = nomoney.rename(columns = {"운영기관별(1)":"운영기관", "시점":"연도", "대상별(1)":"대상"})
```


```python
# 무임승차데이터에서 '대상' 컬럼에서 '노인'만 가져오기
nm_elderly = nomoney[nomoney["대상"] == "노인"].reset_index(drop=True)
nm_elderly
```


```python
# 운영기관에서 서울 지하철 데이터만 가져오기
seoul_subway = ['서울교통공사', '서울메트로 9호선(주)', '서울교통공사9호선운영부문', '우이 신설경전철(주)']
nm_elderly = nm_elderly[nm_elderly["운영기관"].isin(seoul_subway)]
display(nm_elderly)
nm_elderly.dtypes
```


```python
# 연도를 입력하면 무임승차수 출력되는 함수

def yr_nm(yr):
    nm_yr = nm_elderly[nm_elderly["연도"] == yr]
    nm_sum = nm_yr["무임승차 (천명)"].sum()
    
    return nm_sum *1000
```


```python
yr_nm(2017),yr_nm(2018),yr_nm(2019)
```

###### 무임승차수 구하기


```python
# 인구수 데이터프레임에 연도별 노인무임승차수 컬럼 만들|기
pop_1721["무임승차수"] = [yr_nm(yr) for yr in range(2017,2022)]
pop_1721
```

###### 무임승차비율 구하기


```python
# 무임승차비율 구하기 
# 무임승차비율 = 무임승차수 / 노인인구수
pop_1721["무임승차비율"] = pop_1721["무임승차수"] / pop_1721["노인인구수"] 
pop_1721
```


```python
# 무임승차인원비율의 평균구하기
old_carry_mn = pop_1721["무임승차비율"].mean()
old_carry_mn
```

##### 22-26 년도의 승차수 예측

* 해당연도의 승차인원 : (해당연도 인구수 - 아이 인구수) * 승차인원비율(carry_p_mn)
* 해당연도의 무임승차인원 : 해당연도 노인 수 * 무임승차비율(old_carry_mn)

###### 데이터 가져오기


```python
# 22-26년도 인구수 데이터 가져오기
pop_2226 = el_pop[['노인인구수','전체인구수','65-70세 노인인구수']]
```


```python
pop_melt
```

###### 22-26년 아이수 컬럼추가


```python
# 연도를 입력하면 0-4세 인구수 출력되는 함수

def yr_baby(yr):
    baby_yr = pop_melt[pop_melt["연도"] == yr]
    
     # 해당연도의 0-4세 인구 수 가져오기
    baby_pop = (baby_yr[baby_yr["연령대"] == "0 - 4세"]).iloc[0, -1]
    
    return baby_pop
```


```python
# 함수 잘 작동되는지 확인
yr_baby(2022)
```


```python
# 인구수 데이터프레임에 '0-4세인구수' 컬럼추가
pop_2226['0-4세인구수'] = [yr_baby(yr) for yr in range (2022,2027)]
pop_2226
```

###### 해당연도의 승차인원 컬럼추가

(해당연도 인구수 - 아이 인구수) * 승차인원비율(carry_p_mn)


```python
pop_2226['승차인원'] = (pop_2226['전체인구수'] - pop_2226['0-4세인구수']) * carry_p_mn
```


```python
pop_2226['승차인원'] = (pop_2226['전체인구수'] - pop_2226['0-4세인구수']) * carry_p_mn
```

#### 해당연도의 무임승차인원, 연령상향시 무임승차수 추가
해당연도의 무임승차인원 : 해당연도 노인 수 * 무임승차비율(old_carry_mn)


```python
pop_2226['무임승차인원'] = pop_2226['노인인구수'] * old_carry_mn
```


```python
pop_2226
```


```python
pop_2226['연령상향시 무임승차인원'] = (pop_2226['노인인구수'] - pop_2226['65-70세 노인인구수']) * old_carry_mn
```


```python
pop_2226
```

##### 승차인원, 무임승차인원 시각화


```python
# 기본 틀 만들기
fi, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

# 필요 데이터 그리기
ax[0].plot(pop_2226.index, pop_2226["승차인원"], label="승차인원", marker = 'o', color ='b')
ax[1].plot(pop_2226.index, pop_2226["무임승차인원"], label="무임승차인원", marker = 'o', color ='y')

# x축 눈금설정
ax[0].set_xticks(pop_2226.index)
ax[1].set_xticks(pop_2226.index)

# 범례적용하기
ax[0].legend(loc='best')
ax[1].legend(loc='best')

# 그리드 그리기
ax[0].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
ax[1].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')

```


```python
plt.plot(pop_2226.index, pop_2226["승차인원"], label="승차인원", marker = 'o', color ='b')
plt.plot(pop_2226.index, pop_2226["무임승차인원"], label="무임승차인원", marker = 'o', color ='y')
plt.xticks(pop_2226.index)

plt.xlabel('연도')
plt.ylabel('인원수')
plt.title('향후 승차수', fontsize=16)

# 범례 적용
plt.legend(loc='best')

# 그리드 그리기
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')

plt.show()
```

#### 상향 전후 운임손실 구하기

1인당 운임손실 * 승차인원


```python
# 상향전(노인:65세이상) 운임비용
f_fare_ch
```


```python
# 상향후(노인:65세이상) 운임손실값
f_fare_ch2
```


```python
pop_2226
```


```python
f_fare_ch["운임손실(원)"].iloc[0]
```

##### 연령상향 전 운임손실비용


```python
# 연령상향 전 총 운임손실비용
loss = pd.DataFrame({"연도" : [2022,2023,2024,2025,2026],
                     "운임손실(조원)": [(f_fare_ch["운임손실(원)"].iloc[i] * pop_2226["승차인원"].iloc[i] / 1e+12) for i in range(5)]})
```


```python
loss
```

##### 연령상향 후 운임손실비용


```python
# 연령상향 후 총 운임손실비용
loss_up = pd.DataFrame({"연도" : [2022,2023,2024,2025,2026],
                     "운임손실(조원)": [(f_fare_ch2["운임손실(원)"].iloc[i] * pop_2226["승차인원"].iloc[i] / 1e+12) for i in range(5)]})
```


```python
loss_up
```

##### 연령상향전후 운임손실차이 변화율


```python
# 연령상향 전후 운임손실차이
loss_dif = pd.DataFrame({"연도" : [2022,2023,2024,2025,2026],
                     "연령상향시(억원)": [((loss["운임손실(조원)"].iloc[i] - loss_up["운임손실(조원)"].iloc[i]) *10000) for i in range(5)]})

loss_dif
```


```python
# 연령상향시 운임손실차이 변화율
ratio = []
for i in range(5):
    ratio.append((loss_dif["연령상향시(억원)"].iloc[i] - loss_dif["연령상향시(억원)"].iloc[i-1])/loss_dif["연령상향시(억원)"].iloc[i-1])

loss_dif["연령상향시변화율(%)"] = ratio
loss_dif["연령상향시변화율(%)"] = round(loss_dif["연령상향시변화율(%)"]*100, 2)

# 이전년도 연령상향시 운임손실차액이 없는 2022년도 0처리
loss_dif["연령상향시변화율(%)"].iloc[0] = 0.0
```


```python
loss_dif
```

##### 시각화


```python
# 선그래프 그리기

plt.plot(loss["연도"], loss["운임손실(조원)"], label="상향전 운임손실비용", marker = 'o', color ='g')
plt.plot(loss_up["연도"], loss_up["운임손실(조원)"], label="상향후 운임손실비용", marker = 'o', color ='r')
plt.xticks(loss_up["연도"])

plt.xlabel('연도')
plt.ylabel("비용(조원)")
plt.title('노인연령 상향 전후 운임손실비용', fontsize=16)

# 범례 적용
plt.legend(loc='best')

# 그리드 그리기
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')

plt.show()
```


```python
# 바 그래프 그리기
bar_width = 0.35
alpha = 0.5

p1 = plt.bar(loss["연도"] - bar_width/2, loss["운임손실(조원)"], 
             bar_width, 
             color='b', 
             alpha=alpha,
             label="상향 전")

p2 = plt.bar(loss_up["연도"] + bar_width/2, loss_up["운임손실(조원)"], 
             bar_width, 
             color='r', 
             alpha=alpha,
             label="상향 후")

plt.title('노인연령 상향 전후 운임손실비용', fontsize=16)
plt.ylabel('비용(조원)', fontsize=12)
plt.xlabel('연도', fontsize=12)
plt.xticks(loss["연도"], fontsize=10)

plt.legend()

# 숫자 넣는 부분
for rect in p1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, round(height, 2), ha='center', va='bottom', size = 10)

for rect in p2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, round(height, 2), ha='center', va='bottom', size = 10)

plt.show()
```

#### 기본운임 인상시 운임손실 구하기

##### 기본운임 인상시 운임손실비용


```python
# 기본운임 인상시 총 운임손실비용
loss_f_up = pd.DataFrame({"연도" : [2022,2023,2024,2025,2026],
                         "운임손실(조원)": [(f_fare_chp["운임손실(원)"].iloc[i] * pop_2226["승차인원"].iloc[i] /1e+12) for i in range(5)]})
```


```python
loss_f_up
```

##### 기본운임 전후 운임손실차이


```python
# 기본운임 전후 운임손실차이
loss_dif["기본운임인상시(억원)"] = [((loss["운임손실(조원)"].iloc[i] - loss_f_up["운임손실(조원)"].iloc[i]) *10000) for i in range(5)]
```


```python
loss_dif
```


```python
# 기본운임인상시 운임손실변화율
ratio = []
for i in range(5):
    ratio.append((loss_dif["기본운임인상시(억원)"].iloc[i] - loss_dif["기본운임인상시(억원)"].iloc[i-1])/loss_dif["기본운임인상시(억원)"].iloc[i-1])

loss_dif["기본운임인상시변화율(%)"] = ratio
loss_dif["기본운임인상시변화율(%)"] = round(loss_dif["기본운임인상시변화율(%)"]*100,2)

# 이전년도 연령상향시 운임손실차액이 없는 2022년도 0처리
loss_dif["기본운임인상시변화율(%)"].iloc[0] = 0.0
```


```python
loss_dif
```

##### 기본운임 인상 전후 운임손실 시각화


```python
# 바 그래프 그리기
bar_width = 0.35
alpha = 0.5

p1 = plt.bar(loss["연도"] - bar_width/2, loss["운임손실(조원)"], 
             bar_width, 
             color='b', 
             alpha=alpha,
             label="인상 전")

p2 = plt.bar(loss_f_up["연도"] + bar_width/2, loss_f_up["운임손실(조원)"], 
             bar_width, 
             color='r', 
             alpha=alpha,
             label="인상 후")

plt.title('기본운임 인상 전후 운임손실비용', fontsize=16)
plt.ylabel('비용(조원)', fontsize=12)
plt.xlabel('연도', fontsize=12)
plt.xticks(loss["연도"], fontsize=10)

plt.legend()

# 숫자 넣는 부분
for rect in p1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, round(height, 2), ha='center', va='bottom', size = 10)

for rect in p2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height, round(height, 2), ha='center', va='bottom', size = 10)

plt.show()
```

#### 연령상향 전후, 기본요금인상 전후 운임손실변화율 차이 시각화


```python
loss_dif
```


```python
# 선그래프 그리기

plt.plot(loss_dif["연도"], loss_dif[["연령상향시변화율(%)", "기본운임인상시변화율(%)"]], label = ["연령 상향시", "기본요금 인상시"], marker = 'o')
# plt.plot(loss_up["연도"], loss_up["운임손실"], label="상향후 운임손실비용", marker = 'o', color ='r')
plt.xticks(loss_dif["연도"])

plt.xlabel('연도')
plt.ylabel("퍼센트(%)")
plt.title('운임손실비용 차이 변화율', fontsize=16)

# 범례 적용
plt.legend(loc='best')

# 그리드 그리기
plt.grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')

plt.show()
```

#### 연령상향과 기본료 인상 운임비용 비교 시각화하기


```python
# 기본 틀 만들기
fig1, axs1 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig2, axs2 = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
```

##### 노인연령상향시 운임비용


```python
# 노인연령상향시 운임비용 그래프 그리기
axs1[0].plot(f_fare_ch['연도'], f_fare_ch["운임손실(원)"], label="운임손실", marker = 'o')
axs1[0].plot(f_fare_ch['연도'], f_fare_ch["평균운임(원)"], label="평균운임", marker = 'o')
axs1[1].plot(f_fare_ch2['연도'], f_fare_ch2["운임손실(원)"], label="운임손실", marker = 'o')
axs1[1].plot(f_fare_ch2['연도'], f_fare_ch2["평균운임(원)"], label="평균운임", marker = 'o')

fig1
```


```python
# 그리드 그리기
axs1[0].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
axs1[1].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')


# x축 눈금설정
axs1[0].set_xticks(f_fare_ch["연도"])
axs1[1].set_xticks(f_fare_ch["연도"])

fig1
```


```python
# 그래프의 이름 붙이기
axs1[0].set_title("노인연령 상향 전")
axs1[1].set_title("노인연령 상향 후")

# 범례적용하기
axs1[0].legend(loc='upper left')

# 축이름 설정
axs1[0].set_xlabel("연도")
axs1[1].set_xlabel("연도")
axs1[0].set_ylabel("1인당비용(원)")

# 기본 틀 제목 설정
fig1.suptitle("연령상향 전후 1인당 운임비용 예측", fontsize=15)

fig1
```

##### 기본운임상향시 운임비용


```python
# 필요 데이터 그리기
axs2[0].plot(f_fare_ch['연도'], f_fare_ch["운임손실(원)"], label="운임손실", marker = 'o')
axs2[0].plot(f_fare_ch['연도'], f_fare_ch["평균운임(원)"], label="평균운임", marker = 'o')
axs2[1].plot(f_fare_chp['연도'], f_fare_chp["운임손실(원)"], label="운임손실", marker = 'o')
axs2[1].plot(f_fare_chp['연도'], f_fare_chp["평균운임(원)"], label="평균운임", marker = 'o')

fig2
```


```python
# 그리드 그리기
axs2[0].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')
axs2[1].grid(True, axis='y', color='gray', alpha=0.5, linestyle='--')

# x축 눈금설정
axs2[0].set_xticks(f_fare_ch["연도"])
axs2[1].set_xticks(f_fare_chp["연도"])

fig2
```


```python
# 그래프의 이름 붙이기
axs2[0].set_title("기본요금")
axs2[1].set_title("300원인상")

# 범례적용하기
axs2[0].legend(loc='upper left')

# 축이름 설정
axs2[0].set_xlabel("연도")
axs2[1].set_xlabel("연도")
axs2[0].set_ylabel("1인당비용(원)")

# 기본 틀 제목 설정
fig2.suptitle("기본요금인상 전후 1인당 운임비용 예측", fontsize=15)

fig2
```

# 지하철 운영의 실질적인 문제

## 지하철 운영 적자의 진짜 원인은?

### 서울교통공사 재무제표 - 손익계산서 분석 


```python
# 2021년 손익계산서

is_data = pd.read_csv(url_2021, encoding="cp949")
# 서울교통공사까지 표기
is_21 = is_data[:5].copy()
# 불필요 행,열 삭제
is_21 = is_21.drop([2,3], axis=0)
is_21 = is_21.drop(["기관별(1)"], axis=1)
# 필요 데이터 컬럼 수집
is_21_data = is_21[["2021", "2021.7","2021.11","2021.22","2021.23", "2021.48", "2021.104"]].copy()
# 컬럼명 변경
is_21_data.columns = ['매출액', '운수사업수익', "매출원가", "매출총이익",
                      "판매비와 관리비",'영업이익', "당기순이익"]
# 불필요 행 삭제
is_21_data = is_21_data.drop([0,1], axis=0)
# 연도 추가
is_21_data["연도"] = 2021

is_21_data
```


```python
# 2017 ~ 2020 손익계산서 항목 생성 함수

def is_data_f(url, i):
    '''
    손익계산서에서 매출액, 운수수익, 매출원가, 매출총이익, 판관비, 영업손익, 당기손익을 표현하는 함수
    '''
    df = pd.read_csv(url, encoding="cp949")
    # 서울교통공사까지 표기
    df = df[:5].copy()
    # 불필요 행,열 삭제
    df = df.drop([2,3], axis=0)
    df = df.drop(["행정구역별(1)"], axis=1)
    # 필요 데이터 컬럼 수집
    df_data = df[[i, i+".1",i+".5", i+".27", i+".28", i+".52", i+".75"]].copy()
    # 컬럼명 변경
    df_data.columns = ['매출액', '운수사업수익', "매출원가", "매출총이익",
                      "판매비와 관리비",'영업이익', "당기순이익"]
    # 불필요 행 삭제
    df_data = df_data.drop([0,1], axis=0)
    # 연도 추가
    df_data["연도"] = i

    return df_data 
```


```python
# 2020년 손익계산서 산출
url_2020 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2020.csv"
i = str(2020)
is_20_data = is_data_f(url_2020, i)
```


```python
# 2019년 손익계산서 산출
url_2019 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2019.csv"
i = str(2019)
is_19_data = is_data_f(url_2019, i)
```


```python
# 2018년 손익계산서 산출
url_2018 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2018.csv"
i = str(2018)
is_18_data = is_data_f(url_2018, i)
```


```python
# 2017년 손익계산서 산출
url_2017 = "https://raw.githubusercontent.com/LimSeungMin/AIS8_task/main/%EB%8F%84%EC%8B%9C%EC%B2%A0%EB%8F%84%EA%B3%B5%EC%82%AC_%EC%9E%AC%EB%AC%B4%EC%83%81%ED%83%9C%ED%91%9C_%EC%86%90%EC%9D%B5%EA%B3%84%EC%82%B0%EC%84%9C_2017.csv"
i = str(2017)
is_17_data = is_data_f(url_2017, i)
```


```python
# 병합
is_data = pd.concat([is_21_data, is_20_data, is_19_data, is_18_data, is_17_data])
```


```python
# 형변환
is_data["연도"] = is_data["연도"].astype(int)
is_data = is_data.sort_values(by=["연도"])
is_data["매출액"] = pd.to_numeric(is_data["매출액"], errors='coerce')
is_data["운수사업수익"] = pd.to_numeric(is_data["운수사업수익"], errors='coerce')
is_data["매출원가"] = pd.to_numeric(is_data["매출원가"], errors='coerce')
is_data["매출총이익"] = pd.to_numeric(is_data["매출총이익"], errors='coerce')
is_data["판매비와 관리비"] = pd.to_numeric(is_data["판매비와 관리비"], errors='coerce')
is_data["영업이익"] = pd.to_numeric(is_data["영업이익"], errors='coerce')
is_data["당기순이익"] = pd.to_numeric(is_data["당기순이익"], errors='coerce')
```


```python
# 컬럼 설명

# 기준 : 원
# 매출액 : 운영사업수익(운수사업수익 포함) + 수탁사업수익
# 운수사업수익 : 매출액 내 운수사업 수익(서울교통공사는 운수업)
# 매출총이익 : 매출액 - 매출원가
# 영업이익 : 매출총이익 - 판매비와 관리비
# 당기순이익 : 영업이익 + 영업 외 수익 - 영업 외 비용 - 법인세
# 영업이익률 = (영업이익/매출액) * 100

is_data = is_data.reset_index(drop=True)
is_data["영업이익률(%)"] = ((is_data["영업이익"] / is_data["매출액"]) * 100).round(2)
```

### 무임승차 데이터 분석


```python
# 데이터 불러오기
data_cal = pd.read_csv(url_cal, encoding="cp949")
```


```python
# 필요 데이터셋 정리
cal = data_cal[["시점", "계.1", "노인.1"]].copy()

# 컬럼명 변경 및 index 초기화
cal.columns = ["연도", "총_무임승차비용", "노인_무임승차비용"]
cal = cal.drop([0], axis=0)
cal = cal.reset_index(drop=True)

# 데이터 단위 변경(백만원 -> 원)
cal["연도"] = cal["연도"].astype(int)
cal = cal.sort_values(by=["연도"])
cal["총_무임승차비용"] = (cal["총_무임승차비용"].astype(float)) * 1000000
cal["노인_무임승차비용"] = (cal["노인_무임승차비용"].astype(float)) * 1000000
```


```python
# 데이터 병합
hy = is_data.merge(cal, on="연도", how="left")
#index 설정"연도"
hy = hy.set_index(["연도"])
hy
```


```python
# 그래프 출력 시 에러 무시
import warnings
warnings.filterwarnings("ignore")

# 그래프 그릴 때 한글 깨짐 방지 설정
import os

# Mac OS의 경우와 그 외 OS의 경우로 나누어 설정
if os.name == 'posix':
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="Malgun Gothic")

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
```


```python
# 손익계산서 분석.
# 매출액, 매출원가, 매출총이익 비교
plt.plot(hy["매출액"], color='#E69F00', marker='o', label="매출액")
plt.plot(hy["매출원가"], color='#56B4E9',  marker='o', label="매출원가")
plt.plot(hy["매출총이익"], color='#009E73', marker='o', label="매출총이익")
plt.grid(True, axis='y')
plt.title('매출액, 매출원가, 매출총이익 비교')
plt.xlabel('연도')
plt.ylabel('금액')
plt.xticks([2017, 2018, 2019, 2020, 2021])
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
```


```python
# 손익계산서 계정과목 분석
# 매출액, 영업이익, 당기순이익, 영업이익률

x = [2017, 2018, 2019, 2020, 2021]
y1= hy["매출액"]
y2= hy["영업이익"]
y3= hy["당기순이익"]
y4= hy["영업이익률(%)"]

ax1 = plt.subplot(2, 2, 1)               
plt.bar(x, y1, color='#E69F00', label="매출액")
plt.title('매출액')
plt.xlabel('연도')
plt.ylabel('금액')

ax2 = plt.subplot(2, 2, 2)              
plt.bar(x, y2, color='#56B4E9', label="영업이익")
plt.title('영업이익')
plt.xlabel('연도')
plt.ylabel('금액')

ax3 = plt.subplot(2, 2, 3)              
plt.bar(x, y3, color='#CC79A7', label="당기순이익")
plt.title('당기순이익')
plt.xlabel('연도')
plt.ylabel('금액')

ax4 = plt.subplot(2, 2, 4)              
plt.plot(x, y3,color='#009E73', marker='o', label="영업이익률" )
plt.title('영업이익률')
plt.xlabel('연도')
plt.ylabel('금액')


plt.tight_layout()
plt.show()
```


```python
# 손익계산서 계정과목 분석
# 매출액 대비 운수사업수익 비중 -> 5년간 감소 추세. 운수사업 이외 위탁&수탁&부대사업을 통한 수익 증가

t_rate= ((hy.운수사업수익 / hy.매출액) * 100).round(2)
x = [2017, 2018, 2019, 2020, 2021]
y1= hy["매출액"]
y5= hy["운수사업수익"]

plt.subplot(2, 1, 1)       
plt.bar(x, y1, color='#E69F00', label="매출액")
plt.bar(x, y5, color='#56B4E9', label="운수사업수익")
plt.title('매출액 & 운수사업수익 현황')
plt.rcParams['figure.figsize'] = (6, 3)
plt.legend(bbox_to_anchor=(1, 1))
plt.ylabel('금액')

plt.subplot(2, 1, 2)              
plt.plot(t_rate, color='#CC79A7', marker='o', label="운수사업수익비중")
plt.title('운수사업수익 비중')
plt.rcParams['figure.figsize'] = (6, 3)
plt.xlabel('연도')
plt.xticks([2017, 2018, 2019, 2020, 2021])
plt.grid(True, axis='y')
plt.ylabel('비율(%)')

plt.tight_layout()
plt.show()
```


```python
# 왜 적자인가?
# 비용 분석
# 매출액, 매출원가, 판관비를 통해 비용 지출 현황 파악

hy_4 = hy[["매출액", "매출원가", "판매비와 관리비"]]
hy_4.plot(kind="barh", title="매출액, 매출원가, 판관비 비교")
plt.xlabel("금액")
plt.ylabel("연도")
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
```


```python
px.bar(hy_4, title="매출액, 매출원가, 판관비 현황")
```


```python
# 왜 적자인가?
# 매출액 대비 매출원가 비중 -> 코로나 시기 급 상승 이후 하락 추세이나, 지속적인 매출액 < 매출원가 상황.
t_rate2 = ((hy["매출원가"] / hy["매출액"]) * 100).round(2)
t_rate3 = ((hy["판매비와 관리비"] / hy["매출액"]) * 100).round(2)
x = [2017, 2018, 2019, 2020, 2021]

plt.subplot      
plt.plot(x, t_rate2, color='#E69F00', marker='o', label="매출원가 비중")
plt.plot(x, t_rate3, color='#56B4E9', marker='o', label="판관비 비중")
plt.title('매출액 대비 매출원가, 판관비 비중 현황')
plt.grid(True, axis='y')
plt.xticks([2017, 2018, 2019, 2020, 2021])
plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('연도')
plt.ylabel('금액')

plt.show()
```

### 가설설정

1. 무임승차액과 매출액은 상관이 있을 것이다.
2. 무임승차액과 영업이익은 상관이 있을 것이다.



```python
# 시각화를 위한 컬럼명 영어 변경
hy_e = hy.copy()
hy_e.columns = ['revenue_sales', 'revenue_transportation', "cost_of_sales", "gross_profit",
                "selling_expenses",'operating_income', "profit", "rate_operating_income", "total_free_ride", "senior_free_ride"]
```


```python
# n이 너무 작아 상관 계수 분석 불가.
hy_e.corr()
```

#### 무임승차액과 매출액은 상관이 있을것 이다


```python
# 귀무가설 : 총무임승차액은 매출액과 상관이 없다
# 대립가설 : 총무임승차액은 매출액과 상관이 있다

# p-val > 0.05   => 결론 유보
# n이 너무 작아 유의미한 결론 추론 불가
pg.corr(hy_e.total_free_ride, hy_e.revenue_sales)
```


```python
# 귀무가설 : 노인무임승차액은 운수사업수익과 상관이 없다
# 대립가설 : 노인무임승차액은 운수사업수익과 상관이 있다

# p-val > 0.05   => 결론 유보
# n이 너무 작아 유의미한 결론 추론 불가
pg.corr(hy_e.revenue_transportation, hy_e.senior_free_ride)
```


```python
# 매출액, 운수사업수익, 총_무임승차비용, 노인_무임승차비용 비교

t_rate= ((hy.운수사업수익 / hy.매출액) * 100).round(2)
x = [2017, 2018, 2019, 2020, 2021]
y1 = hy["매출액"]
y5 = hy["운수사업수익"]
y6 = hy["총_무임승차비용"]
y7 = hy["노인_무임승차비용"]
     
plt.bar(x, y1, color='#E69F00', label="매출액")
plt.bar(x, y5, color='#56B4E9', label="운수사업수익")
plt.bar(x, y6, color='#009E73', label="총_무임승차비용")
plt.bar(x, y7, color='#F0E442', label="노인_무임승차비용")
plt.title('무임승차액 & 매출액 현황')
plt.rcParams['figure.figsize'] = (6, 3)
plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('연도')
plt.ylabel('금액')

plt.tight_layout()
plt.show()
```


```python
# 운수사업수익 대비 노인무임승차비용 비중
y8 = ((hy["노인_무임승차비용"] / hy["운수사업수익"]) * 100).round(2)

plt.plot(y8, color='#CC97A7', marker='o', label="노인 무임승차")
plt.grid(True, axis='y')
plt.title('운수사업수익 대비 노인무임승차비용 비중')
plt.xlabel('연도')
plt.xticks([2017, 2018, 2019, 2020, 2021])
plt.ylabel('%')
plt.legend()

plt.show()
```

#### 무임승차액과 영업이익은 상관이 있을것이다.


```python
# 귀무가설 : 총_무임승차액은 영업이익과 상관이 없다
# 대립가설 : 총_무임승차액은 영업이익과 상관이 있다

# p-val > 0.05   => 결론 유보
# n이 너무 작아 유의미한 결론 추론 불가
pg.corr(hy_e.total_free_ride, hy_e.operating_income)
```


```python
# 귀무가설 : 노인무임승차액은 영업이익과 상관이 없다
# 대립가설 : 노인무임승차액은 영업이익과 상관이 있다

# p-val > 0.05   => 결론 유보
# n이 너무 작아 유의미한 결론 추론 불가
pg.corr(hy_e.senior_free_ride, hy_e.operating_income)
```


```python
hy_6 = hy[["매출액", "영업이익", "총_무임승차비용", "노인_무임승차비용"]]

hy_6.plot(kind="barh", title="무임승차액과 영업이익 비교")
plt.xlabel("금액")
plt.ylabel("연도")
plt.legend(bbox_to_anchor=(1, 1))

plt.show()
```


```python
# 매출액, 영업이익, 총_무임승차비용, 노인_무임승차비용 비교

t_rate= ((hy.운수사업수익 / hy.매출액) * 100).round(2)
x = [2017, 2018, 2019, 2020, 2021]
y1 = hy["매출액"]
y5 = hy["영업이익"]
y6 = hy["총_무임승차비용"]
y7 = hy["노인_무임승차비용"]
     
plt.bar(x, y1, color='#E69F00', label="매출액")
plt.bar(x, y5, color='#56B4E9', label="영업이익")
plt.bar(x, y6, color='#009E73', label="총_무임승차비용")
plt.bar(x, y7, color='#F0E442', label="노인_무임승차비용")
plt.title('무임승차액 & 영업이익 현황')
plt.rcParams['figure.figsize'] = (6, 3)
plt.legend(bbox_to_anchor=(1, 1))
plt.xlabel('연도')
plt.ylabel('금액')

plt.tight_layout()
plt.show()
```


```python
hy[["영업이익", "노인_무임승차비용"]].plot(figsize=(10, 4), secondary_y="노인_무임승차비용")
plt.ylabel("금액")
```

##### 가설 검정 실패. 비모수 가정을 적용?


```python
# 비모수가정, mwu
pg.mwu(hy_e.total_free_ride , hy_e.operating_income)
```


```python
# 비모수가정, wilcoxon
pg.wilcoxon(hy_e.total_free_ride , hy_e.operating_income)
```


```python
# 켄달 분석 적용
pg.corr(hy_e.total_free_ride , hy_e.operating_income , method="kendall")
```

# 노인 무임승차의 긍정적인 측면

## 노인 무임승차 연령 상향의 불이익 

### 지역별 65세 인구 수 


```python
df_sub = pd.read_csv(url_sub,encoding='cp949')
df_sub.head()
```


```python
df_sub.info()
```


```python
# 시스템 환경에 따른 기본 폰트명을 반환하는 함수

def get_font_family() :
    """
    시스템 환경에 따른 기본 폰트명을 반환하는 함수
    """
    import platform
    system_name = platform.system()
    
    if system_name == 'Darwin' :
        font_family = 'AppleGothic'
    elif system_name == 'Windows' :
        font_family = "Malgun Gothic"
    else :
        #Linux(colab)
        !apt-get install fonts-nanum -qq  > /dev/null
        !fc-cache -fv
        
        import matplotlib as mpl
        mpl.font_manager._rebuild()
        findfont = mpl.font_manager.fontManger.findfont
        mpl.font_manager.findfont = findfont
        mpl.backends.backend_agg.findfont = findfont
        
        font_family = "NanumBarunGothic"
    return font_family

#폰트설정
plt.rc("font", family = get_font_family())
#마이너스폰트 설정
plt.rc("axes", unicode_minus=False)
```

#### 지역별 65세 인구 수 전처리하기


```python
# 불필요한 컬럼 제거
df_sub = df_sub.drop(columns=['성별', '세대구성별'], axis=1)
df_sub.head()
```

* 지역별 지하철 이용률 참고자료
* 2020년도_노인실태조사_보고서(보건복지부) -> 〈부표 Ⅲ-11-34〉 노인의 시･도별 외출할 때 주로 이용하는 교통수단


```python
# 컬럼명 바꾸기

df_sub = df_sub.rename(columns = {'행정구역별(시군구)':'지역', '시점':'연도'})
df_sub.head(2)
```


```python
df_sub['지하철 이용 노인수'] = df_sub['노인인구'] * df_sub['지하철 이용률']
df_sub.head()
```

## 경로무임승차제도 폐지 시 외부활동 자제인원 
* 지하철 경로무임승차제도 폐지 시 43.8%가 외부활동을 자제할 것으로 조사된 바 있음. 
* <노인 교통이용 요금제도 개선방안 연구: 지하철 무임승차를 중심으로> 참고


```python
df_sub['제도 폐지 시 외부활동 자제인원'] = df_sub['지하철 이용 노인수'] * 0.438
df_sub.head()
```

## 지역별 노인 자살생각자 수 / 자살 시도자 수 추정


```python
df_suicide = pd.read_csv(url_sui,encoding='cp949')
df_suicide.head()
```


```python
# 컬럼명 바꾸기

df_suicide = df_suicide.rename(columns = {'행정구역별(시군구)':'지역', '시점':'연도'})
df_suicide.head()
```


```python
df_suicide['자살 생각자 수'] = df_sub['노인인구'] * df_suicide['자살 생각률']
df_suicide.head()
```


```python
df_suicide['자살 시도자 수'] = df_suicide['자살 생각자 수'] * df_suicide['자살 시도율']
df_suicide.head()
```


```python
new_suicide = df_suicide.drop([0,1])
new_suicide.head()
```


```python
new_suicide.shape
```

### 지역별 실제 자살자 수


```python
suicide = pd.read_csv(url_suicide,encoding='cp949')
suicide.head(5)
```


```python
suicide = suicide.drop(columns=['사망원인별(104항목)', '성별'], axis=1)
suicide.head(15)
```


```python
suicide_65 = suicide[suicide["연령"] == '65세 이상'].reset_index()
suicide_65.head()
```


```python
suicide_65 = suicide_65.drop(columns=['index'], axis=1)
suicide_65.head()
```


```python
suicide_65.shape
```


```python
suicide_65 = suicide_65.rename(columns = {'시도별':'지역', '시점':'연도', '사망자수 (명)':'사망자수'})
suicide_65.head(2)
```

### 자살 시도자 중 자살 실현율 구하기


```python
new_suicide['자살 실현율'] = suicide_65['사망자수'] / new_suicide['자살 시도자 수']
new_suicide.head()
```

### 지하철 경로무임승차제도 폐지에 따른 노인 자살 시도자 증가


```python
df_sub['지하철 경로무임승차제도 폐지에 따른 노인 자살 시도자 증가'] = df_sub['제도 폐지 시 외부활동 자제인원'] * df_sub['외부활동을 자제하게 될 인원의 자살시도 증가율']
df_sub.head()
```


```python
df_sub['지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현'] = df_sub['지하철 경로무임승차제도 폐지에 따른 노인 자살 시도자 증가'] * new_suicide['자살 실현율']
df_sub.head()
```

### 자살에 따른 사회적 편익 산출


```python
df_sub.fillna(0, inplace=True) 
df_sub = df_sub.drop([0,1])
df_sub.head()
```


```python
df_sub = df_sub.copy().replace([np.inf], np.nan) # inf, -inf를 nan으로 대체
df_sub.fillna(0, inplace=True) 
df_sub.head(2)
```


```python
df_sub['지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현'].sum()
```


```python
df_sub20 = df_sub[df_sub["연도"] == 2020]
df_sub20.head(2)
```


```python
df_sub21 = df_sub[df_sub["연도"] == 2021]
df_sub21.head(2)
```


```python
df_sub20['자살 사회비용 감소편익'] = df_sub20['지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현'] * 409000000
df_sub20.head(2)
```


```python
df_sub21['자살 사회비용 감소편익'] = df_sub21['지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현'] * 409000000
df_sub21.head(2)
```


```python
df_sub['자살 사회비용 감소편익'] = df_sub['지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현'] * 409000000
df_sub
```

* 보건사회연구원에 따르면 자살의 경제적 손실 추산 결과 1인당 4억900만원의 비용이 발생한다.

* 출처 : 시사위크 http://www.sisaweek.com/news/articleView.html?idxno=202786
* https://www.yna.co.kr/view/AKR20230213114000530
* https://www.asiae.co.kr/article/2023021316401924763


```python
df_sub20['자살 사회비용 감소편익'].sum()
```


```python
df_sub21['자살 사회비용 감소편익'].sum()
```

* 2020년 노인 무임승차 폐지 시 노인의 자살 사회비용 감소편익은 525억 9720만원이다.
* 2021년 노인 무임승차 폐지 시 노인의 자살 사회비용 감소편익은 594억 4175만원이다.

## 노인의 우울증상 사회비용 감소편익


```python
df_dep = pd.read_csv(url_dep,encoding='cp949')
df_dep.info()
```


```python
df_dep
```


```python
df_dep['외부활동 적은 집단의평균 우울증세 경험률'] = df_dep['외부활동 적은 집단의평균 우울증세 경험률'].str.replace("%", "")
df_dep['외부활동 많은 집단의평균 우울증세 경험률'] = df_dep['외부활동 많은 집단의평균 우울증세 경험률'].str.replace("%", "")
```


```python
df_dep['외부활동 적은 집단의평균 우울증세 경험률'] = df_dep['외부활동 적은 집단의평균 우울증세 경험률'].astype('float64')
df_dep['외부활동 많은 집단의평균 우울증세 경험률'] = df_dep['외부활동 많은 집단의평균 우울증세 경험률'].astype('float64')
```


```python
df_dep['외부활동 적은 집단의평균 우울증세 경험률'] = df_dep['외부활동 적은 집단의평균 우울증세 경험률'] / 100
df_dep['외부활동 많은 집단의평균 우울증세 경험률'] = df_dep['외부활동 많은 집단의평균 우울증세 경험률'] / 100
```


```python
df_dep.head(2)
```


```python
df_dep.dtypes
```


```python
df_dep['우울증세 경험률 차이'] = df_dep['외부활동 적은 집단의평균 우울증세 경험률'] - df_dep['외부활동 많은 집단의평균 우울증세 경험률']
df_dep.head(5)
```

### 우울증세 경험률 차이 가설검정
* 가설 : 외부활동이 적은 집단과 외부활동이 많은 집단의 우울증세 경험확률 차이가 클 것이다.
* 귀무가설 : 두 집단의 평균에 우울증세 경험확률 차이가 존재하지 않는다.
* 대립가설 : 두 집단의 평균에 우울증세 경험확률 차이가 존재한다.


```python
import pingouin as pg
few = df_dep['외부활동 적은 집단의평균 우울증세 경험률']
many =  df_dep['외부활동 많은 집단의평균 우울증세 경험률']
pg.ttest(few, many, paired = True)
```


```python
df_sub20 = df_sub20.reset_index().drop(columns=['index'], axis=1)
df_sub21 = df_sub21.reset_index().drop(columns=['index'], axis=1)
```


```python
df_sub20 = df_sub20.drop([7])
df_sub21 = df_sub21.drop([7])
```


```python
df_sub20 = df_sub20.reset_index()
df_sub21 = df_sub21.reset_index()
```


```python
df_sub20 = df_sub20.drop(columns=['index'], axis=1)
df_sub21 = df_sub21.drop(columns=['index'], axis=1)
```


```python
df_sub20['경로무임승차 폐지 시 우울증세 증가 인원'] = df_sub20['제도 폐지 시 외부활동 자제인원'] * df_dep['우울증세 경험률 차이']
df_sub20.reset_index().drop(columns=['index'], axis=1).head(2)
```


```python
df_sub21['경로무임승차 폐지 시 우울증세 증가 인원'] = df_sub21['제도 폐지 시 외부활동 자제인원'] * df_dep['우울증세 경험률 차이']
df_sub21.head()
```

### 노인의 우울증상 사회비용 감소편익 산출
* 참고자료 : 건강보험 심사평가원 > 최근 5년(2017~2021년) 우울증과 불안장애 진료현황 분석
* 2020년 우울증 1인당 진료비 : 532,190원
* 2021년 우울증 1인당 진료비 : 564,712원

### 2020년 무임승차 폐지 시 우울증 사회비용 감소편익


```python
df_sub20['무임승차 폐지 시 우울증 감소편익'] = df_sub20['경로무임승차 폐지 시 우울증세 증가 인원'] * 532190
df_sub20.head(2)
```


```python
df_sub20['무임승차 폐지 시 우울증 감소편익'].sum() 
```


```python
df_sub21['무임승차 폐지 시 우울증 감소편익'] = df_sub21['경로무임승차 폐지 시 우울증세 증가 인원'] * 564712
df_sub21
```


```python
df_sub21['무임승차 폐지 시 우울증 감소편익'].sum() 
```


```python
df_sub_year = df_sub.groupby("연도").sum("지하철 이용 노인수").reset_index()
plt.plot(df_sub_year['연도'], df_sub_year["지하철 이용 노인수"], marker = 'o', markersize = 10, markerfacecolor = 'blue',
        linestyle = '-.')
plt.title("전국 연도별 지하철 이용 노인수")
plt.xticks([2020,2021])
plt.xlabel("연도")
plt.ylabel("지하철 이용 노인수")
plt.grid(axis = 'y')
plt.show()


# x라벨, y라벨

```


```python
df_sub_year = df_sub.groupby("연도").sum("제도 폐지 시 외부활동 자제인원").reset_index()
plt.plot(df_sub_year['연도'], df_sub_year["제도 폐지 시 외부활동 자제인원"], marker = 'o', markersize = 10, markerfacecolor = 'red',
        linestyle = '-.')
plt.title("전국 연도별 제도 폐지 시 외부활동 자제인원 노인수")
plt.xticks([2020,2021])
plt.xlabel("연도")
plt.ylabel("외부활동 자제인원 수")
plt.grid(axis = 'y')
plt.show()
```

* 전국에 제도 폐지 시 외부활동을 자제하겠다는 응답을 보인 노인수가 2020년보다 2021년에 더 많다.


```python
df_sub_year = df_sub.groupby("연도").sum("지하철 경로무임승차제도 폐지에 따른 노인 자살 시도자 증가").reset_index()
plt.plot(df_sub_year['연도'], df_sub_year["지하철 경로무임승차제도 폐지에 따른 노인 자살 시도자 증가"], marker = 'o', markersize = 10, markerfacecolor = 'yellow',
        linestyle = '-.')
plt.title("전국 지하철 무임승차 제도 폐지 시 증가한 자살 시도 노인")
plt.xticks([2020,2021])
plt.xlabel("연도")
plt.ylabel("자살 시도 증가수")
plt.grid(axis = 'y')
plt.show()
```

* 전국에 지하철 무임승차 제도 폐지 시 2020년보다 2021년에 자살 시도한 노인이 증가했다. 


```python
df_sub_year = df_sub.groupby("연도").sum("지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현").reset_index()
plt.plot(df_sub_year['연도'], df_sub_year["지하철 무임승차 폐지를 가정할 때 발생하는 노인 자살실현"], marker = 'o', markersize = 10, markerfacecolor = 'green',
        linestyle = '-.')
plt.title("제도 폐지 시 자살을 실현할 노인의 수")
plt.xticks([2020,2021])
plt.xlabel("연도")
plt.ylabel("자살 실현 노인수")
plt.grid(axis = 'y')
plt.show()
```

* 제도 폐지 시 자살을 실현하는 노인 수를 추정한 결과 2020년보다 2021년에 더 많은 것으로 나타났다.

* 위 결과를 정리하면, 연도가 지남에 따라서 지하철을 무임으로 이용하는 노인이 점점 많아지고 있는 추세임을 확인했으며
* 지하철 무임승차를 폐지했을 때 외부활동을 자제하는 노인들이 많아짐에 따라, 
* 자살을 시도하거나 실제로 자살을 실현하는 노인의 수도 연도가 지나면서 증가하고 있음을 확인할 수 있었다.


```python
df_sub20_region = df_sub20.groupby("지역").sum("무임승차 폐지 시 우울증 감소편익").reset_index()
plt.plot(df_sub20_region['지역'], df_sub20_region["무임승차 폐지 시 우울증 감소편익"], marker = 'o', markersize = 10, markerfacecolor = 'green',
        linestyle = '-.')
plt.title("2020 무임승차 폐지 시 우울증 감소편익 ")
plt.grid(axis = 'y')
plt.xticks(rotation = 60)
plt.ylabel("우울증 감소편익")
plt.show()
```


```python
df_sub21_region = df_sub21.groupby("지역").sum("무임승차 폐지 시 우울증 감소편익").reset_index()
plt.plot(df_sub21_region['지역'], df_sub21_region["무임승차 폐지 시 우울증 감소편익"], marker = 'o', markersize = 10, markerfacecolor = 'blue',
        linestyle = '-.')
plt.title("2021 무임승차 폐지 시 우울증 감소편익")
plt.grid(axis = 'y')
plt.xticks(rotation = 60)
plt.ylabel("우울증 감소편익")
plt.show()
```


```python
df_sub20_region = df_sub20.groupby("지역").sum("자살 사회비용 감소편익").reset_index()
plt.plot(df_sub20_region['지역'], df_sub20_region["자살 사회비용 감소편익"], marker = 'o', markersize = 10, markerfacecolor = 'red',
        linestyle = '-.')
plt.title("2020 무임승차 폐지 시 자살 사회비용 감소편익")
plt.grid(axis = 'y')
plt.xticks(rotation = 60)
plt.ylabel("자살 사회비용 감소편익")
plt.show()
```


```python
df_sub21_region = df_sub21.groupby("지역").sum("자살 사회비용 감소편익").reset_index()
plt.plot(df_sub21_region['지역'], df_sub21_region["자살 사회비용 감소편익"], marker = 'o', markersize = 10, markerfacecolor = 'purple',
        linestyle = '-.')
plt.title("2021 무임승차 폐지 시 자살 사회비용 감소편익")
plt.grid(axis = 'y')
plt.xticks(rotation = 60)
plt.ylabel("자살 사회비용 감소편익")
plt.show()
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
