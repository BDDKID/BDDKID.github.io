---
layout: single
title:  "미니프로젝트 - 네이버 증권 페이지 스크래핑"
---
```python
#라이브러리 불러오기
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import matplotlib.pyplot as plt
```


```python
#url에 요청하기
url = "https://finance.naver.com/sise/sise_market_sum.naver"
response = requests.get(url)
html = bs(response.text)
a = html.select("input",{"name":"fieldIds"})
```


```python
#조건에 대한 value 값을 리스트에 저장
option_value = []

for i in range(3,30):
    option = a[i]["value"]
    option_value.append(option)
option_value    
```




    ['quant',
     'ask_buy',
     'amount',
     'market_sum',
     'operating_profit',
     'per',
     'open_val',
     'ask_sell',
     'prev_quant',
     'property_total',
     'operating_profit_increasing_rate',
     'roe',
     'high_val',
     'buy_total',
     'frgn_rate',
     'debt_total',
     'net_income',
     'roa',
     'low_val',
     'sell_total',
     'listed_stock_cnt',
     'sales',
     'eps',
     'pbr',
     'sales_increasing_rate',
     'dividend',
     'reserve_ratio']




```python
# 딕셔너리를 만들어주기 위한 key값 리스트 생성
option_name = ["거래량","매수호가","거래대금","시가총액", '영업이익','PER','시가','매도호가','전일거래량',
               '자산총계','영업이익증가율','ROE','고가','매수총잔량','외국인비율','부채총계','당기순이익',
               'ROA','저가','매도총잔량','상장주식수','매출액','주당순이익','PBR','매출액증가율','보통주배당금','유보율']

```


```python
# 조건에 대한 딕셔너리 생성
option_dic = { name:value for name, value in zip(option_name, option_value) } 
```


```python
# 원하는 조건을 선택하여 검색한 URL을 받기위한 함수 
def get_selected(option_num):
    # for문을 통해 얻은 selected_url은 선택된 조건을 충족하는 url로 return한다.
    selected_url = "https://finance.naver.com/sise/field_submit.naver?menu=market_sum&returnUrl=http%3A%2F%2Ffinance.naver.com%2Fsise%2Fsise_market_sum.naver%3F%26page%3D1"

    for _ in range(option_num):
        option = str(input())
        selected_url += '&fieldIds=' + option_dic[option]
    
    selected_response = requests.get(selected_url)
    df = pd.read_html(selected_response.text)
    
    #결측값 및 인덱스 초기화
    selected_stock = df[1].drop_duplicates().drop(0)
    selected_stock = selected_stock.reset_index(drop = 'True')
    

    return selected_stock

```


```python
option_num = int(input())
```

    4



```python
option_num = int(input())
top_stocks = get_selected(option_num)
```

    4
    시가총액
    영업이익증가율
    당기순이익
    ROE



```python
# 선택된 조건으로 나온 데이터 프레임
top_stocks
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>종목명</th>
      <th>현재가</th>
      <th>전일비</th>
      <th>등락률</th>
      <th>액면가</th>
      <th>시가총액</th>
      <th>당기순이익</th>
      <th>영업이익증가율</th>
      <th>ROE</th>
      <th>토론실</th>
      <th>Unnamed: 11</th>
      <th>Unnamed: 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>삼성전자</td>
      <td>64600.0</td>
      <td>700.0</td>
      <td>+1.10%</td>
      <td>100.0</td>
      <td>3856480.0</td>
      <td>399075.0</td>
      <td>43.45</td>
      <td>13.92</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>LG에너지솔루션</td>
      <td>506000.0</td>
      <td>11000.0</td>
      <td>-2.13%</td>
      <td>500.0</td>
      <td>1184040.0</td>
      <td>9299.0</td>
      <td>261.71</td>
      <td>10.68</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>SK하이닉스</td>
      <td>91500.0</td>
      <td>800.0</td>
      <td>-0.87%</td>
      <td>5000.0</td>
      <td>666122.0</td>
      <td>96162.0</td>
      <td>147.58</td>
      <td>16.84</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>삼성바이오로직스</td>
      <td>809000.0</td>
      <td>14000.0</td>
      <td>+1.76%</td>
      <td>2500.0</td>
      <td>575798.0</td>
      <td>3936.0</td>
      <td>83.52</td>
      <td>8.21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>LG화학</td>
      <td>685000.0</td>
      <td>0.0</td>
      <td>0.00%</td>
      <td>5000.0</td>
      <td>483558.0</td>
      <td>39539.0</td>
      <td>178.36</td>
      <td>18.47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>삼성전자우</td>
      <td>58200.0</td>
      <td>600.0</td>
      <td>+1.04%</td>
      <td>100.0</td>
      <td>478920.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7.0</td>
      <td>삼성SDI</td>
      <td>687000.0</td>
      <td>15000.0</td>
      <td>+2.23%</td>
      <td>5000.0</td>
      <td>472412.0</td>
      <td>12504.0</td>
      <td>59.02</td>
      <td>8.45</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>현대차</td>
      <td>173900.0</td>
      <td>1000.0</td>
      <td>-0.57%</td>
      <td>5000.0</td>
      <td>371569.0</td>
      <td>56931.0</td>
      <td>178.91</td>
      <td>6.84</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9.0</td>
      <td>NAVER</td>
      <td>211500.0</td>
      <td>7500.0</td>
      <td>+3.68%</td>
      <td>100.0</td>
      <td>346964.0</td>
      <td>164776.0</td>
      <td>9.06</td>
      <td>106.72</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.0</td>
      <td>카카오</td>
      <td>64700.0</td>
      <td>1000.0</td>
      <td>+1.57%</td>
      <td>100.0</td>
      <td>288189.0</td>
      <td>16462.0</td>
      <td>30.51</td>
      <td>17.10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11.0</td>
      <td>기아</td>
      <td>68700.0</td>
      <td>600.0</td>
      <td>-0.87%</td>
      <td>5000.0</td>
      <td>278485.0</td>
      <td>47603.0</td>
      <td>145.14</td>
      <td>14.69</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12.0</td>
      <td>POSCO홀딩스</td>
      <td>311500.0</td>
      <td>2500.0</td>
      <td>-0.80%</td>
      <td>5000.0</td>
      <td>263439.0</td>
      <td>71959.0</td>
      <td>284.43</td>
      <td>13.97</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13.0</td>
      <td>KB금융</td>
      <td>57700.0</td>
      <td>1200.0</td>
      <td>-2.04%</td>
      <td>5000.0</td>
      <td>235934.0</td>
      <td>43844.0</td>
      <td>31.58</td>
      <td>9.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14.0</td>
      <td>셀트리온</td>
      <td>166800.0</td>
      <td>1500.0</td>
      <td>+0.91%</td>
      <td>1000.0</td>
      <td>234863.0</td>
      <td>5958.0</td>
      <td>5.33</td>
      <td>16.04</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15.0</td>
      <td>신한지주</td>
      <td>44750.0</td>
      <td>150.0</td>
      <td>-0.33%</td>
      <td>5000.0</td>
      <td>227681.0</td>
      <td>41126.0</td>
      <td>20.74</td>
      <td>8.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16.0</td>
      <td>삼성물산</td>
      <td>120700.0</td>
      <td>1000.0</td>
      <td>+0.84%</td>
      <td>100.0</td>
      <td>225573.0</td>
      <td>18291.0</td>
      <td>39.54</td>
      <td>5.40</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17.0</td>
      <td>현대모비스</td>
      <td>214500.0</td>
      <td>5000.0</td>
      <td>-2.28%</td>
      <td>5000.0</td>
      <td>202242.0</td>
      <td>23625.0</td>
      <td>11.46</td>
      <td>6.87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18.0</td>
      <td>포스코케미칼</td>
      <td>211500.0</td>
      <td>3000.0</td>
      <td>+1.44%</td>
      <td>500.0</td>
      <td>163835.0</td>
      <td>1338.0</td>
      <td>101.89</td>
      <td>7.92</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19.0</td>
      <td>LG전자</td>
      <td>98100.0</td>
      <td>200.0</td>
      <td>+0.20%</td>
      <td>5000.0</td>
      <td>160538.0</td>
      <td>14150.0</td>
      <td>-1.06</td>
      <td>6.32</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20.0</td>
      <td>SK이노베이션</td>
      <td>169500.0</td>
      <td>5100.0</td>
      <td>+3.10%</td>
      <td>5000.0</td>
      <td>156729.0</td>
      <td>5010.0</td>
      <td>172.48</td>
      <td>1.91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21.0</td>
      <td>하나금융지주</td>
      <td>51800.0</td>
      <td>1300.0</td>
      <td>-2.45%</td>
      <td>5000.0</td>
      <td>153278.0</td>
      <td>35816.0</td>
      <td>20.71</td>
      <td>10.86</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22.0</td>
      <td>SK</td>
      <td>205500.0</td>
      <td>7600.0</td>
      <td>+3.84%</td>
      <td>200.0</td>
      <td>152377.0</td>
      <td>57184.0</td>
      <td>6518.63</td>
      <td>10.19</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23.0</td>
      <td>삼성생명</td>
      <td>72600.0</td>
      <td>1300.0</td>
      <td>+1.82%</td>
      <td>500.0</td>
      <td>145200.0</td>
      <td>15977.0</td>
      <td>-4.97</td>
      <td>4.01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24.0</td>
      <td>카카오뱅크</td>
      <td>28900.0</td>
      <td>350.0</td>
      <td>+1.23%</td>
      <td>5000.0</td>
      <td>137775.0</td>
      <td>2041.0</td>
      <td>109.66</td>
      <td>4.91</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25.0</td>
      <td>LG</td>
      <td>83900.0</td>
      <td>500.0</td>
      <td>+0.60%</td>
      <td>5000.0</td>
      <td>131976.0</td>
      <td>26840.0</td>
      <td>55.11</td>
      <td>12.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26.0</td>
      <td>한국전력</td>
      <td>20300.0</td>
      <td>50.0</td>
      <td>+0.25%</td>
      <td>5000.0</td>
      <td>130319.0</td>
      <td>-52292.0</td>
      <td>-243.41</td>
      <td>-7.99</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27.0</td>
      <td>KT&amp;G</td>
      <td>94000.0</td>
      <td>2400.0</td>
      <td>-2.49%</td>
      <td>5000.0</td>
      <td>129055.0</td>
      <td>9718.0</td>
      <td>-9.15</td>
      <td>10.74</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28.0</td>
      <td>LG생활건강</td>
      <td>759000.0</td>
      <td>32000.0</td>
      <td>+4.40%</td>
      <td>5000.0</td>
      <td>118542.0</td>
      <td>8611.0</td>
      <td>5.63</td>
      <td>16.65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29.0</td>
      <td>HMM</td>
      <td>23300.0</td>
      <td>1500.0</td>
      <td>+6.88%</td>
      <td>5000.0</td>
      <td>113946.0</td>
      <td>53372.0</td>
      <td>652.21</td>
      <td>88.62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30.0</td>
      <td>고려아연</td>
      <td>570000.0</td>
      <td>3000.0</td>
      <td>-0.52%</td>
      <td>5000.0</td>
      <td>113220.0</td>
      <td>8111.0</td>
      <td>22.15</td>
      <td>11.07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31.0</td>
      <td>삼성전기</td>
      <td>149900.0</td>
      <td>2900.0</td>
      <td>+1.97%</td>
      <td>5000.0</td>
      <td>111966.0</td>
      <td>9154.0</td>
      <td>62.90</td>
      <td>14.29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32.0</td>
      <td>두산에너빌리티</td>
      <td>16590.0</td>
      <td>310.0</td>
      <td>+1.90%</td>
      <td>5000.0</td>
      <td>105895.0</td>
      <td>6458.0</td>
      <td>752.54</td>
      <td>10.67</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33.0</td>
      <td>SK텔레콤</td>
      <td>47850.0</td>
      <td>1500.0</td>
      <td>+3.24%</td>
      <td>100.0</td>
      <td>104712.0</td>
      <td>24190.0</td>
      <td>11.10</td>
      <td>13.63</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34.0</td>
      <td>엔씨소프트</td>
      <td>475500.0</td>
      <td>3500.0</td>
      <td>+0.74%</td>
      <td>500.0</td>
      <td>104391.0</td>
      <td>3957.0</td>
      <td>-54.51</td>
      <td>12.62</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35.0</td>
      <td>S-Oil</td>
      <td>91100.0</td>
      <td>2800.0</td>
      <td>+3.17%</td>
      <td>2500.0</td>
      <td>102563.0</td>
      <td>13785.0</td>
      <td>294.78</td>
      <td>21.76</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36.0</td>
      <td>현대중공업</td>
      <td>113300.0</td>
      <td>5200.0</td>
      <td>+4.81%</td>
      <td>5000.0</td>
      <td>100580.0</td>
      <td>-8142.0</td>
      <td>-2561.66</td>
      <td>-14.87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37.0</td>
      <td>삼성에스디에스</td>
      <td>128500.0</td>
      <td>1300.0</td>
      <td>+1.02%</td>
      <td>500.0</td>
      <td>99430.0</td>
      <td>6334.0</td>
      <td>-7.29</td>
      <td>8.80</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38.0</td>
      <td>삼성화재</td>
      <td>209500.0</td>
      <td>4000.0</td>
      <td>+1.95%</td>
      <td>500.0</td>
      <td>99250.0</td>
      <td>11247.0</td>
      <td>44.28</td>
      <td>7.09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39.0</td>
      <td>우리금융지주</td>
      <td>13480.0</td>
      <td>110.0</td>
      <td>+0.82%</td>
      <td>5000.0</td>
      <td>98143.0</td>
      <td>28074.0</td>
      <td>75.92</td>
      <td>10.59</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40.0</td>
      <td>KT</td>
      <td>35500.0</td>
      <td>100.0</td>
      <td>+0.28%</td>
      <td>5000.0</td>
      <td>92695.0</td>
      <td>14594.0</td>
      <td>41.19</td>
      <td>9.36</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41.0</td>
      <td>대한항공</td>
      <td>24800.0</td>
      <td>450.0</td>
      <td>+1.85%</td>
      <td>5000.0</td>
      <td>91319.0</td>
      <td>5788.0</td>
      <td>1221.20</td>
      <td>11.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42.0</td>
      <td>크래프톤</td>
      <td>183900.0</td>
      <td>1700.0</td>
      <td>+0.93%</td>
      <td>100.0</td>
      <td>90547.0</td>
      <td>5199.0</td>
      <td>-17.35</td>
      <td>17.86</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43.0</td>
      <td>카카오페이</td>
      <td>67300.0</td>
      <td>1300.0</td>
      <td>+1.97%</td>
      <td>500.0</td>
      <td>89511.0</td>
      <td>-339.0</td>
      <td>-52.03</td>
      <td>-2.45</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44.0</td>
      <td>한화솔루션</td>
      <td>46450.0</td>
      <td>200.0</td>
      <td>-0.43%</td>
      <td>5000.0</td>
      <td>88849.0</td>
      <td>6163.0</td>
      <td>24.26</td>
      <td>8.79</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45.0</td>
      <td>아모레퍼시픽</td>
      <td>147400.0</td>
      <td>4000.0</td>
      <td>+2.79%</td>
      <td>500.0</td>
      <td>86218.0</td>
      <td>1809.0</td>
      <td>140.10</td>
      <td>4.20</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46.0</td>
      <td>기업은행</td>
      <td>10590.0</td>
      <td>20.0</td>
      <td>+0.19%</td>
      <td>5000.0</td>
      <td>84447.0</td>
      <td>24259.0</td>
      <td>52.03</td>
      <td>9.21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47.0</td>
      <td>하이브</td>
      <td>194000.0</td>
      <td>3600.0</td>
      <td>+1.89%</td>
      <td>500.0</td>
      <td>80226.0</td>
      <td>1408.0</td>
      <td>30.74</td>
      <td>6.83</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48.0</td>
      <td>현대글로비스</td>
      <td>183200.0</td>
      <td>500.0</td>
      <td>+0.27%</td>
      <td>500.0</td>
      <td>68700.0</td>
      <td>7832.0</td>
      <td>70.09</td>
      <td>14.41</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49.0</td>
      <td>LG이노텍</td>
      <td>284000.0</td>
      <td>6500.0</td>
      <td>+2.34%</td>
      <td>5000.0</td>
      <td>67215.0</td>
      <td>8883.0</td>
      <td>85.64</td>
      <td>30.94</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50.0</td>
      <td>롯데케미칼</td>
      <td>182300.0</td>
      <td>900.0</td>
      <td>-0.49%</td>
      <td>5000.0</td>
      <td>62484.0</td>
      <td>14256.0</td>
      <td>330.26</td>
      <td>9.87</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 시각화에 필요한 데이터들만 가져오기
pd_data = pd.DataFrame()
pd_data['종목명'] = top_stocks['종목명']
pd_data['시가총액'] = top_stocks['시가총액']
pd_data['영업이익증가율'] = top_stocks['영업이익증가율']
pd_data['당기순이익'] = top_stocks['당기순이익']
pd_data['ROE'] = top_stocks['ROE']
```


```python
pd_data = pd_data.dropna()
```


```python
pd_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>종목명</th>
      <th>시가총액</th>
      <th>영업이익증가율</th>
      <th>당기순이익</th>
      <th>ROE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>삼성전자</td>
      <td>3689326.0</td>
      <td>43.45</td>
      <td>399075.0</td>
      <td>13.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LG에너지솔루션</td>
      <td>1098630.0</td>
      <td>261.71</td>
      <td>9299.0</td>
      <td>10.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SK하이닉스</td>
      <td>637730.0</td>
      <td>147.58</td>
      <td>96162.0</td>
      <td>16.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>삼성바이오로직스</td>
      <td>569392.0</td>
      <td>83.52</td>
      <td>3936.0</td>
      <td>8.21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LG화학</td>
      <td>441202.0</td>
      <td>178.36</td>
      <td>39539.0</td>
      <td>18.47</td>
    </tr>
    <tr>
      <th>6</th>
      <td>삼성SDI</td>
      <td>425652.0</td>
      <td>59.02</td>
      <td>12504.0</td>
      <td>8.45</td>
    </tr>
    <tr>
      <th>7</th>
      <td>현대차</td>
      <td>351484.0</td>
      <td>178.91</td>
      <td>56931.0</td>
      <td>6.84</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NAVER</td>
      <td>321536.0</td>
      <td>9.06</td>
      <td>164776.0</td>
      <td>106.72</td>
    </tr>
    <tr>
      <th>9</th>
      <td>카카오</td>
      <td>272599.0</td>
      <td>30.51</td>
      <td>16462.0</td>
      <td>17.10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>POSCO홀딩스</td>
      <td>260902.0</td>
      <td>284.43</td>
      <td>71959.0</td>
      <td>13.97</td>
    </tr>
    <tr>
      <th>11</th>
      <td>기아</td>
      <td>260649.0</td>
      <td>145.14</td>
      <td>47603.0</td>
      <td>14.69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>KB금융</td>
      <td>233071.0</td>
      <td>31.58</td>
      <td>43844.0</td>
      <td>9.80</td>
    </tr>
    <tr>
      <th>13</th>
      <td>셀트리온</td>
      <td>228808.0</td>
      <td>5.33</td>
      <td>5958.0</td>
      <td>16.04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>삼성물산</td>
      <td>223330.0</td>
      <td>39.54</td>
      <td>18291.0</td>
      <td>5.40</td>
    </tr>
    <tr>
      <th>15</th>
      <td>신한지주</td>
      <td>220304.0</td>
      <td>20.74</td>
      <td>41126.0</td>
      <td>8.80</td>
    </tr>
    <tr>
      <th>16</th>
      <td>현대모비스</td>
      <td>197527.0</td>
      <td>11.46</td>
      <td>23625.0</td>
      <td>6.87</td>
    </tr>
    <tr>
      <th>17</th>
      <td>LG전자</td>
      <td>157593.0</td>
      <td>-1.06</td>
      <td>14150.0</td>
      <td>6.32</td>
    </tr>
    <tr>
      <th>18</th>
      <td>하나금융지주</td>
      <td>152390.0</td>
      <td>20.71</td>
      <td>35816.0</td>
      <td>10.86</td>
    </tr>
    <tr>
      <th>19</th>
      <td>SK이노베이션</td>
      <td>144709.0</td>
      <td>172.48</td>
      <td>5010.0</td>
      <td>1.91</td>
    </tr>
    <tr>
      <th>20</th>
      <td>SK</td>
      <td>143850.0</td>
      <td>6518.63</td>
      <td>57184.0</td>
      <td>10.19</td>
    </tr>
    <tr>
      <th>21</th>
      <td>포스코케미칼</td>
      <td>142920.0</td>
      <td>101.89</td>
      <td>1338.0</td>
      <td>7.92</td>
    </tr>
    <tr>
      <th>22</th>
      <td>삼성생명</td>
      <td>142200.0</td>
      <td>-4.97</td>
      <td>15977.0</td>
      <td>4.01</td>
    </tr>
    <tr>
      <th>23</th>
      <td>카카오뱅크</td>
      <td>132531.0</td>
      <td>109.66</td>
      <td>2041.0</td>
      <td>4.91</td>
    </tr>
    <tr>
      <th>24</th>
      <td>한국전력</td>
      <td>130961.0</td>
      <td>-243.41</td>
      <td>-52292.0</td>
      <td>-7.99</td>
    </tr>
    <tr>
      <th>25</th>
      <td>KT&amp;G</td>
      <td>130016.0</td>
      <td>-9.15</td>
      <td>9718.0</td>
      <td>10.74</td>
    </tr>
    <tr>
      <th>26</th>
      <td>LG</td>
      <td>129931.0</td>
      <td>55.11</td>
      <td>26840.0</td>
      <td>12.36</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LG생활건강</td>
      <td>117761.0</td>
      <td>5.63</td>
      <td>8611.0</td>
      <td>16.65</td>
    </tr>
    <tr>
      <th>28</th>
      <td>고려아연</td>
      <td>112425.0</td>
      <td>22.15</td>
      <td>8111.0</td>
      <td>11.07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>삼성전기</td>
      <td>110547.0</td>
      <td>62.90</td>
      <td>9154.0</td>
      <td>14.29</td>
    </tr>
    <tr>
      <th>30</th>
      <td>HMM</td>
      <td>104410.0</td>
      <td>652.21</td>
      <td>53372.0</td>
      <td>88.62</td>
    </tr>
    <tr>
      <th>31</th>
      <td>두산에너빌리티</td>
      <td>103725.0</td>
      <td>752.54</td>
      <td>6458.0</td>
      <td>10.67</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SK텔레콤</td>
      <td>102633.0</td>
      <td>11.10</td>
      <td>24190.0</td>
      <td>13.63</td>
    </tr>
    <tr>
      <th>33</th>
      <td>엔씨소프트</td>
      <td>100220.0</td>
      <td>-54.51</td>
      <td>3957.0</td>
      <td>12.62</td>
    </tr>
    <tr>
      <th>34</th>
      <td>S-Oil</td>
      <td>99298.0</td>
      <td>294.78</td>
      <td>13785.0</td>
      <td>21.76</td>
    </tr>
    <tr>
      <th>35</th>
      <td>삼성화재</td>
      <td>97592.0</td>
      <td>44.28</td>
      <td>11247.0</td>
      <td>7.09</td>
    </tr>
    <tr>
      <th>36</th>
      <td>현대중공업</td>
      <td>96319.0</td>
      <td>-2561.66</td>
      <td>-8142.0</td>
      <td>-14.87</td>
    </tr>
    <tr>
      <th>37</th>
      <td>삼성에스디에스</td>
      <td>94788.0</td>
      <td>-7.29</td>
      <td>6334.0</td>
      <td>8.80</td>
    </tr>
    <tr>
      <th>38</th>
      <td>KT</td>
      <td>94653.0</td>
      <td>41.19</td>
      <td>14594.0</td>
      <td>9.36</td>
    </tr>
    <tr>
      <th>39</th>
      <td>우리금융지주</td>
      <td>92464.0</td>
      <td>75.92</td>
      <td>28074.0</td>
      <td>10.59</td>
    </tr>
    <tr>
      <th>40</th>
      <td>대한항공</td>
      <td>90214.0</td>
      <td>1221.20</td>
      <td>5788.0</td>
      <td>11.60</td>
    </tr>
    <tr>
      <th>41</th>
      <td>카카오페이</td>
      <td>87915.0</td>
      <td>-52.03</td>
      <td>-339.0</td>
      <td>-2.45</td>
    </tr>
    <tr>
      <th>42</th>
      <td>한화솔루션</td>
      <td>87032.0</td>
      <td>24.26</td>
      <td>6163.0</td>
      <td>8.79</td>
    </tr>
    <tr>
      <th>43</th>
      <td>아모레퍼시픽</td>
      <td>86862.0</td>
      <td>140.10</td>
      <td>1809.0</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>44</th>
      <td>크래프톤</td>
      <td>84666.0</td>
      <td>-17.35</td>
      <td>5199.0</td>
      <td>17.86</td>
    </tr>
    <tr>
      <th>45</th>
      <td>기업은행</td>
      <td>83331.0</td>
      <td>52.03</td>
      <td>24259.0</td>
      <td>9.21</td>
    </tr>
    <tr>
      <th>46</th>
      <td>하이브</td>
      <td>77331.0</td>
      <td>30.74</td>
      <td>1408.0</td>
      <td>6.83</td>
    </tr>
    <tr>
      <th>47</th>
      <td>현대글로비스</td>
      <td>66938.0</td>
      <td>70.09</td>
      <td>7832.0</td>
      <td>14.41</td>
    </tr>
    <tr>
      <th>48</th>
      <td>LG이노텍</td>
      <td>65321.0</td>
      <td>85.64</td>
      <td>8883.0</td>
      <td>30.94</td>
    </tr>
    <tr>
      <th>49</th>
      <td>롯데케미칼</td>
      <td>64266.0</td>
      <td>330.26</td>
      <td>14256.0</td>
      <td>9.87</td>
    </tr>
  </tbody>
</table>
</div>




```python
roe_max_value = pd_data['ROE'].max()
roe_mean_value = pd_data['ROE'].mean()
bpp_max_value = pd_data['영업이익증가율'].max()
bpp_mean_value = pd_data['영업이익증가율'].mean()
```


```python
visual_data = pd_data[
    (pd_data['ROE'] >= roe_mean_value) |
    (pd_data['영업이익증가율'] >= bpp_mean_value)]
visual_data.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>종목명</th>
      <th>시가총액</th>
      <th>영업이익증가율</th>
      <th>당기순이익</th>
      <th>ROE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>삼성전자</td>
      <td>3689326.0</td>
      <td>43.45</td>
      <td>399075.0</td>
      <td>13.92</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LG에너지솔루션</td>
      <td>1098630.0</td>
      <td>261.71</td>
      <td>9299.0</td>
      <td>10.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SK하이닉스</td>
      <td>637730.0</td>
      <td>147.58</td>
      <td>96162.0</td>
      <td>16.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LG화학</td>
      <td>441202.0</td>
      <td>178.36</td>
      <td>39539.0</td>
      <td>18.47</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NAVER</td>
      <td>321536.0</td>
      <td>9.06</td>
      <td>164776.0</td>
      <td>106.72</td>
    </tr>
    <tr>
      <th>9</th>
      <td>카카오</td>
      <td>272599.0</td>
      <td>30.51</td>
      <td>16462.0</td>
      <td>17.10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>POSCO홀딩스</td>
      <td>260902.0</td>
      <td>284.43</td>
      <td>71959.0</td>
      <td>13.97</td>
    </tr>
    <tr>
      <th>11</th>
      <td>기아</td>
      <td>260649.0</td>
      <td>145.14</td>
      <td>47603.0</td>
      <td>14.69</td>
    </tr>
    <tr>
      <th>13</th>
      <td>셀트리온</td>
      <td>228808.0</td>
      <td>5.33</td>
      <td>5958.0</td>
      <td>16.04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>SK</td>
      <td>143850.0</td>
      <td>6518.63</td>
      <td>57184.0</td>
      <td>10.19</td>
    </tr>
    <tr>
      <th>27</th>
      <td>LG생활건강</td>
      <td>117761.0</td>
      <td>5.63</td>
      <td>8611.0</td>
      <td>16.65</td>
    </tr>
    <tr>
      <th>29</th>
      <td>삼성전기</td>
      <td>110547.0</td>
      <td>62.90</td>
      <td>9154.0</td>
      <td>14.29</td>
    </tr>
    <tr>
      <th>30</th>
      <td>HMM</td>
      <td>104410.0</td>
      <td>652.21</td>
      <td>53372.0</td>
      <td>88.62</td>
    </tr>
    <tr>
      <th>31</th>
      <td>두산에너빌리티</td>
      <td>103725.0</td>
      <td>752.54</td>
      <td>6458.0</td>
      <td>10.67</td>
    </tr>
    <tr>
      <th>32</th>
      <td>SK텔레콤</td>
      <td>102633.0</td>
      <td>11.10</td>
      <td>24190.0</td>
      <td>13.63</td>
    </tr>
    <tr>
      <th>34</th>
      <td>S-Oil</td>
      <td>99298.0</td>
      <td>294.78</td>
      <td>13785.0</td>
      <td>21.76</td>
    </tr>
    <tr>
      <th>40</th>
      <td>대한항공</td>
      <td>90214.0</td>
      <td>1221.20</td>
      <td>5788.0</td>
      <td>11.60</td>
    </tr>
    <tr>
      <th>44</th>
      <td>크래프톤</td>
      <td>84666.0</td>
      <td>-17.35</td>
      <td>5199.0</td>
      <td>17.86</td>
    </tr>
    <tr>
      <th>47</th>
      <td>현대글로비스</td>
      <td>66938.0</td>
      <td>70.09</td>
      <td>7832.0</td>
      <td>14.41</td>
    </tr>
    <tr>
      <th>48</th>
      <td>LG이노텍</td>
      <td>65321.0</td>
      <td>85.64</td>
      <td>8883.0</td>
      <td>30.94</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import font_manager,rc
import matplotlib.pyplot as plt
import seaborn as sns
import platform
```


```python
font_path = ''
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname = font_path).get_name()
    rc('font',family = font_name)
elif platform.system() == 'Darwin':
    font_path = '/Users/$USER/Library/Fonts/AppleGothic.ttf'
    rc('font',family = 'AppleGothic')
else:
    print('Check your OS System')
%matplotlib inline
```


```python
visual_df = visual_data[:10]
```


```python
bpp_max = visual_df['영업이익증가율'].max()
bpp_mean = visual_df['영업이익증가율'].mean()
roe_max = visual_df['ROE'].max()
roe_mean = visual_df['ROE'].mean()

```


```python
plt.figure(figsize=(10,10))
plt.title('ROE,영엽이익증가율 기준 상위종목 top10')
sns.scatterplot(x = 'ROE',
               y = '영업이익증가율',
               size = '시가총액',
               hue = visual_df['종목명'],
               data = visual_df,sizes = (0,1000))
plt.plot([0,roe_max],
        [bpp_mean,bpp_mean],
        'r--',
        lw = 1)
plt.plot([roe_mean,roe_mean],
        [0,bpp_max],
        'r--',
        lw=1)
for index, row in visual_df.iterrows():
    x = row['ROE']
    y = row['영업이익증가율']
    s = row['종목명']
    plt.text(x,y,s,size=10)
plt.show()
```


    
![png](output_18_0.png)
    



```python

```
