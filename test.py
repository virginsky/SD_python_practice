import numpy as np
import pandas as pd
import matplotlib as plt

# print(np.__version__)
# print(pd.__version__)
# print(plt.__version__)


# df1 = pd.DataFrame(np.arange(16).reshape(4, -1),
#                    columns=('c1', 'c2', 'c3', 'c4'),
#                    index=('r1', 'r2', 'r3', 'r4'))

# print(df1)
# df1.drop('c2',axis = 1) #칼럼으로 접근
# print(df1.drop(columns = ['c3','c4']))

# merge 데이터병합

# data1 = pd.DataFrame({'id':['01','02','03','04','05','06'],
#                     'col1': np.random.randint(0,50,6),
#                     'col2': np.random.randint(1000,2000,6)})

# data2 = pd.DataFrame({'id':['04','05','06','07'],
#                     'col1': np.random.randint(0,50,4),
#                     'col2': np.random.randint(1000,2000,4)})

# print(data1, data2)
# print(pd.merge(data1,data2, on='id'))
# print(pd.merge(data1,data2, how = 'inner', on='id')) #교집합
# print(pd.merge(data1,data2, how = 'outer', on='id')) #합집합
# print(pd.merge(data1,data2, how = 'right', on='id'))
# df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# df
# print(df.max()) # default -> axis = 1(행) -> axis = 0(열)
# print(df.min()) # default -> axis = 1(행) -> axis = 0(열)

# print(df.isnull().sum().sum())
# print(df.isna().sum().sum())

# 결측치를 어떻게 넣을까? -> 가장 간단하게 넣을 것

# ---------------------------------------------

# obj = pd.Series(['apple', 'mango', np.nan, None, 'peach'])
# print(obj.isna().sum()) #공란 몇개인지 확인
# print(obj)
# print(obj.dropna())     #공란 drop

# ---------------------------------------------

# frame = pd.DataFrame(   [[np.nan, np.nan, np.nan, np.nan],
#                         [10,    5,  40, 6],
#                         [5, 2,  30, 8],
#                         [20,    np.nan,     20, 6],
#                         [15,    3,  10, np.nan]],
#                         columns = ['x1','x2','x3','x4'])
# frame['e']=np.nan #numpy ->브로딩캐스팅

# print(frame)
# print(frame.fillna(1000))   #넣는 종류가 다양함, 필요에따라 입력

# ---------------------------------------------

# data = pd.DataFrame({'id' : ['0001', '0002', '0003', '0001'],
#                     'name': ['a','b','c','a'],
#                     'phone':[0,1,2,0]})
# print(data)
# print(data.duplicated())    #중복체크
# print(data.drop_duplicates())   #중복체크 후 바로 drop
# print(data.drop_duplicates(subset=['id'], keep='last')) #id를 기준으로 중복을 검색하고, 마지막열을 남김

# ---------------------------------------------

# obj = pd.Series([10,-999,4,5,7,'n'])
# print(obj)
# print(obj.replace(-999, np.nan))        #-999를 nan으로 치환
# print(obj.replace([-999, 'n'], np.nan)) #-999와 n을 Nan으로 치환

# ---------------------------------------------

# binning : 연속형 데이터를 구간으로 나누어 범주화
# 숫자로 되어있는걸 -> 카테고리별 구간으로 나눔

# age = [20,35,67,39,59,44,56,77,28,20,22,80,32,46,52,19,33,5,15,50,29,21,33,48,85,80,31,10]
# bins = [0,20,40,60,100]

# cuts = pd.cut(age,bins)
# print(cuts)
# print(cuts.categories)
# print(cuts.codes)
# # 구간을 균등한 길이로 나눔
# print(pd.cut(age, 4, precision=1).value_counts())

# --------------------------------------

# [중요] get_dummies : categorical variable(명목형 변수)를 one-hot encoding해줌

# df = pd.DataFrame({ 'col1':[10,20,30,40],
#                     'col2':['a','b','c','d']})
# print(df)
# print(pd.get_dummies(df))

# df1 = pd.DataFrame({'col1':['001','002','003','004','005','006'],
#                     'col2':[10,20,30,40,50,60],
#                     'col3':['서울시','경기도','서울시','제주도','경기도','서울시']})
# print(df1)
# print(pd.get_dummies(df1))

# --------------------------------------

# groupby

# kbo = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# print(kbo.head())
# print(kbo.shape)
# print(kbo['팀'].unique())
# print(kbo.info())
# print(kbo.groupby('팀').count())
# print(kbo.groupby(['연도','팀']).sum())     #group을 연도로 묶고, 그 안에서 팀으로 한번 더 묶음
# print(kbo.groupby('팀') ['승률'].max())      #group을 팀으로 묶고, 그 중 승률만 보여줌
# print(kbo.groupby(['연도','팀']) ['승률','순위'].max())

# 맥 설치 성공!
