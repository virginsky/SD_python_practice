# # 강의 스케쥴 
#  A. 파이썬 (초급~고급) (3월 스케쥴 - 2주)
#     선형대수(numpy)
#     Pandas
#     Matplotlib

#  B. Machine Learning algorism(4월 스케쥴 - 4주)
#     1. Linear regression (21.04.06)
#         Gradient Descent(p17 경사하강법 - 비용최소화하기)
#         Ridge, Lasso, elastic
#         Sklearn boston hous price
#         Kaggle 실습(1~2개)
#     2. Multiple regression
#         Gradient Descent(p17 경사하강법 - 비용최소화하기)
#         Ridge, Lasso, elastic
#         Sklearn boston hous price
#         Kaggle 실습(1~2개)
#     3. Poisson regression
#     4. Logstic regression
#     5. Classification (+ Kaggle 실습 6~7개 )
#         a. Decision tree
#             entropy, Informthea gain?
#             Gini index
#         b. Bresting, Bagging, Voting, Stacting?
#             Random Forest
#             Gradiet Botosting
#             Ada boosting
#             XGBoosting
#             Light GBM
#             Catboost
#             TabNet
#     6. Clustering ( + Kaggle 실습 3~4개 )
#         K-means
#         K-meroid
#         Gaussian mixed model
#         DBSCAN
#     7. bayesion -> * Likelihood *
#         DACinear D Anaraysi?
#         Support Vector machine
#             Kernel
        
#  C. Deep Learning (5월 스케쥴 - 4주)
#     1. Perception
#     2. Nuti Perceturon
#     3. Convolutinal neurat neti?
#     4. Recurend neural netc?

#  D. Etc..(선형대수, 확률론, 수리통계, 미적분학, 해석학)








import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm


# print(np.__version__)
# print(pd.__version__)
# # print(plt.__version__) #.pyplot는 버젼 안나옴
# print(sns.__version__)
# print(sklearn.__version__)
# print(sm.__version__)

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













# ---------------------------Day 17 복습(210402)------------------
# 맥 설치 성공!
# p153 연습문제 5-2
# a = input()
# b = input()
# c = input()

# if a > 100 or b > 100 or c > 100:
#     print("잘못된 숫자가 입력되었습니다.")
# elif a > 65 and b > 65 and c > 65:
#     print("합격")
# else:
#     print("불합격")

# < 가위바위보 문제>
# 사용자 입력과 random함수를 사용하여,
# 사용자와 컴퓨터가 대결하는 가위바위보 게임을 만들어보자
# 입력 : [문자열]"가위","바위" 혹은 "보"
# 출력 : [문자열] 결과 반환

# 구구단 예제
# def gugu(num):
#     for i in range(1, 10):
#         print(f'{num} x {i} = {num * i}')


# print(gugu(3))
# print(gugu(5))












# # ---------------------------(Day 18 - 210405)------------------








# # x = np.arange(9).reshape(3,-1)
# # print(x)
# # print(np.diag(x))
# # print(np.diag(np.diag(x)))

# #dot = 1 - dim(가능)
# # ?? 못적음

# a = np.random.randint(-3,3,10).reshape(2,-1)
# b = np.random.randint(0,5,15).reshape(5,3)

# print(a.shape, b.shape)  #shape 확인하는 습관을 들이면 좋음
# # print(a,b)

# # Matrix Multiplication : 행렬의 곱
# ab = np.matmul(a,b)
# print(ab.shape, '/n') #/n -> enter를 하세요(파이썬한테 지시)
# # print(ab.shape, '/t') #/t -> tab을 눌러서 작동하시오

# b = np.arange(16).reshape(4,-1)
# print(b)
# print(np.trace(b)) # 대각선의 합

# # Determiant - 역행렬이 존재하는지 여부를 확인하는 방법으로 행렬식(det)
# d = np.array([[1,2],[3,4]])
# print(np.linalg.det(d)) # 0 이 아니면 역행렬이 존재함 (ad-bc = 0일경우 역행렬x)

# # Invermaterix - 역행렬 구하기
# print(np.linalg.inv(d))
# # non singular -> 역행렬 존재
# # singular -> 역행렬이 존재하지 않음

# # ---------------------Pandas------------------------------------------
# # missing data처리가 용이
# # 축의 이름에 따라 데이터를 정렬할 수 있는 자료구조 제공
# # 일반 데이터베이스처럼 데이터를 합치고 관계연산을 수행하는 기능
# # 시계열 데이터 처리가 용이

# #DataFrame (indexing / slicing 연습)
# frame = pd.DataFrame(np.arange(24).reshape(4,-1),
#                     columns = ['c1','c2','c3','c4','c5','c6'],
#                     index = ['r1','r2','r3','r4'])
# print(frame)
# # print(frame.c3)
# # print(frame[['c1','c3']])       #multi index로 입력해야 함? (row는 싱글인덱스?)
# # print(frame['r1':'r3'])

# # iloc / loc 연습
# #위치 인덱싱 : integer-location based property (우선순위가 행) - 우선순위를 []로 한번 더 묶어주는개념인가?
# print(frame.iloc[[0],[3]])      
# print(frame.iloc[[0,1],1:4])
# #레이블 인덱싱 : label-location based property (우선순위가 열)
# print(frame.loc[['r1'],['c4']]) 
# print(frame.loc['r1':'r2',['c2','c3','c4']])


# # -------------------------Series-------------------------
# # 구글드라이브에서 바로 오픈하는것 없나? 코랩처럼 - Google Drive for VSCode
# df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# # print(df)
# print(df.head()) #위에서부터 디폴트 5개
# print(df.tail()) #아래부터 디폴트 5개
# print(df.describe()) #많이 사용

# map : series의 각각의 element들을 다른 어떤 값으로 대체하는 역할
# lambda 공부하자..

# 데이터 삭제 - drop (기본적으로 return 없음)
# frame = pd.DataFrame(np.arange(16).reshape(4,4),
#                         columns = ['c1','c2','c3','c4'],
#                         index = ['r1','r2','r3','r4'])
# print(frame)
# print(frame.drop('r1'))
# print(frame.drop('c1', axis = 1))
# print(frame.drop(columns = ['c3', 'c4']))

# # return값을 주고 싶을 경우 -> option : inplace 사용
# print(frame.drop(['r2'],inplace = True))
# print(frame)        #r2값 drop이 리턴(적용) 됨

# 데이터 병합 - concat, merge


# missing data 처리(isnull, notnull, fillna, dropna)

# df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# print(df.isnull().sum())

# obj = pd.Series(['apple', 'mango', np.nan, None, 'Peach'])
# print(obj)
# print(obj.isnull().sum())
# print(obj.isna().sum())
# print(obj.dropna()) # option인 thresh, any는 2개이상일떄만 사용가능

# frame = pd.DataFrame([[np.nan,5,np.nan,np.nan,np.nan], 
#                     [10,5,40,6,np.nan],
#                     [5,2,30,8,np.nan],
#                     [20,np.nan,4,7,20]])
# print(frame)
# print(frame.fillna(0))
# print(frame.fillna('없음'))
# print(frame.fillna(frame.mean()))
# print(frame.drop_duplicates)


# p91 데이터 변형 (get_dummies 중요) -> 명목형 변수를 one-hot encoding 해줌


# Groupby
# df = pd.read_csv('C:/Users/sundooedu/Documents/JH_version/Day16/kbo.csv')
# print(df.shape)
# print(df.columns)
# print(df['팀'].unique())
# print(df.describe())    #통계량
# print(df.info())        #데이터형, columns
# print(df.groupby('팀').count())
# print(df.groupby('팀').mean())
# print(df.groupby(['연도', '팀']).sum())
# print(df.groupby(['팀','승률']).max())
# print(df.groupby(['연도','팀'])['승률','순위'].max())

# grouped = df.groupby('팀')

# for name, group in grouped:
#     print(name)
#     print(group)
#     print('-'*50)
# print(grouped.get_group('한화'))

# -----------------------Matplotlib----------------------------
# VS code는 그래프 모양이 이쁘지 않음(결과가 안나오는건 아님) - 아나콘다는 이쁨(?)

# plt.plot([1,2,3,1000,5000,6000])        # plt.plot : 점만 찍어도 다 이어서 보여줌
# plt.show()

# plt.plot([1,2,3,1000,5000,6000], marker = 'o', color = 'red')
# plt.show()

# #
# x = np.linspace(0, 2*np.pi, 400)
# y = np.sin(x**2)
# fig,ax = plt.subplots()
# ax.plot(x,y)
# ax.set_title('A single plot')
# plt.show()

# # p12 그래프 내용 및 이름 참조
# x = np.arange(10)
# y = x+10
# plt.plot(x,y)
# plt.show()

# plt.xlim([0,10])
# plt.ylim([0,20])
# plt.plot(x,y)
# plt.show()

# x = np.linspace(-2,2,1000)
# y = x**3
# plt.plot(x,y)
# plt.show()

# # 예시그래프
# x = np.linspace(-1.4, 1.4, 30)
# plt.figure(1)
# plt.subplot(211) #2행 1열 첫번째
# plt.plot(x, x**2)
# plt.title("Square and Cube")
# plt.subplot(212) #2행 1열 두번째
# plt.plot(x, x**3)
# plt.figure(2, figsize=(10, 5))
# plt.subplot(121)
# plt.plot(x, x**4)
# plt.title("y = x**4")
# plt.subplot(122)
# plt.plot(x, x**5)
# plt.title("y = x**5")
# plt.figure(1)      # 그림 1로 돌아가며, 활성화된 부분 그래프는 212 (하단)이 됩니다
# plt.plot(x, -x**3, "r:")
# plt.show()



# x = np.linspace(0, 2*np.pi, 500)
# ############
# y = np.sin(x)
# z = np.cos(x)
# # w = np.tan(x)
# ############
# fig,ax = plt.subplots()

# ax.legend()
# plt.show()

# # Bar Plot
# data = {'apple':21, 'banana':15, 'pear':5, 'kiwi':20}
# names = list(data.keys())
# values = list(data.values())

# fig, ax = plt.subplots()
# ax.bar(names, values)
# plt.show()

## Histogram = hist()
# data = np.random.rand(10000)
# fig, ax = plt.subplots()
# ax.hist(data, bins = 100, facecolor = 'b')
# plt.show()

# # Scatter Plot : Scatter()
# np.random.seed(700)
# n = 50
# x = np.random.rand(n)
# y = np.random.rand(n)
# plt.scatter(x,y)
# plt.show()

# # Pie chart : Pie()
# ratio = [34,32,16,18]
# labels = ['apple','banana','melon','grapes']
# plt.pie(ratio, labels = labels, autopct = '%.1f%%')
# plt.show()

# 데이터분석 시 유의사항
# 1. 결측치 : model작동 x
# 2. 이상치 : model이 휘둘림(평균 등)

# #각종 표현방식(그래프)는 이런게 있따 정도로 알고있으면 됨 -> 추후 분석 시 적합한 것 활용
# tips = sns.load_dataset('tips')
# sns.relplot(x = 'total_bill', y = 'tip', hue = 'smoker', style = 'smoker', data = tips)
# sns.set()
# plt.show()



# # ---------------------------사이킷런으로 시작하는 머신러닝-----------------------------
# # ---------------------------붓꽃 예시-----------------------------
# #  (프로젝트의 전반적인 진행단계 설명)
# from sklearn.datasets import load_iris
# iris_dataset = load_iris()
# # print(iris_dataset)
# # print('iris_dataset의 키 : \n{}'.format(iris_dataset.keys()))

# #DESCR의 설명
# # print(iris_dataset['DESCR'][:193]+ '\n...')

# #target_names의 값은 우리가 예측하려는 붓꽃 품종의 이름을 문자열로 가지고 있다.
# # print('타깃의 이름 : {}'.format(iris_dataset['target_names']))

# # print('특성의 이름 : {}'.format(iris_dataset['feature_names']))
# # print('data 타입 : {}'.format(type(iris_dataset['data'])))
# # print('data의 크기 :{}'.format(iris_dataset['data'].shape))
# # print('data의 처음 다섯행 :\n {}'.format(iris_dataset['data'][:5]))
# # print('target의 타입 : {}'.format(type(iris_dataset['target'])))
# # print('target의 크기 : {}'.format(type(iris_dataset['target'].shape)))
# # print('타깃: \n{}'.format(iris_dataset['target']))



# # 성과 측정 : 훈련 데이터와 테스트 데이터
# # 우리가 만든 모델이 새 데이터에 적용하기 전에 이 모델이 진짜 잘 작동하는지 알아야 함.
# # 불행히도 모델을 만들 때 쓴 데이터는 평가 목적으로 사용 불가
# # 훈련 데이터에 속한 어떤 데이터라도 정확히 맞출 수 있기 때문에.(기억 가능성 때문)
# # 데이터를 기억한다는 것은 모델을 잘 일반화하지 않았다는 뜻(새로운 데이터에 대해서는 잘 작동을 안한다)
# # 모델의 성능을 평가하려면 레이블을 알고 있는 (이전에 본적 없는) 새 데이터를 모델에 적용해봐야 함. 머신러닝 모델을 만들때 훈련데이터(Train data) 혹은 훈련 세트(train Set)로 훈련을 시키고, 모델이 잘 작동하는지 측정하는 것을 테스트 데이터(test data), 테스트 세트(test set) 혹은 홀드아웃 세트(hold-out set)라고 부름.
# # -scikit-learn 데이터는 대문자 X로 표시하고 레이블은 소문자 y로 표기함.

# # Default : 75% and 25% 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.2,random_state = 0)
# # # random_state = np.seed    : 중요(이걸 안하면 분석을 돌릴때마다 데이터 변경 됨)
# # print('X_train 크기:{}'.format(X_train.shape))
# # print('y_train 크기:{}'.format(y_train.shape))
# # print('X_test 크기:{}'.format(X_test.shape))
# # print('y_test 크기:{}'.format(y_test.shape))


# # 가장 먼저 해야 할일 : 데이터 살펴보기
# # 시각화는 데이터를 조사하는 아주 좋은 방법.
# # 산점도(Scatter Matrix)가 그중 하나
# # 그래프를 그려주려면 Numpy ->DataFrame으로 바꿔줘야 함
# import pandas as pd
# #X_train 데이터를 사용해서 데이터프레임을 만듬.

# # 열의 이름은 iris_dataset.feature_names에 있는 문자열을 사용
# iris_dataframe = pd.DataFrame(X_train,columns = iris_dataset.feature_names)

# #데이터프레임을 사용해 y_train에 따라 색으로 구분된 산점도 행렬을 만듬
# pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha =0.8)
# # plt.show()


# # 첫번째 머신러닝 모델 : K-최근접 이웃 알고리즘
# # 새로운 데이터 포인트에 대한 예측이 필요하면 알고리즘은 새 데이터 포인트에서 가장 가까운 훈련 데이터 포인트를 찾음. 그런 다음 찾은 훈련 데이터의 레이블을 새 데이터 포인트의 레이블로 지정
# # k는 가장 가까운 이웃 '하나'가 아니라 훈련 데이터에서 새로운 데이터 포인트에 가장 가까운 'k'개의 이웃을 찾는다는 뜻.
# # 이웃들의 클래스 중 빈도가 가장 높은 클래스를 예측값으로 예측
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 1)

# knn.fit(X_train,y_train)

# # 예측하기
# # 야생에서 꽃받침의 길이가 5cm,폭이 2.9cm이고 꽃잎의 길이가 1cm, 폭이 0.2cm 인 붓꽃을 보았다고 가정. 이게 무슨 품종인지 맞추는 것을 예측한다고 하자

# import numpy as np
# X_new = np.array([[5, 2.9, 1, 0.2]])
# print('X_new.shape : {}'.format(X_new.shape))

# # scikit-learn은 항상 데이터가 2차원 배열일 것으로 예상.
# prediction = knn.predict(X_new)
# # print('예측 : {}'.format(prediction))
# # print('예측한 타깃의 이름 : {}'.format(iris_dataset['target_names'][prediction]))


# #### 모델의 신뢰 -> 모델 평가하기

# y_pred = knn.predict(X_test)
# print('테스트 세트에 대한 예측값 :\n {}'.format(y_pred))
# print('테스트 세트에 대한 정확도 : {:.2f}'.format(np.mean(y_pred == y_test)))











# # ---------------------------(Day 19 - 210406)------------------












# # ################# 회귀분석 (Regression Analysis) #######################
# # Regression_210406.ipynb

# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(0)
# # y = 4X + 6 식을 근사(w1=4, w0=6). random 값은 Noise를 위해 만듬
# X = 2 * np.random.rand(100,1)
# y = 6 +4 * X+np.random.randn(100,1)

# # X, y 데이터 셋 scatter plot으로 시각화
# plt.scatter(X, y)
# # plt.show()

# # w1 과 w0 를 업데이트 할 w1_update, w0_update를 반환. 
# def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
#     N = len(y)      # 벡터의 길이
#     # 먼저 w1_update, w0_update를 각각 w1, w0의 shape와 동일한 크기를 가진 0 값으로 초기화
#     w1_update = np.zeros_like(w1)
#     w0_update = np.zeros_like(w0)
#     # 예측 배열 계산하고 예측과 실제 값의 차이 계산
#     y_pred = np.dot(X, w1.T) + w0       #np.matmul써도 되지만, 어차피 벡터계산이기에 dot을 사용
#     diff = y-y_pred                     #error function (실제값 - 예측값)
         
#     # w0_update를 dot 행렬 연산으로 구하기 위해 모두 1값을 가진 행렬 생성 
#     w0_factors = np.ones((N,1))         #초기값 ones로 세팅(N크기만큼 받아들이고,)

#     # w1과 w0을 업데이트할 w1_update와 w0_update 계산  (error function : mse(mean square error))
#     w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))        #summation_i^n (y-y_hat)(-x_i)
#     w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
#     return w1_update, w0_update




#     # 입력 인자 iters로 주어진 횟수만큼 반복적으로 w1과 w0를 업데이트 적용함. 
# def gradient_descent_steps(X, y, iters=10000):
#     # w0와 w1을 모두 0으로 초기화. 
#     w0 = np.zeros((1,1))
#     w1 = np.zeros((1,1))
    
#     # 인자로 주어진 iters 만큼 반복적으로 get_weight_updates() 호출하여 w1, w0 업데이트 수행. 
#     for ind in range(iters):    #w1_update = gradient descent 방법
#         w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
#         w1 = w1 - w1_update     #w1(new) = w1(old) - update  // update=0 -> new = old -> 최적의 값을 찾음
#         w0 = w0 - w0_update
              
#     return w1, w0



# def get_cost(y, y_pred):
#     N = len(y) 
#     cost = np.sum(np.square(y - y_pred))/N      # root(실제-예측) 다 더해서 저장한게 cost
#     return cost

# w1, w0 = gradient_descent_steps(X, y, iters=1000)   #  1000번 반복하여 최적의 값을 뽑아 그때의 cost를 출력
# print("w1:{0:.3f} w0:{1:.3f}".format(w1[0,0], w0[0,0]))
# y_pred = w1[0,0] * X + w0
# print('Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))

# plt.scatter(X, y)
# plt.plot(X,y_pred)
# # plt.show()


# # data가 적으면 gradient descent방법을 사용하겠지만, 
# # 데이터가 클 경우에는 미분자체가 계산량이 많아지거나 변수가 많아서 미분이 많아짐
# # 통계에서는 모집단(전체) -> 표본(sample)통계량 혹은 결론 -> Stochastic
# def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000):
#     w0 = np.zeros((1,1))
#     w1 = np.zeros((1,1))
#     prev_cost = 100000
#     iter_index =0
    
#     for ind in range(iters):
#         np.random.seed(ind)
#         # 전체 X, y 데이터에서 랜덤하게 batch_size만큼 데이터 추출하여 sample_X, sample_y로 저장 (참고 : https://medium.com/@shistory02/numpy-permutation-vs-shuffle-34fe56f0c246)
#         # Stochastic gradient descent / Mini-batch graident descnet (참고 : https://nonmeyet.tistory.com/entry/Batch-MiniBatch-Stochastic-%EC%A0%95%EC%9D%98%EC%99%80-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EC%98%88%EC%8B%9C)
#         stochastic_random_index = np.random.permutation(X.shape[0])
#         sample_X = X[stochastic_random_index[0:batch_size]]
#         sample_y = y[stochastic_random_index[0:batch_size]]
#         # 랜덤하게 batch_size만큼 추출된 데이터 기반으로 w1_update, w0_update 계산 후 업데이트
#         w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
#         w1 = w1 - w1_update
#         w0 = w0 - w0_update
    
#     return w1, w0


# w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
# print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
# y_pred = w1[0,0] * X + w0
# print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))



# ##################################################################



# #보폭에 따른 결과 비교
# import numpy as np
# import matplotlib.pyplot as plt

# lr_list = [0.001, 0.1, 0.3, 0.4]

# def get_derivative(lr_list):

#   w_old = 2
#   derivative = [w_old]

#   y = [w_old ** 2] # 손실 함수를 y = x^2로 정의함.

#   for i in range(1,10):
#     #먼저 해당 위치에서 미분값을 구함

#     dev_value = w_old **2

#     #위의 값을 이용하여 가중치를 업데이트
#     w_new = w_old - lr * dev_value
#     w_old = w_new

#     derivative.append(w_old) #업데이트 된 가중치를 저장 함,.
#     y.append(w_old ** 2) #업데이트 된 가중치의 손실값을 저장 함.

#   return derivative, y

# x = np.linspace(-2,2,50) 
# x_square = [i**2 for i in x]

# fig = plt.figure(figsize=(12, 7))

# for i,lr in enumerate(lr_list):
#   derivative, y =get_derivative(lr)
#   ax = fig.add_subplot(2, 2, i+1)
#   ax.scatter(derivative, y, color = 'red')
#   ax.plot(x, x_square)
#   ax.title.set_text('lr = '+str(lr))

# plt.show()











# ################# Sklearn Linear Regression Tutorial with Boston House Dataset #######################

# # import numpy as np
# # import pandas as pd

# ## Visualization Libraries
# # import seaborn as sns
# # import matplotlib.pyplot as plt



# #imports from sklearn library

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error



# # #To plot the graph embedded in the notebook
# # # %matplotlib inline  # VS에서는 필요x
# #         # load_boston의 column 의 의미
# #             # CRIM: 지역별 범죄 발생률
# #             # ZN: 25,000을 초과하는 거주 지역의 비율
# #             # INDUS: 비상업 지역의 넓이 비율
# #             # CHAS: 찰스강에 대한 더미 변수(강의 경계에 위치 여부에 따라 맞으면1, 아니면 0)
# #             # NOX: 일산화질소의 농도
# #             # RM: 거주할 수 있는 방의 개수
# #             # AGE: 1940년 이전에 건축된 소유 주택의 비율
# #             # DIS: 5개 주요 고용센터까지의 가중 거리
# #             # RAD: 고속도로 접근 용이도
# #             # TAX: 10,000달러당 재산세율
# #             # PTRATIO: 지역의 교사와 학생수의 비율
# #             # B: 지역의 흑인 거주 비율
# #             # LSTAT: 하위 계층의 비율
# #             # MEDV: 본인 소유의 주택 가격(중앙값)
# # #loading the dataset direclty from sklearn
# boston = datasets.load_boston()
# # print(type(boston))
# # print(boston.keys())
# # print(boston.data.shape)
# # print(boston.feature_names)

# bos = pd.DataFrame(boston.data, columns = boston.feature_names)     # row,columns 표형태인 데이터로 바꾸는 것
# bos['PRICE'] = boston.target
# # print(bos)
# # print(bos.isnull().sum())   #print(bos.isna().sum())
# # print(bos.describe())

# # sns.set(rc={'figure.figsize':(11.7, 8.27)})
# # plt.hist(bos['PRICE'], bins=30)
# # plt.xlabel("House prices in $1000")
# # # plt.show()


# # #Created a dataframe without the price col, since we need to see the correlation between the variables
# # correlation_matrix = bos.corr().round(2)
# # sns.heatmap(data=correlation_matrix, annot=True)    #숫자를 그림에 보이려면 annot=True
# # # plt.show()




# # plt.figure(figsize=(20,5))

# # features = ['LSTAT', 'RM']
# # target = bos['PRICE']

# # for i, col in enumerate(features):
# #     plt.subplot(1, len(features), i+1) # 1행2열을 만들고, 첫번째features를 그리고, 그 옆에 두번째features를 그려라
# #     x = bos[col]
# #     y = target
# #     plt.scatter(x, y, marker='o')
# #     plt.title("Variation in House prieces")
# #     plt.xlabel(col)
# #     plt.ylabel('"House prices in $1000"')
# # # plt.show()


# # # 선형분석 시작?
# # X_rooms = bos.RM        #여기서 RM 대신 타 변수를 넣어서 돌리면 됨
# # y_price = bos.PRICE

# # X_rooms = np.array(X_rooms).reshape(-1,1)
# # y_price = np.array(y_price).reshape(-1,1)

# # print(X_rooms.shape)
# # print(y_price.shape)

# # ######################################
# # # Train / Test 분리
# # X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size = 0.2, random_state=5)
# # print(X_train_1.shape)
# # print(X_test_1.shape)
# # print(Y_train_1.shape)
# # print(Y_test_1.shape)
# # ######################################
# #     # Sklearn의 Linear regression 클래스
# #         # input parameter
# #             # fit_intercept : 불 값으로, default = True. Intercept(절편) 값을 계산할 것인지 말지를 지정함. 만일 False로 지정하면 Intercept가 사용되지 않고 0으로 지정됨.
# #             # normalize : 불 값으로, 디폴트는 False임. fit_intercept가 False 인 경우에는 이 파라미터가 무시됨. 만일 True이면 회귀를 수행하기 전에 입력 데이터 세트를 정규화 함

# #         # Features
# #             # coef_ : fit() 메서드를 수행했을 때 회귀 계수가 배열 형태로 저장하는 속성. Shape는 (Target 값 개수, 피쳐 개수)
# #             # intercept_ : intercept 값\

# #     # 다중 공성선 문제(Multi-collinearity problem)
# #         # 모형의 일부 설명 변수가 다른 설명 변수와 상관 정도가 높아, 데이터 분석 시 부정적인 영향을 미치는 현상을 말함.
# #         # -> 어처구니 없는 해석을 하게 만듬
# #             # 피쳐 간의 상관관계가 매우 높은 경우 분산이 매우 커져서 오류에 매우 민감해짐. 이러한 현상을 다중 공선성(Multi-collinearity)
# #             # RMSE(Root Mean Squared Error) :MSE 값은 오류의 제곱을 구하므로 실제 오류 평균보다 더 커지는 특성이 있으므로 MSE에 루트를 씌움.
# #         # R2 = 예측값Variance / 실제값Varivance

# # reg = LinearRegression()
# # reg.fit(X_train_1, Y_train_1)     #data와 label을 같이 학습을 시킴

# # y_train_predict_1 = reg.predict(X_train_1)
# # rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
# # r2 = round(reg.score(X_train_1, Y_train_1),2)

# # print("The model performance for training set")
# # print("--------------------------------------")
# # print('RMSE is {}'.format(rmse))
# # print('R2 train score is {}'.format(r2))
# # print("\n")

# # ##################################Train 끝##################################

# # # model evaluation for test set
# # y_pred_1 = reg.predict(X_test_1)
# # rmse_1 = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
# # r2_1 = round(reg.score(X_test_1, Y_test_1),2)

# # print("The model performance for training set")
# # print("--------------------------------------")
# # print("Root Mean Squared Error: {}".format(rmse_1))
# # print("R2 test score : {}".format(r2_1))
# # print("\n")

# # prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) 
# # plt.scatter(X_rooms,y_price)
# # plt.plot(prediction_space, reg.predict(prediction_space), color = 'black', linewidth = 3)
# # plt.ylabel('value of house/1000($)')
# # plt.xlabel('number of rooms')
# # # plt.show()




# # #########################모든 변수를 넣고 회귀분석 돌리기######################
# # #################### Regression Model for All the variables######################
# # X = bos.drop('PRICE', axis = 1)
# # y = bos['PRICE']

# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # reg_all = LinearRegression()
# # reg_all.fit(X_train, y_train)

# # # model evaluation for training set
# # y_train_predict = reg_all.predict(X_train)
# # rmse_a = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
# # r2 = round(reg_all.score(X_train, y_train),2)
# # print("The model performance for training set")
# # print("--------------------------------------")
# # print('RMSE is {}'.format(rmse))
# # print('R2 score is {}'.format(r2))
# # print("\n")

# # y_pred = reg_all.predict(X_test)
# # rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
# # r2 = round(reg_all.score(X_test, y_test),2)

# # print("The model performance for training set")
# # print("--------------------------------------")
# # print("Root Mean Squared Error: {}".format(rmse))
# # print("R^2: {}".format(r2))
# # print("\n")

# # ################# FINISH - Sklearn Linear Regression Tutorial with Boston House Dataset #######################















# # # ---------------------------(Day 20 - 210407)------------------










# # Regression_210406.ipynb 파일 계속
# # import statsmodels.api as sm
# # import statsmodels.formula.api as smf
# X = bos.drop('PRICE', axis = 1)
# y = bos['PRICE']
# X_constant = sm.add_constant(X)

# #  요약통계량 OLD 방법 -> R^2이용해도 ok
# model_1 = sm.OLS(y, X_constant)
# lin_reg = model_1.fit()
# # print(lin_reg.summary())







################## 사이킷런 LinearRegression을 이용한 보스턴 주택 가격 예측 #################
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from scipy import stats
# from sklearn.datasets import load_boston

# # boston 데이터셋 로드
# boston = load_boston()

# # boston 데이터셋 DataFrame 변환
# bostonDF = pd.DataFrame(boston.data, columns = boston.feature_names)

# # boston dataset의 target array는 주택가격임. 이를 PRICE 컬럼으로 DataFrame에 추가
# bostonDF['PRICE'] = boston.target
# # print(bostonDF.shape)
# # print(bostonDF.head())

# # 2개의 행과 4개의 열을 가진 subplots를 이용, axs는 4x2개의 ax를 가짐.
# fog, axs = plt.subplots(figsize = (16,8), ncols=4, nrows=2)
# Im_features = ['RM', 'ZN', 'INDUS', 'NOX', 'AGE', 'PTRATIO', 'LSTAT', 'RAD']
# for i, feature in enumerate(Im_features):
#     row = int(i/4)
#     col = i%4
#     #시본의 regplot를 이용해 산점도와 선형회귀직선을 함께 표현
#     sns.regplot(x=feature, y='PRICE', data=bostonDF, ax=axs[row][col])
# # plt.show()














############ Polynomial Regression과 오버피팅/언더피팅 이해 ################
    # Polynomial Regression 이해
    # 단항 피처  [x1,x2] 를 degree = 2, 즉 2차 다항 피차로 변환한다면?  
    # (x1+x2)2 의 식 전개에 대응되는  [1,x1,x2,x1x2,x1^2,x2^2] 의 다항 피처들로 변환

    # 1차 단항 피처들의 값이  [x1,x2]=[0,1]  일 경우
    # 2차 다항 피처들의  [1,x1=0,x2=1,x1x2=0,x1^2=0,x2^2=1] 형태인 [1,0,1,0,0,1]로 변환

from sklearn.preprocessing import PolynomialFeatures
# import numpy as np



    # K-fold cross validation       출처: https://3months.tistory.com/321 [Deep Play]
    # K 겹 교차 검증(Cross validation)이란 통계학에서 모델을 "평가" 하는 한 가지 방법입니다. 
    # 소위 hold-out validation 이라 불리는 전체 데이터의 일부를 validation set 으로 사용해 
    # 모델 성능을 평가하는 것의 문제는 데이터셋의 크기가 작은 경우 테스트셋에 대한 
    # 성능 평가의 신뢰성이 떨어지게 된다는 것입니다. 만약 테스트셋을 어떻게 잡느냐에 따라 성능이 다르면, 
    # 우연의 효과로 인해 모델 평가 지표에 편향이 생기게 됩니다.

    # 이를 해결하기 위해 K-겹 교차 검증은 모든 데이터가 최소 한 번은 테스트셋으로 쓰이도록 합니다.
    # 아래의 그림을 보면, 데이터를 5개로 쪼개 매번 테스트셋을 바꿔나가는 것을 볼 수 있습니다. 
    # 첫 번째 Iteration에서는 BCDE를 트레이닝 셋으로, A를 테스트셋으로 설정한 후, 성능을 평가합니다. 
    # 두 번째 Iteration에서는 ACDE를 트레이닝셋으로, B를 테스트셋으로하여 성능을 평가합니다. 
    # 그러면 총 5개의 성능 평가지표가 생기게 되는데, 
    # 보통 이 값들을 평균을 내어 모델의 성능을 평가하게 됩니다. 
    # (아래 데이터는 모두 사실은 트레이닝 데이터입니다. 
    # Iteration이라는 상황안에서만 테스트셋이 되는 것입니다.) 
    # 이 때, 데이터를 몇 개로 쪼갰느냐가 K-겹 교차검증의 K가 됩니다.






########################### Kaggle _ Titanic data practice###############################
train = pd.read_csv('C:/Users/sundooedu/Documents/GitHub/SD_python_practice/machinelearning/Regression/Kaggle_titanic/train.csv')
test = pd.read_csv('C:/Users/sundooedu/Documents/GitHub/SD_python_practice/machinelearning/Regression/Kaggle_titanic/test.csv')
print(train.info())
print(train.describe())
print(train.isna().sum())
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()   #생존자를 카운트
    dead =  train[train['Survived']==0][feature].value_counts()      #사망자를 카운트
    df = pd.DataFrame([survived,dead])      #[생존자,사망자]를 DataFrame
    df.index = ['Survived', 'Dead'] #index화
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))         # stacked = True : 데이터를 쌓아서 보여줌

bar_chart('Sex')
bar_chart('Pclass')
# plt.show()

train_test_data = [train,test]

for dataset in train_test_data:
  dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand = False)

# one-hot encoding
title_mapping = {'Mr':0, "Miss":1, 'Mrs':2,'Master':3,
                 'Dr':3,'Rev':3,'Col':3,'Major':3,'Mlle':3,'Ms':3,'Sir':3,'Don':3,'Countess':3,
                 'Capt':3,'Lady':3,'Jonkheer':3,'Mme':3}
for dataset in train_test_data:
  dataset['Title'] = dataset['Title'].map(title_mapping)
bar_chart('Title')
# plt.show()

# 쓸모없는 데이터 정리
train.drop('Name',axis = 1, inplace = True)
test.drop('Name',axis = 1, inplace = True)
train.drop('Ticket',axis = 1, inplace = True)
test.drop('Ticket',axis = 1, inplace = True)
train.drop('Cabin',axis = 1, inplace = True)
test.drop('Cabin',axis = 1, inplace = True)
train.drop('Embarked',axis = 1, inplace = True)
test.drop('Embarked',axis = 1, inplace = True)



# 성별 인코딩
sex_mapping = {'male':0,'female':1}
for dataset in train_test_data:
  dataset['Sex'] = dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
print(train.head())     #이제 더이상 문자열데이터 없음

#missing Age를 각 Title에 대한 연령의 중간값으로 채움(Mr,Mrs,Miss,others)
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace = True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'),inplace = True)
print(train.isna().sum())   #이제 빈값 없음

# import matplotlib.pyplot as plt
# import seaborn as sns

#변수의 분포를 시각화하거나, 여러 변수들 사이의 상관관계를 여러개의 그래프로 쪼개서 표현할때 유용함
# FeactGrid는 Colum,row, hue를 통한 의미구분을 통해 총 3차원까지 구현이 가능함.
#aspect : subplot의 세로 대비 가로의 비율.
facet = sns.FacetGrid(train, hue ='Survived', aspect=4)
facet.map(sns.kdeplot,'Age',shade = True) # kde : 이차원 밀집도 그래프
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
sns.axes_style('dark')

plt.show()

# https://colab.research.google.com/drive/1_j49hszgmW0uqVC7_whBlxDf6nRunPgY?usp=sharing
# 이거보고 회귀분석 다시 해보자