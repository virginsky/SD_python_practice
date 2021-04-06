import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 


# print(np.__version__)
# print(pd.__version__)
# # print(plt.__version__) #.pyplot는 버젼 안나옴
# print(sns.__version__)
# print(sklearn.__version__)

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











