import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm
import graphviz as grp
import mglearn

'''

# 분류 알고리즘
# 분류(Classification)는 학습데이터로 주어진 데이터의 피처와 레이블 값(결정 값, 클래스 값)을 머신러닝 알고리즘으로 학습해 모델을 생성하고, 이렇게 생성된 모델에 새로운 데이터 값이 주어졌을 때 미지의 레이블값을 예측 하는것

# 대표적인 분류 알고리즘

# 베이즈(Bayes) 통계와 생성 모델에 기반한 나이브 베이즈(Naive Bayes)
# 독립변수와 종속변수의 선형 관계성에 기반한 로지스틱 회귀(Logistic Regression)
# 데이터 균일도에 따른 규칙 기반의 결정트리(Dicision Tree)
# 개별 클래스 간의 최대 분류 마진을 효과적으로 찾아주는 서포트 벡터 머신(Support Vector machine)
# 근접 거리를 기준으로 하는 최소 근접(Nearest Neighbor) 알고리즘
# 심층 연결 기반의 신경망(Neural Network)
# 서로 다른(또는 같은) 머신러닝 알고리즘을 결합한 앙상블(Ensemble)






#########################################################################


# K-Nearest Neighbors - iris 꽃 종류 분류를 위한 시각화

from sklearn.datasets import load_iris
iris = load_iris()
# print(iris)
# print(iris.keys())
# print(iris.feature_names)
# print(iris.target)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size = 0.2,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.neighbors import KNeighborsClassifier      #classification -> Classifier   / Regression -> Regressor
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,y_train)
print('accuracy : {:.2f}'.format(knn.score(X_test,y_test)))

# - 결론
#   - KNN의 장단점 그리고 언제 활용을 해야하는지 다음과 같이 심플하게 정리해 보았습니다.

# - 장점
#   - 쉬운 모델, 쉬운 알고리즘과 이해 (입문자가 샘플데이터를 활용할 때 좋음)
#   - 튜닝할 hyperparameter 스트레스가 없음
#   - 초기 시도해보기 좋은 시작점이 되는 모델

# - 단점
#   - 샘플 데이터가 늘어나면 예측시간도 늘어나기 때문에 매우 느려짐
#   - pre-processing을 잘하지 않으면 좋은 성능을 기대하기 어려움
#   - feature가 많은(수 백개 이상) 데이터셋에서는 좋은 성능을 기대하기 어려움
#   - feature의 값이 대부분 0인 데이터셋과는 매우 안좋은 성능을 냄
#   - 결론, kaggle과 현업에서는 더 좋은 대안들이 많기 때문에 자주 쓰이는 알고리즘은 아닙니다. 
#     하지만, 초기에 학습을 목표로 해볼 필요는 있습니다!









#########################################################################


# # Navie Bayes(나이브 베이즈 분류)

# 날씨에 따라 축구를 했는지 안했는지에 대한 과거 데이터입니다. 
    # 이 과거 데이터를 먼저 Training 시켜 모델을 만든 뒤 그 모델을 기반으로 어떤 날씨가 주어졌을 때 
    # 축구를 할지 안 할지 판단하는 것이 목적입니다.
    # Frequency Table은 주어진 과거 데이터를 횟수로 표현한 것입니다. 
    # Likelihood Table 1은 각 Feature (여기서는 날씨)에 대한 확률, 
    # 각 Label (여기서는 축구를 할지 말지 여부)에 대한 확률을 구한 것입니다. 
    # Likelihood Table 2는 각 Feature에 대한 사후 확률을 구한 것입니다.


# ## Feature가 하나일 때 나이브 베이즈 분류

# Q1) 날씨가 Overcast(흐린)일때,
# P(Yes|overcast) = P(overcast|Yes) * P(Yes)/P(overcast)
# 1. 사전확률(Prior Probability)
# P(overcast) = 4/14 = 0.29
# P(Yes) = 9/14 = 0.64
# 2. 사후 확률(posterior probability)
# P(Overcast|NO) = 0/9 = 0
# 3. 베이즈 정리 공식에 대입

# P(NO|overcast) = P(overcast|NO) * (P(No) / P(overcast)) = 0*(0.36 / 0.29) = 0
    # 즉, 날씨가 Overcast일 때 축구를 할 확률이 0이라는 뜻입니다.

# P(Yes|Overcast) = 0.98, P(No|Overcast) = 0입니다. 
    # 즉, 날씨가 Overcast일 때 축구를 하는 확률은 0.98, 축구를 하지 않을 확률은 0입니다. 
    # 두 확률을 비교한 뒤 더 높은 확률의 Label로 분류를 하면 됩니다. 
    # 두 확률을 비교했을 때 'Yes' Label의 확률이 0.98로 더 높습니다. 
    # 따라서 나이브 베이즈 분류기는 날씨가 Overcast일 때 축구를 할 것이라고 판단합니다.




# ## Feature가 Multiple일 때 나이브 베이즈 분류

# P(Paly=Yes | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play=Yes) P(Play=Yes) / P(Weather=Overcast, Temp=Mild)
# P(Weather=Overcast, Temp=Mild | Play=Yes) = P(Overcast|Yes) P(Mild|Yes)
# P(Weather=Overcast, Temp=Mild) = P(Weather=Overcast) P(Temp=Mild) = (4/14) * (6/14) = 0.1224
# 1. 사전 확률
    # P(Yes) = 9/14 = 0.64
# 2. 사후 확률
    # P(Overcast|Yes) = 4/9 = 0.44
    # P(Mild|Yes) = 4/9 = 0.44
# 3. 베이즈 공식에 대입
    # P(Weather=Overcast, Temp=Mild | Play=Yes) = P(Overcast|Yes) P(Mild|Yes) = 0.44 * 0.44 = 0.1936
    # P(Paly=Yes | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play=Yes) P(Play=Yes) / P(Weather=Overcast, Temp=Mild)
    # = 0.1936 * 0.64 / 0.1224 = 1

# 문제 2. 날씨가 overcast, 기온이 Mild일 때 경기를 하지 않을 확률은?
# P(Paly=No | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play=No) P(Play=No) / P(Weather=Overcast, Temp=Mild)
# P(Weather=Overcast, Temp=Mild | Play=No) = P(Overcast|Yes) P(Mild|No)
# 1. 사전 확률
    # P(No) = 5/14 = 0.36
# 2. 사후 확률
    # P(Overcast|No) = 0/5 = 0
    # P(Mild|No) = 2/5 = 0.4
# 3. 베이즈 공식에 대입
    # P(Weather=Overcast, Temp=Mild | Play=No) = P(Overcast|No) P(Mild|No) = 0 * 0.4 = 0
    # P(Paly=No | Weather=Overcast, Temp=Mild) = P(Weather=Overcast, Temp=Mild | Play=No) P(Play=No) / P(Weather=Overcast, Temp=Mild)
    # = 0 * 0.36 / 0.1224 = 0

# 축구를 할 확률은 1이고, 축구를 하지 않을 확률은 0입니다. 
# 축구를 할 확률이 더 크기 때문에 날씨가 Overcast이고 기온이 Mild일 때는 축구를 할 것이라고 분류합니다. 
# 이렇듯 나이브 베이즈는 베이즈 정리를 활용하여 확률이 더 큰 Label로 분류를 합니다.






#########################################################################


# Scikit-learn을 활용한 나이브 베이즈 분류기 구축

# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny',
'Rainy','Sunny','Overcast','Overcast','Rainy']
temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']

play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


# Import LabelEncoder
from sklearn import preprocessing

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print(weather_encoded)


# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)
print("Temp:",temp_encoded)
print("Play:",label)


#Combinig weather and temp into single listof tuples
features = zip(weather_encoded,temp_encoded)
features = list(features)
print(features)


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted) # 1: Yes

# 결론
    # 아까 베이즈 정리를 활용하여 직접 계산했을 때, 
    # 날씨가 Overcast, 기온이 Mild일 때 play로 예측을 했습니다. 
    # sklearn의 naive_bayes에서도 동일한 결과가 나옵니다. 1이 Play를 한다입니다.








#########################################################################

# Label이 여러개인 나이브 베이즈

#Import scikit-learn dataset library
from sklearn import datasets

#Load dataset
wine = datasets.load_wine()

# print the names of the 13 features
print("Features: ", wine.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", wine.target_names)

print(wine.data.shape)
print(wine.data[:5])
print(wine.target)


# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=109)

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


# 나이브 베이즈의 장단점
    # - 장점
        # 1. 간단하고, 빠르며, 정확한 모델입니다.
        # 2. computation cost가 작습니다. (따라서 빠릅니다.)
        # 3. 큰 데이터셋에 적합합니다.
        # 4. 연속형보다 이산형 데이터에서 성능이 좋습니다.
        # 5. Multiple class 예측을 위해서도 사용할 수 있습니다.
    # - 단점
    #  - feature 간의 독립성이 있어야 합니다. 
    #       하지만 실제 데이터에서 모든 feature가 독립인 경우는 드뭅니다. 
    #       장점이 많지만 feature가 서로 독립이어야 한다는 크리티컬한 단점이 있습니다.
    #  - feature간 독립성이 있다는 말은 feature간에 서로 상관관계가 없다는 뜻입니다. 
    #       X1과 X2라는 feature가 있을 때 X1이 증가하면 X2도 같이 증가한다고 합시다. 
    #       그럼 X1과 X2는 서로 상관관계가 있다고 말할 수 있고, 이는 X1과 X2가 독립성이 없다는 뜻입니다. 
    #       X1과 X2가 독립성이 있으려면 X1이 증가하든 말든, X2에는 아무런 영향을 미치지 않아야 합니다. 
    #       하지만 우리가 얻을 수 있는 데이터에서는 feature간의 독립성이 항상 보장되지는 않습니다. 
    #       나이브 베이즈 모델은 feature간 독립성이 있다는 가정하에 성립되는 모델이기 때문에 
    #       실생활에서 바로 적용하기는 어려움있습니다.

'''







#########################################################################









'''

 **결정트리와 앙상블**
 - 결정 트리는 매우 쉽고 유연하게 적용될 수 있는 알고리즘. 데이터의 스케일링이나 
    정규화 등의 사전 가공의 영향이 매우 적음. 하지만 예측 성능을 향상시키기 위해 복잡한 규칙 구조를 가져야 하여, 
    이로 인한 과적합(Overfitting)이 발생해 반대로 예측 성능이 저하될 수도 있다는 단점이 있습니다.
 - 하지만 이러한 단점이 앙상블 기법에서는 오히려 장점으로 작용. 
    앙상블은 매우 많은 여러개의 약한 학습기(즉, 예측 성능이 상대적으로 떨어지는 학습 알고리즘)을 결합해 
    확률적 보완과 오류가 발생한 부분에 대한 가중치를 계속 업데이트 하면서 예측 성능을 향상시키는데, 
    결정트리가 좋은 약한 학습기가 되기 때문(GBM, LightGBM etc)

# 결정트리(Dicision Tree)
- 일반적으로 쉽게 표현하는 방법은 if/else 로 스무고개 게임을 한다고 생각하면 된다.
- 결정 트리(Decision Tree, 의사결정트리, 의사결정나무라고도 함)는 분류(Classification)와 
  회귀(Regression) 모두 가능한 지도 학습 모델 중 하나입니다. 
  결정 트리는 스무고개 하듯이 예/아니오 질문을 이어가며 학습합니다. 
  매, 펭귄, 돌고래, 곰을 구분한다고 생각해봅시다. 매와 펭귄은 날개를 있고, 돌고래와 곰은 날개가 없습니다. 
  '날개가 있나요?'라는 질문을 통해 매, 펭귄 / 돌고래, 곰을 나눌 수 있습니다. 
  매와 펭귄은 '날 수 있나요?'라는 질문으로 나눌 수 있고, 돌고래와 곰은 '지느러미가 있나요?'라는 질문으로 나눌 수 있습니다. 
    <img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FwlH1u%2FbtqwWZI9Xen%2FkFJDjGSFJAPxhyatC3Xhs0%2Fimg.png' width=700 height=300>

특정 기준(질문)에 따라 데이터를 구분하는 모델을 결정 트리 모델이라고 함.
    <img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdy5OwG%2FbtqDwdHofoT%2FNtDy9lqXkhWTRTwEz6txd0%2Fimg.png' width=700 height=300>
많은 규칙이 있으면 분류를 결정하는 방식이 복잡해짐-> 과적합으로 이어지기 쉬움

**결론 :트리가 깊이가 깊어질수록 결정 트리의 예측 성능이 저하될 가능성이 높음.** 
참고 : https://jaaamj.tistory.com/21



### 데이터의 균일도

    <img src='https://blog.kakaocdn.net/dn/dhoo7N/btqDvzqEhPH/QdOsfkqc2hcwHISIh0peo1/img.png' width=700 height=300>
참고 : https://jaaamj.tistory.com/21

위 그림에서는 C > B > A 순으로 균일도가 높다고 할 수 있습니다. 
C는 모두 파란색 공으로 데이터가 모두 균일한 상태입니다. 
B의 경우는 일부의 하얀색 공을 가지고 있지만 대부분 파란색 공으로 구성되어 있어 C다음으로 균일도가 높습니다

이러한 데이터 세트에서 균일도는 데이터를 구분하는데 있어서 필요한 정보의 양에 영향을 미치게 됩니다.

정보의 균일도를 측정하는 대표적인 방법에는 엔트로피를 이용한 정보 이득(Information Gain)지수와 지니계수가 있습니다.

### 불순도(Impurity)
불순도(Impurity)란 해당 범주 안에 서로 다른 데이터가 얼마나 섞여 있는지를 뜻합니다. 
아래 그림에서 위쪽 범주는 불순도가 낮고, 아래쪽 범주는 불순도가 높습니다. 
바꾸어 말하면 위쪽 범주는 순도(Purity)가 높고, 아래쪽 범주는 순도가 낮습니다. 
위쪽 범주는 다 빨간점인데 하나만 파란점이므로 불순도가 낮다고 할 수 있습니다. 
반면 아래쪽 범주는 5개는 파란점, 3개는 빨간점으로 서로 다른 데이터가 많이 섞여 있어 불순도가 높습니다.
    <img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqLXhZ%2FbtqwWyZl6iV%2FUZnQbf9L5HAFzf6hFfxK71%2Fimg.png' width=400 height=300>


한 범주에 하나의 데이터만 있다면 불순도가 최소(혹은 순도가 최대)이고, 
한 범주 안에 서로 다른 두 데이터가 정확히 반반 있다면 불순도가 최대(혹은 순도가 최소)입니다. 
결정 트리는 불순도를 최소화(혹은 순도를 최대화)하는 방향으로 학습을 진행합니다.

엔트로피(Entropy)는 불순도(Impurity)를 수치적으로 나타낸 척도입니다. 엔트로피가 높다는 것은 불순도가 높다는 뜻이고, 
엔트로피가 낮다는 것은 불순도가 낮다는 뜻입니다. 엔트로피가 1이면 불순도가 최대입니다. 
즉, 한 범주 안에 서로 다른 데이터가 정확히 반반 있다는 뜻입니다. 엔트로피가 0이면 불순도는 최소입니다. 
한 범주 안에 하나의 데이터만 있다는 뜻입니다. 엔트로피를 구하는 공식은 아래와 같습니다
<img src='https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FpL6pO%2FbtqwVDN1V94%2FTYgn5iFrPTfgdVwZhxVKl1%2Fimg.png' width=500 height=100>

**Pi = 한 영역 안에 존재하는 데이터 가운데 범주 i에 속하는 데이터의 비율**


#### 정보 이득(Information Gain)
정보 이득은 엔트로피라는 개념을 기반으로 함. 엔트로피는 주어진 데이터 집합의 혼잡도를 의미하는데, 
서로 다른 값이 섞여 있으면 엔트로피가 높고, 같은 값이 섞여 있으면 엔트로피가 낮음. 정보 이득 지수는 1에서 엔트로피 지수를 뺀 값. 
즉, 1-엔트로피 지수. 결정트리는 이 정보 이득 지수로 분할 기준을 정함. 즉, 정보 이득이 높은 속성을 기준으로 분할

#### 지니 계수(Gini Index)
원래 결제학에서 불평등 지수를 나타낼 때 사용하는 계수. 
경제학자인 코라도 지니(Corrado Gini)의 이름에서 딴 계수로서 0이 가장 평등하고 1로 갈수록 불평등함. 
머신러닝에 적용될 때는 지니 계수가 낮을수록 데이터 균일도가 높은 것으로 해석 되어, 계수가 낮은 속성을 기준으로 분할.
순도와 관련해 부연설명을 드리면 A 영역에 속한 모든 레코드가 동일한 범주에 속할 경우(=불확실성 최소=순도 최대) 엔트로피는 0입니다. 
반대로 범주가 둘뿐이고 해당 개체의 수가 동일하게 반반씩 섞여 있을 경우(=불확실성 최대=순도 최소) 엔트로피는 1의 값을 갖습니다. 
엔트로피 외에 불순도 지표로 많이 쓰이는 지니계수(Gini Index) 공식은 아래와 같습니다.
<img src='https://qph.fs.quoracdn.net/main-qimg-690a5cee77c5927cade25f26d1e53e77' width=500 height=500>

아래는 범주가 두 개일 때 한쪽 범주에 속한 비율(p)에 따른 불순도의 변화량을 그래프로 나타낸 것입니다. 
보시다시피 그 비율이 0.5(두 범주가 각각 반반씩 섞여 있는 경우)일 때 불순도가 최대임을 알 수가 있습니다. 
오분류오차(misclassification error)는 따로 설명드리지 않은 지표인데요, 오분류오차는 엔트로피나 지니계수와 더불어 불순도를 측정할 수 있긴 하나 
나머지 두 지표와 달리 미분이 불가능한 점 때문에 자주 쓰이지는 않는다고 합니다.
<img src='http://i.imgur.com/n3MVwHW.png' width=500 height=500>


#####  결정트리 주요 hyperparameter

1. max_depth
 - 트리의 최대 깊이를 규정
 - defualt 는 None.None으로 설정하면 완벽하게 클래스 결정 값이 될 때까지 깊이를 계속 키우며 
      분할하거나 노드가 가지는 데이터 개수가 min_samples_split 보다 작아질 때까지 게속 깊이를 증가시킴.
 - 깊이가 깊어지면, min_samples_split 설정대로 최대 분할하여 과적합할 수 있으므로 적절한 값으로 제어 필요

2. max_features
 - 최적의 분할을 위해 고려할 최대 피쳐 개수, 디폴트는 None으로 데이터 세트의 모든 피처를 사용해 분할 수행.
 - int 형으로 지정하면 대상 피처의 개수, float 형으로 지정하면 전체 피처 중 대상 피처의 퍼센트임.
 - 'sqrt'는 전체 피처 중 sqrt(전체 피처 개수) 즉 ${\sqrt {전체 피처}}$개수 만큼 선정.
 - 'auto'로 지정하면 sqrt와 동일
 - 'log'는 전체 피처 중 log2(전체 피처 개수) 선정
 - 'None'은 전체 피처 선정

3. min_samples_split
 - 노드를 분할하기 위한 최소한의 샘플 데이터 수로 과적합을 제어하는 데 사용됨.
 - 디폴트는 2이고 작게 설정할수록 분할되는 노드가 많아져서 과적합 가능성 증가
 - 과적합을 제어, 1로 설정하는 경우 분할되는 노드가 많아져서 과적합 가능성 증가

4. min_samples_leaf
 - 말단 노드(leaf)가 되기 위한 최소한의 샘플 데이터 수
 - min_samples_split와 유사하게 과적합 제어 용도. 그러나 비대칭적(imbalanced) 데이터의 경우 
      클래스의 데이터가 극도로 작을 수 있으므로, 이 경우는 작게 설정 필요.

5. max_leaf_nodes
 - 말단 노드(Leaf)의 최대 개수.

**결정트리를 그리가 위해 graphviz를 이용함**\

설치 참고 : https://wiznxt.tistory.com/776

///////////////////////////////////

Graphviz 설치참조 화면

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import tree

import pandas as pd
from sklearn.datasets import load_iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

# Step 1: Import the model you want to use
# This was already imported earlier in the notebook so commenting out
#from sklearn.tree import DecisionTreeClassifier
# Step 2: Make an instance of the Model
clf = DecisionTreeClassifier(max_depth = 2, 
                             random_state = 0)
# Step 3: Train the model on the data
clf.fit(X_train, Y_train)
# Step 4: Predict labels of unseen (test) data
# Not doing this step in the tutorial
# clf.predict(X_test)

tree.plot_tree(clf);
plt.show()
'''




'''

# 결정 트리 모델의 시각화(Decision Tree Visualiozation)

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# DecisionTree Classifier 생성
dt_clf = DecisionTreeClassifier(min_samples_leaf=6,random_state=156)

# 붓꽃 데이터를 로딩하고, 학습과 테스트 데이터 셋으로 분리
iris_data = load_iris()
X_train , X_test , y_train , y_test = train_test_split(iris_data.data, iris_data.target,
                                                       test_size=0.2,  random_state=11)

# DecisionTreeClassifer 학습. 
dt_clf.fit(X_train , y_train)

from sklearn.tree import export_graphviz

# export_graphviz()의 호출 결과로 out_file로 지정된 tree.dot 파일을 생성함. 
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names , \
feature_names = iris_data.feature_names, impurity=True, filled=True)

import graphviz


# 위에서 생성된 tree.dot 파일을 Graphviz 읽어서 Jupyter Notebook상에서 시각화 

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

'''




'''
# 유방암 데이터로 살펴보는 Dicision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42) # stratify : target:
tree = DecisionTreeClassifier(random_state=0, max_depth=5)  # 여기에 파라미터를 조절해서 여러가지 돌려볼 것
tree.fit(X_train, y_train)
print("Accuracy on training set : {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set : {:.3f}".format(tree.score(X_test, y_test)))
'''



# 컴퓨터 메모리 각격 동향 데이터셋
# x축이 날짜,y축은 해당 년도의 램(RAM) 1메가바이트당 가격
import os
import pandas as pd
import numpy as np

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
# plt.show()


from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
# use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# predict prices based on date
X_train = data_train.date[:, np.newaxis]
# we use a log-transform to get a simpler relationship of data to target
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()
plt.show()