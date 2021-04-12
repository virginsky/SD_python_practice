import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm

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


from sklearn.neighbors import KNeighborsClassifier
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