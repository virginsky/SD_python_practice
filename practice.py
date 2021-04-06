import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm

# # 문제 1)
# # 고객의 개인정보보호를 위하여 이름을 비 식별화를 하려고 한다. 고객의 이름을 비 식별화 하여서 출력해보세요.
# # 예시) 홍길동 -> 홍*동

# names=['홍길동', '홍계월', '김철수', '이영희', '박첨지']
# for s in names:
#     n_name = s[0] + '*' + s[2]
#     print(n_name)


# # 문제 2)
# # ‘students.txt’ 파일을 읽어서 딕셔너리로 학생들의 정보를 저장하라. 
# # 해당 파일은 이름, 국어점수, 영어점수, 수학점수 순으로 되어있으며, 
# # 각 컬럼의 구분자는 tab(\t)으로 이루어져있다. 
# # 이름을 key로 하고, 국어점수, 영어점수, 수학점수를 
# # 순서대로 담은 리스트를 value로 하는 딕셔너리로 저장하라. (encoding을 주의하시오.)
# # pandas로 파일열떄 참고 : https://rfriend.tistory.com/250

# # read 이용
# with open('C:/Users/sundooedu/Documents/GitHub/SD_python_practice/파이썬/students.txt', 'r', encoding='UTF-8') as f:
#     data = f.read()
# print(data)

# lines = data.split('\n')
# print(lines)

# students={}
# for stu in lines:
#     s = stu.split('\t')
#     students[s[0]] = s[1:]
# print(students)

# # readlines 이용
with open('C:/Users/sundooedu/Documents/GitHub/SD_python_practice/파이썬/students.txt', 'r', encoding='UTF-8') as f:
        data1 = f.readlines()
print(data1)
    