import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm

# # 13일 시험 관련

# read/readline 시험x
# pandas/numpy 시험x(사용 시 감점)

# 뭐 하나 더 말했던거같은데

# # 사람평균(국영수)

# # add_all 문제(pdf p210)
# input:[1,2,3,4,5]
# output:15
# input:(1,2,3,4,5)
# output:15

# # 깃허브예시 중 3문제(1,5,6,7,8,11,12,17,18,19,20,21, 21x, 22,23)



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



# # 문제 5) 복리이자율7%로 1000만원 저금시 2000만원이 되기까지 몇년이 걸리는가?
# # 정답 : 11년

# A = 1000
# i = 0
# while A:
#         if A<2000:
#                 i += 1
#                 A = A*(1.07)
#                 print(f"{i}년차 {A}만원")
#         else:
#                 print(f"2000만원이 되기까지 {i}년이 걸렸습니다.")
#                 break


# 문제 6)
# 다음 문장에서 모음('aeiou')을 제거하시오 (Hint : List comprehension or for 문을 사용하세요.)
# list comprehension [저장할값 for 원소 in 반복가능한 객체]
# for + if문 : [저장할값 if 조건 else 저장할값   for 원소   in 반복가능한 객체 ]
A = ["Life is too short, you need python!"] 
search = ['a','e','i','o','u']
# A_1 = [A.replace('e','') for A in A]
# print(A_1)

def remove_vowels(text): # function names should start with verbs! :)
    return ''.join(ch for ch in text if ch.lower() not in 'aeiou')

print(remove_vowels(A))