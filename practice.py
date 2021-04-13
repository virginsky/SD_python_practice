import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import statsmodels.api as sm

# [4월 13일 파이썬 문제 출제 완료]
# 상 (이중 중첩문) ->새로운 문제 출제
# 리스트에서 *args를 사용할 수 있는지?(중)
# 깃허브예시 중 3문제(1,5,6,7,8,11,12,17,18,19,20,22,23)
#     중(github 문제)
#     중(github 문제)
#     하(github 문제)
# ** 패키지 혹은 내장되어 있는 함수 사용시,-10점 감점.




##############################################################################


# # 사람평균(국영수)
kor_score   = [39,69,20,100,80]
math_score  = [32,59,85,30,90]
eng_score   = [49,70,48,60,100]
midterm_score = [kor_score, math_score, eng_score]

# 정답(출제의도) ******* 난이도 높음 but 이해해야함
student_score = [0,0,0,0,0]     #빈칸 만들어 놓기
i=0
for subject in midterm_score :  #"과목선택" : student_score[i] row로 감 -> row로 데이터를 쓰였기 때문
  for score in subject :        #과목선택 후 
    student_score[i] += score   #각 학생마다 개별로 교과점수를 저장
      print(subject,score,'|: |',i,student_score)
        i += 1                      #학생 index 구분
          i = 0                     #과목이 바뀔떄마다 학생 인덱스 초기화
          else :                    #for가 끝났을때 작동
  a,b,c,d,e = student_score
  student_average = [a/3, b/3, c/3, d/3, e/3]   #학생별 점수를 언패킹
  print(student_average)


####################### # # add_all 문제(pdf p210) #######################3
# input:[1,2,3,4,5]
# output:15
# input:(1,2,3,4,5)
# output:15

def add_all(*inputs) : 
  s = 0
  for i in range(len(inputs)) :
    s += inputs[i]
  return s
print(add_all(1,2,3,4,5,6,7,8,9,10))

# #arg : tuple -> list로 받게하는 loop작성? - 왜 하는거여??
def add_all(*args) :
  s = 0
  for i in args:
    for j in i :
      s +=j
  return s
print(add_all([1,2,3,4,5]))



#############################################  깃허브  ###################################

# 문제 1)
# 고객의 개인정보보호를 위하여 이름을 비 식별화를 하려고 한다. 고객의 이름을 비 식별화 하여서 출력해보세요.
# 예시) 홍길동 -> 홍*동

names=['홍길동', '홍계월', '김철수', '이영희', '박첨지']
for i in range(len(names)):
      names[i] = names[i].replace(names[i][1],'*')
print(names)


# 문제 2)
# ‘students.txt’ 파일을 읽어서 딕셔너리로 학생들의 정보를 저장하라. 
# 해당 파일은 이름, 국어점수, 영어점수, 수학점수 순으로 되어있으며, 
# 각 컬럼의 구분자는 tab(\t)으로 이루어져있다. 
# 이름을 key로 하고, 국어점수, 영어점수, 수학점수를 
# 순서대로 담은 리스트를 value로 하는 딕셔너리로 저장하라. (encoding을 주의하시오.)
# pandas로 파일열떄 참고 : https://rfriend.tistory.com/250

# read 이용
with open('C:/Users/sundooedu/Documents/GitHub/SD_python_practice/파이썬/students.txt', 'r', encoding='UTF-8') as f:
    data = f.read()
print(data)

lines = data.split('\n')
print(lines)

students={}
for stu in lines:
    s = stu.split('\t')
    students[s[0]] = s[1:]
print(students)



# 문제 5) 복리이자율7%로 1000만원 저금시 2000만원이 되기까지 몇년이 걸리는가?
# 정답 : 11년

A = 1000
i = 0
while A:
        if A<2000:
                i += 1
                A = A*(1.07)
                print(f"{i}년차 {A}만원")
        else:
                print(f"2000만원이 되기까지 {i}년이 걸렸습니다.")
                break


# 문제 6)
# 다음 문장에서 모음('aeiou')을 제거하시오 (Hint : List comprehension or for 문을 사용하세요.)
# list comprehension [저장할값 for 원소 in 반복가능한 객체]
# for + if문 : [저장할값 if 조건 else 저장할값   for 원소   in 반복가능한 객체 ]
A = ["Life is too short, you need python!"] 
search = ['a','e','i','o','u']
for i in range(len(search)) :
    A = [A.replace(search[i],'') for A in A]
print(A)


# 문제 7)
# 리스트 중에서 홀수에만 2를 곱하여 저장하는 코드 작성
A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
B = []
for i in range(len(A)):
    if i%2 == 1:
        B.append(i*2)
print(B)

문제 8)
# LIst Comprehension을 이용해서 list1이 1부터 100사이의 8의 배수를 가지도록 만들어 보세요
list1 = [i for i in range(1,101) if i%8==0 ]
print(list1)


###########################gg#######################
문제 11)  
# 행렬의 덧셈은 행과 열의 크기가 같은 두 행렬의 같은 행, 같은 열의 값을 서로 더한 결과가 됩니다. 
# 2개의 행렬 arr1과 arr2를 입력받아, 행렬 덧셈의 결과를 반환하는 함수, solution을 완성해주세요.
# 제한 조건 행렬 arr1, arr2의 행과 열의 길이는 500을 넘지 않습니다.
# 문제풀이1) for 문 작성 
arr1 = np.arange(36).reshape(6,-1)
arr2 = np.arange(36).reshape(6,-1)
sum = 0
arr3 = np.zeros(36).reshape(6,-1)
def solution(arr3):
  for i in range(len(arr1)):
    for j in range(len(arr1)):
        arr3[i][j] = arr1[i][j]+arr2[i][j]
  print(arr3)
solution(arr3)
# 문제풀이2) zip 함수 이용 
ziparr=zip(arr1,arr2)
arr3 = np.zeros(36).reshape(6,-1)
def solution1(ziparr):  
  for i in ziparr:
    sum = np.array(i[0] + i[1])
    print(sum)
solution1(ziparr)
# 문제풀이3) list comprehension을 이용
arr3 = np.zeros(36).reshape(6,-1)
def solution3(arr1,arr2):
  ziparr = zip(arr1,arr2)
  for i in ziparr:
    sum = i[0] + i[1]
    print(sum)
solution3(arr1,arr2)
###########################gg#######################



# 문제 12)
# 1부터 입력받은 숫자 n 사이에 있는 소수의 개수를 반환하는 함수, solution을 만들어 보세요. 
# 소수는 1과 자기 자신으로만 나누어지는 수를 의미합니다. (1은 소수가 아닙니다.)
    # [제한 조건]
    # n은 2이상 1000000이하의 자연수입니다.
        # 입출력 예 #1
            # 1부터 10 사이의 소수는 [2,3,5,7] 4개가 존재하므로 4를 반환
        # 입출력 예 #2
            # 1부터 5 사이의 소수는 [2,3,5] 3개가 존재하므로 3를 반환
def solution(n):
    for i in range(2,n+1):
        chk = True
        for j in range(2,i):
            if i%j==0:
                chk=False 
                break
        if chk:
            a.append(i)
    print(f"2부터 {n}까지 소수의 갯수 : {len(a)} 개")

n = int(input("2이상 100000이하의 자연수를 입력하세요 : "))
a = []
print(solution(n))




###########################gg#######################
# ## 문제 17)
# 아래 조건에 따라 리스트를 회전하는 프로그램을 작성하시오.
# 조건 입력값은 한 행의 문자열로 주어지며, 각 값은 공백으로 구분된다.
# 첫 번째 값은 리스트를 회전하는 양과 방향(음수의 경우 좌측으로, 양수의 경우 우측으로 회전)이다.
# 첫 번째 값을 제외한 나머지 값은 리스트의 각 항목의 값이다.
# 회전된 리스트를 문자열로 출력한다.
# 구현에 이용할 자료구조에 대한 조건이나 제약은 없다.
# 입력되는 리스트의 항목의 개수는 유한한다.
    # 예 1)
    # 입력: 1 10 20 30 40 50
    # 출력: 50 10 20 30 40
    # 예 2)
    # 입력: 4 가 나 다 라 마 바 사
    # 출력: 라 마 바 사 가 나 다
    # 예 3)
    # 입력: -2 A B C D E F G
    # 출력: C D E F G A B
    # 예 4)
    # 입력: 0 똘기 떵이 호치 새초미
    # 출력: 똘기 떵이 호치 새초미
def program(x):
    x = x.split()
    a = int(x[0])
    del(x[0])
    c = x.copy()
    if a <= 0:
        for i in range(len(x)):
            b = i + a
            while b <= -len(x) :
                b += len(x)
            c[b] = x[i]
    else:
        for i in range(len(x)):
            b = i + a
            while b >= len(x):
                b -= len(x)
            c[b] = x[i]
    print(','.join(c).replace(',',' '))

print(program('3 똘기 떵이 호치 새초미'))
###########################gg#######################


###########################gg#######################
# # 문제 18)
# # 숫자 형태의 문자열을 콤마가 포함된 금액 표기식 문자열로 바꾸어주는 프로그램을 작성하시오. 
# # ※ 단, 프로그래밍 언어에서 지원하는 금액변환 라이브러리는 사용하지 말것
# # 예)
#     # 숫자 금액 1000 -> 1,000
#     # 20000000 -> 20,000,000
#     # 3245.24 -> 3,245.24

x = '198421.15'
def comma(x):
    a = ''
    s = []
    for i in range(len(x)):
        if x[i] == '.' :
            c=x[i:]
            b=x[:i]
            for j in reversed(range(len(b))):
              a = b[j] + a
              s.append(j)
              if len(s)%3 == 0 and j!= 0:
                a = ',' +a
            return (a + c)

        if x.find('.') == -1:
          b = x
          for j in reversed(range(len(b))):
              a = b[j] + a
              s.append(j)
              if len(s)%3 == 0 and j!= 0:
                a = ',' +a
          return a
          

print(comma(x))
###########################gg#######################


# 문제 19)
# 입력한 숫자의 약수를 모두 찾고, 갯수를 출력할 것
n = int(input("약수를 찾을 숫자를 입력하시오 : "))
list=[]
for i in range(1,n+1):
    if n%i==0 :
        list.append(i)
print(f"입력한 {n}의 약수는 {list}이며, 갯수는 {len(list)}개 입니다.")



# 문제 20)
# 2^15 = 32768 의 각 자리수를 더하면 3 + 2 + 7 + 6 + 8 = 26 입니다. 
# 2^1000의 각 자리수를 모두 더하면 얼마입니까?
N = int(input("자연수를 입력하세요 : "))
b = str(N)
sum = 0
for i in range(len(b)):
    sum += int(b[i])
print(sum)



# 문제 22)  ## 걍 구글링.. 
# 2진법이란, 어떤 자연수를 0과 1로만 나타내는 것이다.
# 예를 들어 73은 64(2^6)+8(2^3)+1(2^0)이기 때문에 1001001으로 표현한다.
# 어떤 숫자를 입력받았을 때 그 숫자를 2진법으로 출력하는 프로그램을 작성하시오
a = int(input("2진수로 변환할 자연수를 입력하세요 : "))
binary = ''
while a > 0:
    div = a // 2
    mod = a % 2
    a = div 
    binary += str(mod)
print(binary[::-1])


###########################gg#######################
# 문제 23) 가성비 최대화
# 기계를 구입하려 하는데 이 기계는 추가 부품을 장착할 수 있다. 추가 부품은 종류당 하나씩만 장착 가능하고, 모든 추가 부품은 동일한 가격을 가진다.
# 원래 기계의 가격과 성능, 추가 부품의 가격과 각 부품의 성능이 주어졌을 때, 추가 부품을 장착하여 얻을 수 있는 최대 가성비를 정수 부분까지 구하시오(가격 및 성능은 상대적인 값으로 수치화되어 주어진다).
#     e.g.)
#     원래 기계의 가격 : 10
#     원래 기계의 성능 : 150
#     추가 부품의 가격 : 3
#     추가 부품의 성능 : 각각 30, 70, 15, 40, 65
#     추가 부품을 장착하여 얻을 수 있는 최대 가성비 : 17.81... → 17
#     (성능이 70과 65인 부품을 장착하면 됨)
# 원래 기계의 성능 / 추가부품의 가격
# 가성비 측도=(원래 기계의 성능 + 추가 부품의 성능 / 성능 + 가격)

a = 10
b= 150
c = 3
d= [30,70,15,40,65]
f = 0
d.sort(reverse = True)
x = []
for i in d:
  f += i
  x.append(i)
  if (f+b) / (c*len(x) + a) > ((f-i) + b)/(c*(len(x)-1) + a):
    print(i)
    print(str((f+b) / (c*len(x) + a))[:2])
 
###########################gg#######################


# [중] 중첩 루프를 이용해 우유 배달을 하는 프로그램을 작성하시오. 
    # (단 아래에서 Unpaid 리스트는 우유 값이 미납된 세대에 대한 정보를 포함하고 있는데, 
    # 해당 세대에는 우유를 배달하지 않아야 합니다.)
Apart = [[101,102,103,104],[201,202,203,204],[301,302,303,304],[401,402,403,404]]
Unpaid = [101,204,302,402]

for i in range(len(Apart)):
    for j in range(len(Apart)):
        if Apart[i][j] in Unpaid:
            print(f'우유 배달해야 하지 말아야 할 곳 {Apart[i][j]}')
        else:
            print(f'우유 해야 할 곳 {Apart[i][j]}')
