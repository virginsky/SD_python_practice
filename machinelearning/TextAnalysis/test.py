# sent = '김철수는 극중 두 인격의 사나이 이광수 역을 맡았다. 철수는 한국 유일의 태권도 전승자를 가리는 결전의 날을 앞두고 10년간 함께 훈련한 사형인 유연재(김광수 분)를 찾으러 속세로 내려온 인물이다.'
# new_sent = sent.replace(" ", '') # 띄어쓰기가 없는 문장 임의로 만들기
# print(new_sent)

# #원 문장과 비교
# from pykospacing import spacing
# kospacing_sent = spacing(new_sent)
# print(sent)
# print(kospacing_sent)



# # Py-Hanspell
# # Py-Hanspell은 네이버 한글 맞춤법 검사기를 바탕으로 만들어진 패키지입니다.
# # pip install git+https://github.com/ssut/py-hanspell.git
# from hanspell import spell_checker

# sent = "맞춤법 틀리면 외 않되? 쓰고싶은대로쓰면돼지 "
# spelled_sent = spell_checker.check(sent)

# hanspell_sent = spelled_sent.checked
# print(hanspell_sent)

# spelled_sent = spell_checker.check(new_sent)

# hanspell_sent = spelled_sent.checked
# print(hanspell_sent)
# print(kospacing_sent) # 앞서 사용한 kospacing 패키지에서 얻은 결과

# # 이 패키지는 띄어쓰기 또한 보정합니다. PyKoSpacing에 사용한 예제를 그대로 사용해봅시다.
# spelled_sent = spell_checker.check(new_sent)

# hanspell_sent = spelled_sent.checked
# print(hanspell_sent)
# print(kospacing_sent) # 앞서 사용한 kospacing 패키지에서 얻은 결과




# # SOYNLP를 이용한 단어 토큰화
# # soynlp는 품사 태깅, 단어 토큰화 등을 지원하는 단어 토크나이저입니다. 비지도 학습으로 단어 토큰화를 한다는 특징을 갖고 있으며, 데이터에 자주 등장하는 단어들을 단어로 분석합니다. soynlp 단어 토크나이저는 내부적으로 단어 점수 표로 동작합니다. 이 점수는 응집 확률(cohesion probability)과 브랜칭 엔트로피(branching entropy)를 활용합니다.
# # 신조어 문제
# # soynlp를 소개하기 전에 기존의 형태소 분석기가 가진 문제는 무엇이었는지, SOYNLP가 어떤 점에서 유용한지 정리해봅시다. 기존의 형태소 분석기는 신조어나 형태소 분석기에 등록되지 않은 단어 같은 경우에는 제대로 구분하지 못하는 단점이 있었습니다.
# # pip install soynlp
# # pip install konlpy
# from pykospacing import spacing
# import jpype
# from konlpy.tag import Okt 
# okt = Okt()
# print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

# from konlpy.tag import Kkma  
# kkma=Kkma()  
# print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
# print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
# print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  



# # # powershell에서 wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
# import codecs
# with codecs.open("ratings_train.txt", encoding='utf-8') as f:
#     data = [line.split('\t') for line in f.read().splitlines()]
#     data = data[1:]   # header 제외
# # print(data)
# docs = [row[1] for row in data]
# print(docs)
# print(len(docs))

# import warnings
# warnings.simplefilter("ignore")

# from konlpy.tag import Okt

# tagger = Okt()

# def tokenize(doc):
#     tokens = ['/'.join(t) for t in tagger.pos(doc)]
#     return tokens

# from nltk.util import ngrams
# from tqdm import tqdm
# sentences = []
# for d in tqdm(docs):
#     tokens = tokenize(d)
#     bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SS", right_pad_symbol="SE")
#     sentences += [t for t in bigram]
# print(sentences[:30])

# from nltk.probability import ConditionalProbDist, ConditionalFreqDist, MLEProbDist
# cfd = ConditionalFreqDist(sentences)
# cpd = ConditionalProbDist(cfd, MLEProbDist)

# def korean_most_common(c, n, pos=None):
#     if pos is None:
#         return cfd[tokenize(c)[0]].most_common(n)
#     else:
#         return cfd["/".join([c, pos])].most_common(n)

# print(korean_most_common("나", 10))
# print(korean_most_common("의", 10))
# print(korean_most_common(".", 10, "Punctuation"))

# def korean_bigram_prob(c, w):
#     context = tokenize(c)[0]
#     word = tokenize(w)[0]
#     return cpd[context].prob(word)

# print(korean_bigram_prob("이", "영화"))
# print(korean_bigram_prob("영화", "이"))

# def korean_generate_sentence(seed=None, debug=False):
#     if seed is not None:
#         import random
#         random.seed(seed)
#     c = "SS"
#     sentence = []
#     while True:
#         if c not in cpd:
#             break
            
#         w = cpd[c].generate()

#         if w == "SE":
#             break

#         w2 = w.split("/")[0]
#         pos = w.split("/")[1]

#         if c == "SS":
#             sentence.append(w2.title())
#         elif c in ["`", "\"", "'", "("]:
#             sentence.append(w2)
#         elif w2 in ["'", ".", ",", ")", ":", ";", "?"]:
#             sentence.append(w2)
#         elif pos in ["Josa", "Punctuation", "Suffix"]:
#             sentence.append(w2)
#         elif w in ["임/Noun", "것/Noun", "는걸/Noun", "릴때/Noun",
#                    "되다/Verb", "이다/Verb", "하다/Verb", "이다/Adjective"]:
#             sentence.append(w2)
#         else:
#             sentence.append(" " + w2)
#         c = w

#         if debug:
#             print(w)

#     return "".join(sentence)

# print(korean_generate_sentence(0))
# print(korean_generate_sentence(1))
# print(korean_generate_sentence(2))
# print(korean_generate_sentence(3))
# print(korean_generate_sentence(5))










# # # 21.05.13
# # from pykospacing import spacing
# # import jpype
# import numpy as np
# from scipy import sparse

# # 0이 아닌 데이터 추출
# # dense = np.array([3,0,1],
# #                   [0,2,0])
# data = np.array([3,1,2]) #dense에서 추출한 data
# # 행 위치와 열 위치를 각각 array로 생성
# row_pos = np.array([0,0,1])
# col_pos = np.array([0,2,1])
# # sparse 패키지의 coo_matrix를 이용하여 coo형식으로 희소 행렬 생성
# sparse_coo = sparse.coo_matrix((data, (row_pos, col_pos)))

# print(data)
# print(row_pos)
# print(col_pos)
# print(type(sparse_coo))
# print(sparse_coo)


# from scipy import sparse

# dense2 = np.array([[0,0,1,0,0,5],
#              [1,4,0,3,2,5],
#              [0,6,0,3,0,0],
#              [2,0,0,0,0,0],
#              [0,0,0,7,0,8],
#              [1,0,0,0,0,0]])

# # 0 이 아닌 데이터 추출
# data2 = np.array([1, 5, 1, 4, 3, 2, 5, 6, 3, 2, 7, 8, 1])

# # 행 위치와 열 위치를 각각 array로 생성 
# row_pos = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5])
# col_pos = np.array([2, 5, 0, 1, 3, 4, 5, 1, 3, 0, 3, 5, 0])

# # COO 형식으로 변환 
# sparse_coo = sparse.coo_matrix((data2, (row_pos,col_pos)))

# # 행 위치 배열의 고유한 값들의 시작 위치 인덱스를 배열로 생성
# row_pos_ind = np.array([0, 2, 7, 9, 10, 12, 13])

# # CSR 형식으로 변환 
# sparse_csr = sparse.csr_matrix((data2, col_pos, row_pos_ind))

# print('COO 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
# print(sparse_coo.toarray())
# print('CSR 변환된 데이터가 제대로 되었는지 다시 Dense로 출력 확인')
# print(sparse_csr.toarray())



# import pandas as pd
# from math import log

# docs = [
#   '먹고 싶은 사과',
#   '먹고 싶은 바나나',
#   '길고 노란 바나나 바나나',
#   '저는 과일이 좋아요'
# ] 
# vocab = list(set(w for doc in docs for w in doc.split()))
# print(vocab)

# #TF,IDF 그리고 TF-IDF 값을 구하는 함수
# N = len(docs) # 총 문서의 수

# def tf(t, d): # 특정 문서 d에서의 특정 단어t의 반복 횟수
#     return d.count(t)

# def idf(t):
#     df = 0
#     for doc in docs:
#         df += t in doc
#     return log(N/(df + 1))

# def tfidf(t, d):
#     return tf(t,d)* idf(t)


# #TF를 구해보자. ->DTM을 데이터프레임에 저장하여 출력

# result = []
# for i in range(N): # 각 문서에 대해서 아래 명령을 수행
#     result.append([])
#     d = docs[i]
#     for j in range(len(vocab)):
#         t = vocab[j]        
#         result[-1].append(tf(t, d))

# tf_ = pd.DataFrame(result, columns = vocab)
# print(tf_)

# #정상적인 DTM이 출력되어짐. 이제 각 단어에 대한 IDF 값을 구해보자

# result = []
# for j in range(len(vocab)):
#     t = vocab[j]
#     result.append(idf(t))

# idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
# print(idf_)

# #TF-IDF 행렬을 출력

# result = []
# for i in range(N):
#     result.append([])
#     d = docs[i]
#     for j in range(len(vocab)):
#         t = vocab[j]

#         result[-1].append(tfidf(t,d))

# tfidf_ = pd.DataFrame(result, columns = vocab)
# print(tfidf_)



# from sklearn.feature_extraction.text import CountVectorizer
# corpus = [
#     'you know I want your love',
#     'I like you',
#     'what should I do ',    
# ]
# vector = CountVectorizer()
# print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
# print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.



# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [
#     'you know I want your love',
#     'I like you',
#     'what should I do ',    
# ]
# tfidfv = TfidfVectorizer().fit(corpus)
# print(tfidfv.transform(corpus).toarray())
# print(tfidfv.vocabulary_)







# 실습 : 20 Newsgroup 분류하기
# 188846개의 뉴스 20개의 뉴스 카테고리로 분류하기.

# 텍스트 정규화
# 피처 벡터화
# 머신러닝 학습/예측/평가
# Pipeline 적용
# GridSearch 최적화


# from sklearn.datasets import fetch_20newsgroups
# news_data = fetch_20newsgroups(subset= 'all', random_state=42)
# # print(news_data)
# print(news_data.keys())

# import pandas as pd
# print('target 클래스의 값과 분포도 \n',pd.Series(news_data.target).value_counts().sort_index())
# print('target 클래스의 이름들 \n',news_data.target_names)

# print(news_data.data[0])



from sklearn.datasets import fetch_20newsgroups

# subset='train'으로 학습용(Train) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
train_news= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
X_train = train_news.data
y_train = train_news.target
print(type(X_train))

# subset='test'으로 테스트(Test) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
test_news= fetch_20newsgroups(subset='test',remove=('headers', 'footers','quotes'),random_state=156)
X_test = test_news.data
y_test = test_news.target
print('학습 데이터 크기 {0} , 테스트 데이터 크기 {1}'.format(len(train_news.data) , len(test_news.data)))

from sklearn.feature_extraction.text import CountVectorizer
# Count Vectorization으로 feature extraction 변환 수행.
cnt_vect = CountVectorizer()
cnt_vect.fit(X_train, y_train)
X_train_cnt_vect = cnt_vect.transform(X_train)

# 학습 데이터로 fit()된 CounterVectorizer를 이용하여서 테스트 데이터를 feature extraction 변환 수행
X_test_cnt_vect = cnt_vect.transform(X_test)
print('학습 데이터 text의 CounterVectorizer Shape : ',X_train_cnt_vect.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LogisticRegression을 이용하여 학습/예측/평가 수행. 
lr_clf = LogisticRegression()
lr_clf.fit(X_train_cnt_vect , y_train)
pred = lr_clf.predict(X_test_cnt_vect)
print('CountVectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test,pred)))