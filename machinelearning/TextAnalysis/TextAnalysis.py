# # 21.05.18

# # 잠재 디리클레 할당(Latent Dirichlet Allocation,LDA)
# # 토픽 모델링 - 20 뉴스그룹

# from sklearn.datasets import fetch_20newsgroups
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# # 모터사이클,야구,그래픽스,윈도우즈,중동,기독교,의학,우주 주제를 추출
# cats = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'comp.windows.x',
#         'talk.politics.mideast', 'soc.religion.christian', 'sci.electronics', 'sci.med'  ]
    
# # 위에서 cats 변수로 기재된 category만 추출. featch_20newsgroups( )의 categories에 cats 입력
# news_df= fetch_20newsgroups(subset='all',remove=('headers', 'footers', 'quotes'), 
#                             categories=cats, random_state=0)
# # print(news_df)

# #LDA 는 Count기반의 Vectorizer만 적용합니다.  
# count_vect = CountVectorizer(max_df=0.95, max_features=1000, min_df=2, stop_words='english', ngram_range=(1,2))
# feat_vect = count_vect.fit_transform(news_df.data)
# print('CountVectorizer Shape:', feat_vect.shape)

# # LDA 객체 생성 후 Count 피처 벡터화 객체로 LDA 수행
# lda = LatentDirichletAllocation(n_components=8, random_state=0)
# lda.fit(feat_vect)
# # components : 주제별로 개별 단어들의 연관도 정규화 숫자가 들어 있음
# # component_.shape : 주제 개수x 피쳐 단어 개수,

# # 토픽별 단어 확인
# def display_topics(model,featrue_names, no_top_words):
#     for topic_index, topic in enumerate(model.components_):
#         print('Topic #',topic_index)

#         # components_ array에서 가장 값이 큰 순으로 정렬했을 떄 그 값의 array index를 반환
#         topic_word_indexes = topic.argsort()[::-1]
#         top_indexes = topic_word_indexes[:no_top_words]

#         # top_indexes 대상인 index별로 feature_names에 해당하는 word feature 추출 후 join으로 concat
#         feature_concat = ' '.join([feature_names[i] for i in top_indexes])   
#         print(feature_concat)

# # CountVectorizer 객체 내의 전체 word들의 명칭을 get_features_names( )를 통해 추출
# feature_names = count_vect.get_feature_names()

# # Topic별 가장 연관도가 높은 word를 15개만 추출
# display_topics(lda, feature_names, 15)





# 멜론 1~100위 실시간차트 크롤링
import reqruests
from bs4 import BeautifulSoup

