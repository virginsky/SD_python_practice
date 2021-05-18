import pandas as pd
import glob, os

# pd.set_option('display.max_colwidth',1000)
# glob 해당 파일 안에 확장자가 일치하면 모든걸 가져오는 패키지
path = r'C:\Users\sundooedu\Documents\GitHub\SD_python_practice\machinelearning\TextAnalysis\OpinosisDataset1.0\topics'
all_files = glob.glob(os.path.join(path,'*.data'))
# print(all_files)
filename_list = []
opinion_text = []

# 개별 파일들의 파일명을 filename_list 리스트로 취합
# 개별 파일의 내용을 -> to_string -> opinion_text 리스트로 취합
for file_ in all_files:
    # 개별 파일을 일어서 dataframe으로 생성
    df = pd.read_table(file_,index_col=None,header=0,encoding='latin1')
    # 절대경로로 주어진 file명을 가공, 만약 리눅스에서 수행시는 아래 \\를 /로 변경, 맨 마지막 .data확장자도 제거
    filename_ = file_.split('\\')[-1]   # 파일+확장자
    filename = filename_.split('.')[0]  # 파일명과 확장자 분리
    # 파일명 리스트와 파일내용 리스트에 파일명과 파일 내용을 추가
    filename_list.append(filename)
    opinion_text.append(df.to_string())

# 파일명 리스트와 파일내용 리스트를 dataframe으로 생성
document_df = pd.DataFrame({'filename':filename_list, 'opinion_text':opinion_text})
# print(document_df.head())

# Lemmatization을 위한 함수 생성
from nltk.stem import WordNetLemmatizer
import nltk
import string

#.까지 다 쪼갤려고 실시.
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
lemmar = WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmar.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

#토큰화+lemmatation까지 한꺼번에 함.
tfidf_vect = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english' , \
                             ngram_range=(1,2), min_df=0.05, max_df=0.85 )

#opinion_text 컬럼값으로 feature vectorization 수행
feature_vect = tfidf_vect.fit_transform(document_df['opinion_text'])

from sklearn.cluster import KMeans

# 5개 집합으로 군집화 수행. 예제를 위해 동일한 클러스터링 결과 도출용 random_state=0 
km_cluster = KMeans(n_clusters=5, max_iter=10000, random_state=0)
km_cluster.fit(feature_vect)
cluster_label = km_cluster.labels_
cluster_centers = km_cluster.cluster_centers_

document_df['cluster_label'] = cluster_label
# print(document_df)
print(document_df[document_df['cluster_label']==0].sort_values(by='filename'))
print(document_df[document_df['cluster_label']==1].sort_values(by='filename'))
print(document_df[document_df['cluster_label']==2].sort_values(by='filename'))
print(document_df[document_df['cluster_label']==3].sort_values(by='filename'))