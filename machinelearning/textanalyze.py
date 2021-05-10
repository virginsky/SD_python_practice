from nltk.tokenize import TreebankWordTokenizer
from nltk import word_tokenize, sent_tokenize 
import nltk
nltk.download('punkt')
nltk.download('stopwords') # 유의미한 단어 토큰만을 선별하기 위해 불용어 처리를 배우는 과정
nltk.download('wordnet')
# 품사 태깅
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')

'''
tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))


text_sample = 'The Matrix is everywhere its all around us, here even in this room. \
               You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes.'

# sent_tokenize : . 위주로 자름(문장 구분)
sentences = sent_tokenize(text=text_sample)
print(type(sentences), len(sentences))
print(sentences)

# word_tokenize : 단어로 자름
sentence = "The Matrix is everywhere its all around us, here even in this room."
words = word_tokenize(sentence)
print(type(words), len(words))
print(words)


def tokenize_text(text):
    # 데이터를 문장별로 tokenize
    sentences = sent_tokenize(text)
    # 문장 -> 단어
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

word_tokens = tokenize_text(text_sample)
print(type(word_tokens), len(word_tokens))
print(word_tokens)



print('영어 stop words 갯수:',len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:20])


stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
for sentence in word_tokens:
    filtered_words=[]
    for word in sentence:
        # 소문자로 변환
        word = word.lower()
        # tokenize된 개별 word가 stop words들의 단어에 포함되지 않으면 word_tokens에 추가
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)

print(all_tokens)



# Stemming과 Lemmatization
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
print(stemmer.stem('working'),stemmer.stem('works'),stemmer.stem('worked'))
print(stemmer.stem('amusing'),stemmer.stem('amuses'),stemmer.stem('amused'))
print(stemmer.stem('happier'),stemmer.stem('happiest'))
print(stemmer.stem('fancier'),stemmer.stem('fanciest'))


from nltk.stem import WordNetLemmatizer


lemma = WordNetLemmatizer()
print(lemma.lemmatize('amusing','v'),lemma.lemmatize('amuses','v'),lemma.lemmatize('amused','v'))
print(lemma.lemmatize('happier','a'),lemma.lemmatize('happiest','a'))
print(lemma.lemmatize('fancier','a'),lemma.lemmatize('fanciest','a'))



from nltk.tokenize import word_tokenize
text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
print(word_tokenize(text))

# 품사 태깅
x = word_tokenize(text)
print(pos_tag(x))
#  PRP는 인칭 대명사, VBP는 동사, RB는 부사, VBG는 현재부사, IN은 전치사, NNP는 고유 명사, NNS는 복수형 명사, CC는 접속사, DT는 관사



# ## 에러
# import platform
# print(platform.architecture())
# from konlpy.tag import Okt
# okt = Okt()
# print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))



# 어간 추출(Stemming)
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
s = PorterStemmer()
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words=word_tokenize(text)
print(words)
print([s.stem(w) for w in words]) #사전에 없는 단어 삭제, but 단순규칙에 의하기떄문에 정확치않음.

words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([s.stem(w) for w in words])

from nltk.stem import LancasterStemmer
l=LancasterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([l.stem(w) for w in words])



# 불용어(Stopwords) 제거
print('영어 stop words 갯수:',len(nltk.corpus.stopwords.words('english')))
print(nltk.corpus.stopwords.words('english')[:20])



# 위 예제의 3개의 문장별로 얻은 word_tokens list 에 대해 stop word 제거 Loop
tokenizer=TreebankWordTokenizer()
stopwords = nltk.corpus.stopwords.words('english')
all_tokens = []
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
# print(tokenizer.tokenize(text))
text_sample = 'The Matrix is everywhere its all around us, here even in this room. \
               You can see it out your window or on your television. \
               You feel it when you go to work, or go to church or pay your taxes.'
def tokenize_text(text):
    # 데이터를 문장별로 tokenize
    sentences = sent_tokenize(text)
    # 문장 -> 단어
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    return word_tokens

word_tokens = tokenize_text(text_sample)

for sentence in word_tokens:
    filtered_words=[]
    # 개별 문장별로 tokenize된 sentence list에 대해 stop word 제거 Loop
    for word in sentence:
        #소문자로 모두 변환합니다. 
        word = word.lower()
        # tokenize 된 개별 word가 stop words 들의 단어에 포함되지 않으면 word_tokens에 추가
        if word not in stopwords:
            filtered_words.append(word)
    all_tokens.append(filtered_words)
    
print(all_tokens)





from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)



import re
r1 = re.compile('a.c')   # a와 c 사이에 어떤 1개의 문자라도 올 수 있다는 뜻
print(r1.search('kkk'))
print(r1.search('abc'))

#  * : 바로 앞의 문자가 0개 이상인 경우(혹은 여러개)
r2 = re.compile('ab*c')
print(r2.search('a'))   # none
print(r2.search('abbbc'))

r3 = re.compile('<*>')
print(r3.search('<b>dfdfa</b>'))

text = """이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""
print(re.findall("\d+", text)) #\d : 숫자를 한글자씩 찾는다 // + : 연결하시오
# 숫자를 찾고 뒤에 있으면 계속 연결하시오


text="""Regular expression : A regular expression, regex or regexp[1] 
(sometimes called a rational expression)[2][3] is, in theoretical computer 
science and formal language theory, a sequence of characters that define a search pattern."""
print(re.sub('[^a-zA-Z]',' ',text)) #정규 표현식 패턴과 일치하는 문자열을 찾고, 일치하지 않으면 다른 문자열로 대체

text = """100 John    PROF
101 James    STUD
102 Mac    STUD"""
print(re.split('\s+', text))
print(re.findall('\d+', text))
print(re.findall('[a-zA-Z]+', text))
print(re.findall('[A-Z][a-z]+', text)) # 첫글자만 대문자인 알파벳 연결해서 가져오기
letters_only = re.sub('[^a-zA-Z]', ' ', text)
print(letters_only)



import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer("[\w]+")    # 문자 또는 숫자가 1개 이상인 경우 인식하는 코드
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

#  \s+ 는 공백을 의미하므로 공백을 기준으로 토큰화된 결과값을 출력한다. 
# gaps=True를 설정하지 않으면 공백만이 출력된다.(공백으로 시작하는 것들을 찾음) 
# 그 전 결과와 비교한다면 어퍼스트로피나 온점을 제외하지 않고, 토큰화가 수행
tokenizer = RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))
'''


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
text = sent_tokenize(text)
print(text)
# 정제 및 단어 토큰화
vocab = {}
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i)
    result = []

    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    sentences.append(result)
print(sentences)
print(vocab)

#빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)

#높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i=i+1
        word_to_index[word] = i
print(word_to_index)

vocab_size = 5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
print(word_to_index)

word_to_index['OOV'] = len(word_to_index) + 1

#word_to_index를 사용하여 sentences의 모든 단어들을 맵핑되는 정수로 인코딩
encoded = []
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word_to_index[w])
        except KeyError:
            temp.append(word_to_index['OOV'])
    encoded.append(temp)
print(encoded)


from collections import Counter
print(sentences)

# 단어 집합(Vocabulary)을 만들기 위해서 
#sentences에서 문장의 경계인 [,]를 제거하고 단어들을 하나의 리스트로 만듬
words = sum(sentences, [])
# 위 작업은 words = np.hstack(sentences)로도 수행 가능.
print(words)

vocab = Counter(words) # 파이썬의 Counter 모듈을 이용하면 단어의 모든 빈도를 쉽게 계산할 수 있습니다.
print(vocab)

from nltk import FreqDist
import numpy as np
# np.hstack으로 문장 구분을 제거하여 입력으로 사용 . ex) ['barber', 'person', 'barber', 'good' ... 중략 ...
vocab = FreqDist(np.hstack(sentences))
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
print(vocab)