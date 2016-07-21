import pandas as pd
import numpy as np

import bs4
import re
import nltk

from gensim.models import word2vec


#Word2Vec expects single sentences, each one as a list of words. In other words, the input format is a list of lists.
#Word2Vec的输入希望是list的形式，list中每一个是一句sentence，每一个sentence又是由一个list（组成句子的词）构成
#example:   [ ["this", "is", "a", "nice", "car"], ["i", "like", "it"] ]




#训练word2vec时候停用词要保留，别的时候都要去除停用词
# Function to convert a  text to a list of words
# The input is a single string (a text), and  the output is a list (a preprocessed  text) ["this", "is", "a", "nice", "car"]
def text_to_wordlist(raw_sentence):
    clean_html               =  bs4.BeautifulSoup(raw_sentence, "lxml").get_text()       #去掉html格式
    letters_only             =  re.sub(  "[^a-zA-Z]",   " ",  clean_html  )              #将非字母的东西用空格替代
    lower_words_list    =  letters_only.lower().split()                     #Convert to lower case, split into individual words
    return lower_words_list                             



#we'll use NLTK's punkt tokenizer for sentence splitting.
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Define a function to split a text into parsed sentences
def parsed_text(raw_text):
                #将每一个人的评论的整个text通过split成多个sentence按照list形式放入sentences中（根据句号进行切割）, 例如："this is a nice car. I like it." 切割成  ["this is a nice car.",  "I like it."]
                sentences = tokenizer.tokenize(raw_text.strip())
                sentences_list = []
                for sentence in sentences:
                                if len(text)>0:
                                                sentences_list.append(text_to_wordlist(sentence))
                return sentences_list





train                        = pd.read_csv("./dataset/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test                          = pd.read_csv("./dataset/testData.tsv", header=0, delimiter="\t",  quoting=3 )
unlabeled_train   = pd.read_csv("./dataset/unlabeledTrainData.tsv", header=0, delimiter="\t",  quoting=3 )




sentences_list_of_lists=[]
for text in train["review"]:
                sentences_list_of_lists += parsed_text(text)

for text in unlabeled_train["review"]:
                sentences_list_of_lists += parsed_text(text)



# Set values for various parameters
num_features         =  300         # Word vector dimensionality                      
min_word_count   =  40           # Minimum word count                        
num_workers         =  4             # Number of threads to run in parallel
context                     =  10          # Context window size                                                                                    
downsampling       =  1e-3      # Downsample setting for frequent words

model               =  word2vec.Word2Vec(sentences_list_of_lists, workers=num_workers,  size=num_features, min_count = min_word_count,  window = context, sample = downsampling)
model_name  =  "300features_40minwords_10context"
model.save(model_name)

print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.most_similar("man"))