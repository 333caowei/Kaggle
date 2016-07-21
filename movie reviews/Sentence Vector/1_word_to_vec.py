import pandas as pd
import numpy as np

import bs4
import re
import nltk

from gensim.models import doc2vec


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








train                        = pd.read_csv("./dataset/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test                          = pd.read_csv("./dataset/testData.tsv", header=0, delimiter="\t",  quoting=3 )
unlabeled_train   = pd.read_csv("./dataset/unlabeledTrainData.tsv", header=0, delimiter="\t",  quoting=3 )



count=0
sentences_list_of_lists=[]
for text in train["review"]:
                sentences_list_of_lists.append(    doc2vec.LabeledSentence( words=text_to_wordlist(text), tags=['SENT_' + str(count)] )    )
                count+=1
for text in unlabeled_train["review"]:
                sentences_list_of_lists.append(    doc2vec.LabeledSentence( words=text_to_wordlist(text), tags=['SENT_' + str(count)] )    )
                count+=1








# Set values for various parameters
num_features         =  600         # Word vector dimensionality                      
min_word_count   =  40           # Minimum word count                        
num_workers         =  4             # Number of threads to run in parallel
context                     =  10          # Context window size                                                                                    
downsampling       =  1e-3      # Downsample setting for frequent words

model               =  doc2vec.Doc2Vec( workers=num_workers,  size=num_features, min_count = min_word_count,  window = context, sample = downsampling)
model.build_vocab(sentences_list_of_lists)
model.train(sentences_list_of_lists)
model_name  =  "600features_40minwords_10context"
model.save(model_name)

print(model.doesnt_match("man woman child kitchen".split()))
print(model.doesnt_match("france england germany berlin".split()))
print(model.most_similar("man"))







# model = doc2vec.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1, workers=4)
# model.build_vocab(sentences_list_of_lists)

# for epoch in range(1):
#                 model.train(sentences_list_of_lists)
#                 model.alpha -= 0.002  # decrease the learning rate`
#                 model.min_alpha = model.alpha  # fix the learning rate, no decay

# model.save("my_model.doc2vec")
# model_loaded = doc2vec.Doc2Vec.load('my_model.doc2vec')

# print (model.docvecs.most_similar(["SENT_0"]))
# print (model_loaded.docvecs.most_similar(["SENT_1"]))