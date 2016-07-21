import numpy as np
import pandas as pd
import bs4
import re
import nltk

from gensim.models import Word2Vec

import sklearn
from sklearn.ensemble import RandomForestClassifier






#Since each word is a vector in 300-dimensional space, we can use vector operations to combine the words in each review. 
#One method we tried was to simply average the word vectors in a given review (for this purpose, we removed stop words, 
#which would just add noise).



# #Load the model that we created in 1_word_to_vector.py
model = Word2Vec.load("300features_40minwords_10context")
# print(model.syn0.shape)
# #Individual word vectors can be accessed in the following way,which returns a 1x300 numpy array.
# print(model["flower"])
# # index2word is a list that contains the names of the words in the model's vocabulary.
# print(len(model.index2word))






# Function to average all of the word vectors in a given paragraph
#将每一个text中所有词向量加权平均成一维向量，也就是说每一个text由一个一维向量表示
def text_to_vec(wordlist, model, num_features):
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,), dtype="float32")
	nwords = 0.
	words_set = set(model.index2word)
	# Loop over each word in the review and, if it is in the model's vocaublary, add its feature vector to the total
	for word in wordlist:
		if word in words_set: 
			nwords = nwords + 1.
			featureVec = np.add(featureVec, model[word])
	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	return featureVec



#所有text组成一个二维向量，每行表示一个text，每列表示特征维度
def textlist_to_vecs(clean_text_list, model, num_features):
	# Given a set of clean_text_list (each one a list of words), calculate  the average feature vector for each one and return a 2D numpy array  
	# Preallocate a 2D numpy array, for speed
	reviewFeatureVecs = np.zeros((len(clean_text_list),num_features), dtype="float32")
	# Initialize a counter
	counter = 0.
	# Loop through the clean_text_list
	for clean_text in clean_text_list:
		# Call the function (defined above) that makes average feature vectors
		reviewFeatureVecs[counter] = text_to_vec(clean_text, model, num_features)
		# Increment the counter
		counter = counter + 1.
	return reviewFeatureVecs



#训练word2vec时候停用词要保留，别的时候都要去除停用词
# Function to convert a raw text to a string of words
# The input is a single string (a raw movie text), and  the output is a single string (a preprocessed movie text)
def text_to_wordlist(raw_text):
	clean_html         	    =  bs4.BeautifulSoup(raw_text, "lxml").get_text()		#去掉html格式
	letters_only  	    =  re.sub(  "[^a-zA-Z]",   " ",  clean_html  )    			#将非字母的东西用空格替代
	lower_words_list    =  letters_only.lower().split()     				#Convert to lower case, split into individual words
	stopwords  	    = set(nltk.corpus.stopwords.words("english"))			#In Python, searching a set is much faster than searching a list, so convert the stop words to a set
	clean_stopwords   = [w for w in lower_words_list if not w in stopwords]		#Remove stop words
	return ( clean_stopwords)   						






num_features            =  300         # Word vector dimensionality                      
train   	= pd.read_csv("./dataset/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append( text_to_wordlist( review))
trainDataVecs = textlist_to_vecs(clean_train_reviews, model, num_features)

test   	= pd.read_csv("./dataset/testData.tsv", header=0, delimiter="\t",  quoting=3 )
clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append( text_to_wordlist( review))
testDataVecs = textlist_to_vecs( clean_test_reviews, model, num_features )




#Training the random forest...
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)
#Fit the forest to the training set, using the bag of words as  features and the sentiment labels as the response variable
forest = forest.fit( trainDataVecs, train["sentiment"] )
#predict
result = forest.predict(testDataVecs)
# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# Use pandas to write the comma-separated output file
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )