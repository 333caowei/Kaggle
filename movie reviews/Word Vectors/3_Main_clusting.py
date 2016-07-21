from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import bs4
import re
import nltk

import time



#统计每篇评论中，所有出现在word2vec词库中的词，在聚类后所属不同类别下词的个数
#The function above will give us a numpy array for each review, each with a number of features equal to the number of clusters. 
def create_bag_of_centroids( wordlist, word_centroid_map ):
	num_centroids      = max(word_centroid_map.values())+1
	bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index]+=1
	return bag_of_centroids




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





start = time.time()
model = Word2Vec.load("300features_40minwords_10context")
word_vectors = model.syn0
#Trial and error suggested that small clusters, with an average of only 5 words or so per cluster, 
#gave better results than large clusters with many words.
num_clusters = len(model.syn0) // 5
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", str(elapsed), "seconds.")


# Create a Word / Index dictionary, mapping each vocabulary word to a cluster number                                                                                            
word_centroid_map = dict(zip(model.index2word, idx))






train   	= pd.read_csv("./dataset/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train_centroids = np.zeros( (len(train["review"]), num_clusters), dtype="float32" )
clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append( text_to_wordlist( review))
# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
	train_centroids[counter] = create_bag_of_centroids( review,  word_centroid_map )
	counter += 1




# Repeat for test reviews 
test   	= pd.read_csv("./dataset/testData.tsv", header=0, delimiter="\t",  quoting=3 )
test_centroids = np.zeros(( len(test["review"]), num_clusters),  dtype="float32" )
clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append( text_to_wordlist( review))
# Transform the testing set reviews into bags of centroids
counter = 0
for review in clean_test_reviews:
	test_centroids[counter] = create_bag_of_centroids( review,  word_centroid_map )
	counter += 1



#Training the random forest...
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)
#Fit the forest to the training set, using the bag of words as  features and the sentiment labels as the response variable
forest = forest.fit( train_centroids, train["sentiment"] )
#predict
result = forest.predict(test_centroids)
# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# Use pandas to write the comma-separated output file
output.to_csv( "BagOfCentroids.csv", index=False, quoting=3 )