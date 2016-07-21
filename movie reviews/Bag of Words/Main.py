import pandas as pd
import numpy as np

import bs4
import re
import nltk

import sklearn
from sklearn.ensemble import RandomForestClassifier






# Function to convert a raw text to a string of words
# The input is a single string (a raw movie text), and  the output is a single string (a preprocessed movie text)
def text_to_words(raw_text):
	clean_html         	    =  bs4.BeautifulSoup(raw_text, "lxml").get_text()		#去掉html格式
	letters_only  	    =  re.sub(  "[^a-zA-Z]",   " ",  clean_html  )    			#将非字母的东西用空格替代
	lower_words_list    =  letters_only.lower().split()     				#Convert to lower case, split into individual words
	stopwords  	    = set(nltk.corpus.stopwords.words("english"))			#In Python, searching a set is much faster than searching a list, so convert the stop words to a set
	clean_stopwords   = [w for w in lower_words_list if not w in stopwords]		#Remove stop words
	return ( " ".join( clean_stopwords ))   						#Join the words back into one string separated by space, 










#train step
train = pd.read_csv("./dataset/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
print(train.columns.values)
# print("")
# print(train.ix[0:3, 1:2])
# print("")
# print(train['id'][0:5])


#将语料库文本清洗后放入clean_train_review
clean_review_list = []
num_row = len(train)
for i in range(0, num_row):
	clean_review_list.append(text_to_words(train['review'][i]))


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
#这里的stop_words可以是字符串english，也可以是一个自己的list，功能和上面的去除停用词功能类似,
#max_features就是抽取的特征数, To limit the size of the feature vectors, we should choose some maximum vocabulary size. Below, we use the 5000 most frequent words
vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer="word", tokenizer=None, preprocessor = None, stop_words = None, max_features = 5000)			
## fit_transform() does two functions: First, it fits the model, second, it transforms our training data into feature vectors
review_data_features = vectorizer.fit_transform(clean_review_list)##########
#convert the result to an array
review_data_features = review_data_features.toarray()


#Training the random forest...
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)
#Fit the forest to the training set, using the bag of words as  features and the sentiment labels as the response variable
forest = forest.fit( review_data_features, train["sentiment"] )














#test step
test = pd.read_csv("./dataset/testData.tsv", header=0, delimiter="\t",  quoting=3 )
clean_review_list = []
num_row = len(test)
for i in range(0, num_row):
	clean_review_list.append(text_to_words(test['review'][i]))
review_data_features = vectorizer.transform(clean_review_list)##########
review_data_features = review_data_features.toarray()
#predict
result = forest.predict(review_data_features)
# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )