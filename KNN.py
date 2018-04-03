
# coding: utf-8

# ## Assignment 1
# 
# ## KNN and Cross Validation

# The KNN algorithm is a robust and Powerful classifier ,often used as a benchmark for more complex classifiers such as Aritifical Nerual Networks and Support Vector Machine.KNN falls in the supervised learning family of algorithms. Informally, this means that we are given a labelled dataset consiting of training observations (x,y)(x,y) and would like to capture the relationship between xx and yy.  
# 
# In the characterization setting, the K-closest neighbor calculation basically comes down to shaping a greater part vote between the K most comparable examples to a given "concealed" perception. Similarity between variables or data points is defined according to a distance metric between two data points. The common distance metric used is the Euclidean distance,but other measures can be more suitable for a given set of data and include the Manhattan, Chebyshev and Hamming distance.

# In[118]:



get_ipython().magic('matplotlib inline')

#We import the libraries,necessary to carry out statistical analysis and plot various graphs,we then set up the envoirnment 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.preprocessing, sklearn.decomposition
from sklearn.feature_extraction.text import CountVectorizer


if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):
    import prep 

#Display 10 rows
pd.options.display.max_rows = 10
sns.set(style='ticks', context='talk')


# We first import our dataset,using pd.read_csv.Our data set gives us information about room occupancy 

# In[119]:

df = pd.read_csv('K:/Spring 2018/Statistics and Machine Learning/Datasets/occupancy.csv')


# We will analyze the dataset,and implement KNN,to find out occupancy of a particular room is Yes or a No,that is,1 or 0

# In[121]:

plt.figure(figsize=(15,10))
andrews_curves(df.drop("date",axis=1),"Occupancy")
plt.title('Andrews Curves Plot', fontsize=20, fontweight='bold')
plt.legend(loc=1, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
plt.show()


# Now,we will consider our predictor and response data.We will predict the value of occupancy using factors such as Temperate,Humidity and Co2 Levels and store the predictor and response variables in x and y respectively

# In[137]:


x=df.ix[:,(1,2,5)].values  #Store Temperature,Humidity and CO2 in X variable
print(x)
y=df.ix[:,6].values        #Store Occupancy in Y variable


# In[123]:

from sklearn import preprocessing
x=preprocessing.scale(x)


# Below is an example of a typical plot of knn classification if the neighbors considered =5.The value of selected neigbours depends in each case for each data set and is considered an important factor is predicting the accuracy of KNN correctly

# In[124]:

import mglearn as mglearn
mglearn.plots.plot_knn_classification(n_neighbors=5)
plt.show()


# Below is the code where we split our dataset into training and testing.We will split both are x variable and Y variable into train set and test set respectively using the train_test_split,which is a inbuilt function in sklearn.We consider the test size as 0.4 here,which can be adjusted to increase efficiency

# In[125]:

from sklearn.model_selection import train_test_split #import train_test_split to split data correctly into testing and training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=0)


# Now,we will go ahead with performing our KNN,where we first import the KneighboursClassifier and metrics.We consider the neighbors -5.We then fit our training set and testing set into a variable clf to establish relationships between the training sets.In our next line of code,we predict the value of x_test and using the metrics classification report we get the precision,recall,f1-score and support values of the classes which we wish to predict.
# 

# In[126]:

from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
# Create a k-NN classifier with 3 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)#Fit x_train and y_train into clf
print(knn)
pred=knn.predict(x_test) 
print(pred)
#We below,calculate the metric classification report,of the class to be predicted(y) using the found out predicted value(x_test)
print(metrics.classification_report(y_test,pred))



# We also find out the comparison between the response variable testing data(y_test) and the predictor variable testing data,to check the accraucy obtained through the KNN model.
# The higher the accuracy,the more is the effictiveness of the KNN model used.We do the above using accuracy_score

# In[127]:

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred)*100)


# The accuracy obtained above is 95.74 for the respective dataset considered with neighbours considered 5

# ### Cross-validation is a mechanism to evaluate predictive models by splitting/ partitioning the original sample into a training set to train the model, and a test set to evaluate it. In k-fold cross-validation, the original sample is randomly split into k equal size sub-samples. Out of the k subsamples, a single sample is attained as the validation data for testing the model, and the remaining k-1 subsamples are used as training data.We then repeat this process k times(the folds) with each of the k subsamples being used exactly once The k results from the folds can then be averaged  to produce a single estimation

# We first find out the cross_val_score for our training data and print the,the default value of cv is 3,which stands for 3 fold cross validation,however we will be doing a 10-fold creoss validation

# In[128]:

from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import cross_val_score
#kf = KFold(25, shuffle=False)
knn = KNeighborsClassifier(n_neighbors=25)
scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')#Store the cross_val_score in scores vairable
print(scores)


# In[129]:

#We print the mean of the cross_val_score
print(scores.mean())


# We now try to predict the accuracy of K,with respect to K.That is,which K gives us the highest value of accuracy.We thus find out the cross_Val_score for 10-fold cross validation and append all the mean of the scores and store it in an array represented as k_scores

# In[130]:

k_range = range(1, 40)   #Range of neighbors to be considered,the find out most optimal
k_scores = []            #Empty array,which will later on store all the accuracy scores
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy') #10-fold cross validation
    k_scores.append(scores.mean())
print(k_scores)


# Using matplot lib we know plot the values of k obtained executing the above lines of code and compare it with the accuracy obtained for each value of k.We construct the graph using Matplotlib library

# In[131]:

import matplotlib.pyplot as plt
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# We now try to find out the Missclassification error,to predict which value of K,gives us the best result.The lowest value of Mean Squared error obtained,is the optimal number of neighbours

# In[132]:

myList = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, myList))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation on k range of neighbours
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy') #cross_Val_score computation stored in scores
    cv_scores.append(scores.mean())


# We below calculate the Missclassification errors which is difference between cross validation score and 1.
# We then determine the optimal_k by finding out the index of the minimum MSE obtained.We then plot and print this relation between the number of neighbours and MSE

# In[138]:


MSE = [1- x for x in cv_scores]


# determining best k
optimal_k = (neighbors[MSE.index((min(MSE)))])
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)              #plot x compared to y
plt.xlabel('Number of Neighbors K')   #x-axis
plt.ylabel('Misclassification Error') #y-axis
plt.show()


# Optimal number of neighbours obtained is 27,as the MSE obtained is least for that K.We thus conclude KNN and cross validation.Thank you!

# In[ ]:



