
# coding: utf-8

# # Logistic Regression

# Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).
# Logistic regression is used to find the probability of event=Success and event=Failure. We should use logistic regression when the dependent variable is binary (0/ 1, True/ False, Yes/ No) in nature. Here the value of Y ranges from 0 to 1 and it can represented by following equation.
# 
# odds= p/ (1-p) = probability of event occurrence / probability of no event occurrence
# 
# ln(odds) = ln(p/(1-p))
# 
# logit(p) = ln(p/(1-p)) = b0+b1X1+b2X2+b3X3....+bkXk
# 
# Since we are working here with a binomial distribution (dependent variable), we need to choose a link function which is best suited for this distribution. And, it is logit function. In the equation above, the parameters are chosen to maximize the likelihood of observing the sample values rather than minimizing the sum of squared errors (like in ordinary regression).
# ![image.png](attachment:image.png)

# In[435]:

# Required Python Machine learning Packages
#Statistical analysis
import pandas as pd
import numpy as np
# To split the dataset into train and test datasets
from sklearn.model_selection import train_test_split
# visualization
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[436]:

df = pd.read_csv('C:/Users/Karan Kanwal/Downloads/binary.csv')
#convert the csv to dataframe df
#df = pd.DataFrame()


# In[455]:

df
#Reading dataframe which consists of admit,gre,gpa and rank of a student
#The aim is to predict the admit based on the gre gpa and rank


# In[456]:

#A general insight into the number of entries,null,not null and data types of our imported dataset
df.info()


# In[457]:

#Checking for null values in df dataframe
df.isnull().sum()


# In[458]:

#Function to give us a correlation matrix which gives relation between each of the features
corr = df.corr()


# In[459]:

corr


# In[460]:

#Heatmap stating the correlation matrix using seaborn
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
#plt.show()


# In[461]:

#Now we create a cross table between various features and visualize using histogram
print(pd.crosstab(df['admit'],df['gre'], rownames=['admit']))
 
df.hist()
plt.show()


# In[444]:

print(pd.crosstab(df['admit'], df['gpa'], rownames=['admit']))
 


# In[462]:

#Now we carry out some analysis and visually represent the admits obtained or rejected depending on factors such as gpa and gre
positive = df[df['admit'].isin([1])]  
negative = df[df['admit'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))  
ax.scatter(positive['gpa'], positive['gre'],s=50, c='b', marker='o', label='Accepted')  
ax.scatter(negative['gpa'], negative['gre'], s=50, c='r', marker='x', label='Rejected')  
ax.legend()  
ax.set_xlabel('GPA')  
ax.set_ylabel('GRE')  


# In[463]:

#We set our X and Y values 
#Here our X is the predictor variable 
X=df.iloc[:,1:4]
X=np.matrix(X)
#X.shape

X.shape


# In[464]:

#Y is our response variable which we will try and predict using logistic regression
Y=df.iloc[:,0]
Y=Y.astype(float)
#Below we reshape Y using matrix and transpose to ensure X and Y are of the same shape
Y=np.matrix(Y)
Y=np.transpose(Y)
Y.shape


# In[471]:

#We calculate the result of the logit model and present the summary to gain the stand error,z value. 
#logit function gives the log-odds, or the logarithm of the odds p/(1 − p).
import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary())


# Regularization is a very important technique in machine learning to prevent overfitting. Mathematically speaking, it adds a regularization term in order to prevent the coefficients to fit so perfectly to overfit. The difference between the L1 and L2 is just that L2 is the sum of the square of the weights, while L1 is just the sum of the weights.As follows:
# 
# ![image.png](attachment:image.png)

# ## LOGISTIC REGRESSION WITH L2 Penalty

# L2 regularization on least squares:
# 
# ![image.png](attachment:image.png)

# In[473]:

#We now split our x and y parameter into training and test set,and consider a test size to apply logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(penalty="l2")
logreg.fit(X_train, y_train)
coefL2=logreg.fit(X,Y).coef_.ravel()
coefL2  # coef_l2_LR = clf_l2_LR.coef_.ravel()


# In[475]:

#The default normal is the l2 normal which has been used above,and we calculate the score
logreg.score(X_test,y_test)


# L1 Regression: 
# 
# ![image.png](attachment:image.png)

# In[476]:

#We do the same as above but this time considering the l1 penalty
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression(penalty="l1")
logreg.fit(X_train, y_train)


# In[469]:

coefL1=logreg.fit(X,Y).coef_.ravel()
coefL1


# In[477]:

#We find the score of logreg function here as well using l1 penalty
logreg.score(X_test,y_test)


# As we can see on computing l1 and l2 regression,we can analyze that the logreg score is better when lasso regression is carried out

# In[478]:

#We now for each regularization norm i.e l1 and l2 penalty try and perform logistic regression for a varied set of values 
# We then calculate the l1 penalty score and the sparsity for each of these enumerations.
#We then plot these to to visually represent the co-efficients of logistic regression obtained using l1 and l2
for i, C in enumerate((100, 1, 0.01)):
    clf_l1_LR = LogisticRegression(C=C,penalty='l1', tol=0.01)
    clf_l2_LR = LogisticRegression(C=C,penalty='l2', tol=0.01)
    clf_l1_LR.fit(X, Y)
    clf_l2_LR.fit(X, Y)
    coef_l1_LR = clf_l1_LR.coef_.ravel()
    coef_l2_LR = clf_l2_LR.coef_.ravel()
    sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
    sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100 
    print("C=%.2f" % C)
    print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
    print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, Y))
    print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
    print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))
    
    l1_plot = plt.subplot(3, 2, 2 * i + 1)
    l2_plot = plt.subplot(3, 2, 2 * (i + 1))
    if i == 0:
        l1_plot.set_title("L1 penalty")
        l2_plot.set_title("L2 penalty")
    l1_plot.plot(coef_l1_LR)#imshow(np.abs(coef_l1_LR), interpolation='nearest',cmap='binary', vmax=1, vmin=0)
    l2_plot.plot(coef_l2_LR)#imshow(np.abs(coef_l2_LR), interpolation='nearest',
                   #cmap='binary', vmax=1, vmin=0)
    #plt.text(-5, 3, "C = %.2f" % C)
    

    


# Differences between L1 and L2 Regularization:
# ![image.png](attachment:image.png)
# 
#  **Advantages of regularized linear models:**
#  
#  - Useful for high-dimensional problems (p > n)
#  - Better performance
#  - L1 regularization performs automatic feature selection
#  
# 
# **Disadvantages of regularized linear models:**
#  
#  - Tuning is required
#  - Feature scaling is recommended.Under many circumstances beyond the grouping issue of features, the feature selection is not consistent.
#  - Less interpretable (due to feature scaling)
#  - L1 you will miss out correlation between features by its favor of one of those variables remaining a non-zero value instead of both.

# **References**:
# 
# 
# - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
# - http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# - https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/ 
# - https://github.com/justmarkham/DAT5/blob/master/code/18_regularization.py
# - https://github.com/gabrielcs/movie-ratings-prediction/blob/master/5-Machine_Learning.ipynb

# In[ ]:



