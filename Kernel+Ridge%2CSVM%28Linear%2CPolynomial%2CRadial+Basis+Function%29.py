
# coding: utf-8

# In[2]:

import os                                  #importing os
import numpy as np                         #importing package numpy
import pandas as pd                        #importing package pandas
import seaborn as sns                      #importing package seaborn 
import matplotlib.pyplot as plt            #importing package matplotlib 
import warnings
from subprocess import check_output
from IPython.core.display import display, HTML
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 4)
warnings.simplefilter('ignore')
#init_notebook_mode()
display(HTML("<style>.container { width:100% !important; }</style>"))

get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import scale
from sklearn import cross_validation
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# In[84]:

df=pd.read_csv("K:/Spring 2018/Statistics and Machine Learning/Datasets/mpg.csv")


# In[85]:

df


# In[86]:

df.info()


# In[87]:

df.shape


# In[88]:

df.isnull().any()


# In[89]:

df = df[df.horsepower != '?']


# In[90]:

df['horsepower']=df['horsepower'].astype(float)


# In[91]:

df.dtypes


# In[92]:

corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# In[93]:

sns.pairplot(data=df)


# In[94]:

sns.distplot(df['mpg'])


# In[95]:

df['Country_code'] = df.origin.replace([1,2,3],['USA','Europe','Japan'])


# In[96]:

df['brand name'], df['model name'] = df['name'].str.split(' ',1).str


# In[97]:

plt.figure(figsize=(12,6))
sns.countplot(x = "brand name", data=df)
t = plt.xticks(rotation=45)


# In[98]:

df.head(3)


# In[99]:

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[100]:

factors = ['cylinders','displacement','horsepower','acceleration','weight','origin']
X = pd.DataFrame(df[factors].copy())
y = df['mpg'].copy()


# In[101]:

X = StandardScaler().fit_transform(X)


# In[102]:

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)
X_train.shape[0] == y_train.shape[0]


# In[170]:

from sklearn.linear_model import Ridge
# Create linear regression object with a ridge coefficient 0.5
ridge = Ridge(fit_intercept=True, alpha=0.5)


# In[171]:

ridge.get_params()


# In[172]:

ridge.fit(X_train,y_train)


# In[173]:

y_predicted = ridge.predict(X_test)


# In[177]:

rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
rmse


# In[178]:

ridge.score(X_test,y_test)


# In[109]:

import statsmodels.formula.api as smf
import statsmodels.api as sm
result = smf.ols(formula='mpg ~ cylinders+displacement+horsepower+acceleration+weight+origin',data=df).fit()    
print(result.summary())    


# In[110]:

print('Coefficients: \n', regressor.coef_)
 
# variance score: 1 means perfect prediction
print('Variance score: {}'.format(regressor.score(X_test, y_test)))
 


# In[111]:


# plot for residual error
 
## setting plot style
plt.style.use('fivethirtyeight')
 
## plotting residual errors in training data
plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
 
## plotting residual errors in test data
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
 
## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 2)
 
## plotting legend
plt.legend(loc = 'upper right')
 
## plot title
plt.title("Residual errors")
 
## function to show plot
plt.show()


# In[148]:

from sklearn.kernel_ridge import KernelRidge
clf=KernelRidge(alpha=1.5)


# In[164]:

factors = ['cylinders','displacement','horsepower','acceleration','weight','origin']
X = pd.DataFrame(df[factors].copy())
y = df['mpg'].copy()
X = StandardScaler().fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.33,random_state=324)


# In[165]:

clf.get_params()


# In[166]:

clf.fit(X_train,y_train)


# In[167]:

y_predicted = clf.predict(X_test)


# In[168]:

rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_predicted))
rmse


# In[169]:

clf.score(X_test,y_test)


# In[179]:

df.head()


# In[4]:

demo=pd.read_csv("K:/Spring 2018/Statistics and Machine Learning/Datasets/diabetes .csv")


# In[5]:

demo.head()


# In[56]:

demo = demo[demo.Pregnancies != 0]
demo = demo[demo.Glucose != 0]
demo = demo[demo.SkinThickness != 0]
demo = demo[demo.BMI != 0]
demo = demo[demo.Age != 0]
demo


# In[93]:

from sklearn import svm
X = demo.iloc[:,1:3].values # we only take the first two features. We could
y=demo.iloc[:,8]
# avoid this ugly slicing by using a two-dim dataset
#y = iris.target
y


# In[94]:

C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=1,gamma='auto').fit(X, y)


# In[95]:

svc.score(X,y)


# In[96]:

X[:,0]
X[:,1]
X[:, 0].min()


# In[97]:

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
x_min


# In[99]:

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


# In[101]:

svc = svm.SVC(kernel='rbf', C=1,gamma='auto').fit(X, y)


# In[102]:

svc.score(X,y)


# In[103]:

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))


# In[104]:

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


# In[105]:

svc = svm.SVC(kernel='poly', C=1,gamma='auto').fit(X, y)


# In[106]:

svc.score(X,y)


# In[107]:

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
x_min


# In[108]:

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()


# In[ ]:



