
# coding: utf-8

# # HW 2

# In[6]:

import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
import os
import sklearn

if int(os.environ.get("MODERN_PANDAS_EPUB", 0)):   #setting up the os environment
    import prep 
pd.options.display.max_rows=10


# ## Normal Distribution

# In[7]:

mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 100)
s


# In[8]:

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate some data for this demonstration.
data = norm.rvs(10.0, 2.5, size=500)

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()


# ## Exponential 

# In[9]:

from scipy.stats import expon
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


# In[10]:

mean, var, skew, kurt = expon.stats(moments='mvsk')


# In[11]:

#Display the probability density function (pdf):

x = np.linspace(expon.ppf(0.01), expon.ppf(0.99), 100)
ax.plot(x, expon.pdf(x), 'r-', lw=5, alpha=0.6, label='expon pdf')
plt.show()


# In[12]:

#Check accuracy of cdf and ppf:


# In[13]:

vals = expon.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], expon.cdf(vals))


# In[16]:

import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-1, 2, 100)
    y = np.exp(x)

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('$x$')
    plt.ylabel('$\exp(x)$')

    plt.figure()
    plt.plot(x, -np.exp(-x))
    plt.xlabel('$x$')
    plt.ylabel('$-\exp(-x)$')

    plt.show()

if __name__ == '__main__':
    main()


# In[20]:

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt

# Generate some data for this demonstration.
data = expon.rvs(10.0, 2.5, size=500)

# Fit a exponential distribution to the data:
mu, std = expon.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, normed=True, alpha=0.6, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 1000)
p = expon.pdf(x,loc= mu,scale= std)
a=expon.pdf(x)
#we now plot the graph
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)

plt.title(title)

plt.show()


# In[ ]:

#### We now assess the distribution using r square:


# In[21]:


from sklearn.metrics import r2_score
r2_score(p,a)


# In[25]:


from scipy.stats import bernoulli
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
#Calculate a few first moments:
p = 0.3
mean, var, skew, kurt = bernoulli.stats(p, moments='mvsk')
#Display the probability mass function (pmf):

x = np.arange(bernoulli.ppf(0.01, p),bernoulli.ppf(0.99, p))
ax.plot(x, bernoulli.pmf(x, p), 'bo', ms=8, label='bernoulli pmf')
ax.vlines(x, 0, bernoulli.pmf(x, p), colors='b', lw=5, alpha=0.5)
plt.show()


# In[ ]:



