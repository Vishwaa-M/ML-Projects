#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[3]:


data=datasets.load_breast_cancer()
print(data)


# In[4]:


data.keys()


# In[5]:


#data frame loading
df=pd.DataFrame(data.data,columns=data.feature_names)


# In[6]:


df.head()


# In[7]:


df.keys()


# In[8]:


df["diagnosis"]=data.target
df.head()


# In[9]:


df.describe()


# In[10]:


#target distribution
df["diagnosis"].value_counts()


# In[11]:


#no of rows and columns
df.shape


# In[12]:


df.info()


# In[13]:


#checking for null values
df.isnull().sum()


# In[14]:


x=df.drop(columns="diagnosis",axis=1)
y=df["diagnosis"]


# In[15]:


print(x)


# In[16]:


print(y)


# In[17]:


# dividing dataset into training and test
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[18]:


#model training by logistic regression
model=LogisticRegression()


# In[19]:


model.fit(xtrain,ytrain)


# In[20]:


#accuracy on training data
xtrain_prediction= model.predict(xtrain)
training_data_accuracy=accuracy_score(ytrain,xtrain_prediction)
print(training_data_accuracy)


# In[21]:


#accuracy on test data
xtest_prediction= model.predict(xtest)
test_data_accuracy=accuracy_score(ytest,xtest_prediction)
print(test_data_accuracy)


# In[22]:


parameter_grid=[
    {'penalty':['l1', 'l2', 'elasticnet', 'none'], 
     'C': np.logspace(-4,4,20),
     'solver':['lbfgs','newton-cg', 'liblinear', 'sag', 'saga'],
     'max_iter':[100,1000,2500,5000]
    }
]


# In[25]:


clf=GridSearchCV(model,param_grid=parameter_grid,cv=3,verbose=1,n_jobs=-1)


# In[26]:


best_clf=clf.fit(xtrain,ytrain)


# In[27]:


best_clf.best_estimator_


# In[28]:


# training accuracy after using hyper parameters
xtrain__predictions= best_clf.predict(xtrain)
trainingdata_accuracy=accuracy_score(ytrain,xtrain__predictions)
print(trainingdata_accuracy)


# In[29]:


# testing accuracy after using hyper parameters
xtest__predictions= best_clf.predict(xtest)
testdata_accuracy=accuracy_score(ytest,xtest__predictions)
print(testdata_accuracy)


# In[30]:


# checking prediction of the model
inputdata=(9.173,13.86,59.2,260.9,0.07721,0.08751,0.05988,0.0218,0.2341,0.06963,0.4098,2.265,2.608,23.52,0.008738,0.03938,0.04312,0.0156,0.04192,0.005822,10.01,19.23,65.59,310.1,0.09836,0.1678,0.1397,0.05087,0.3282,0.0849)
inputdata_in_numpyarray=np.asarray(inputdata)


# In[31]:


#reshaping as 1 datapoint
reshaped_inputdata=inputdata_in_numpyarray.reshape(1,-1)


# In[32]:


prediction=best_clf.predict(reshaped_inputdata)
if(prediction==0):
    print("the cancer is malignant")
else:
    print("the cancer is bening")


# In[ ]:




