#!/usr/bin/env python
# coding: utf-8

# In[61]:


#improrting libraries
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# In[62]:


iris=pd.read_csv("Iris.csv") 
print(iris)


# In[63]:


iris.head()


# In[64]:


iris.keys()


# In[65]:


iris["Species"].value_counts()


# In[66]:


sns.FacetGrid(iris, hue="Species", height=10).map(plt.scatter,"SepalLengthCm","PetalLengthCm").add_legend()


# In[67]:


flowermapping={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
iris["Species"]=iris["Species"].map(flowermapping)


# In[68]:


# preparing the training set
x=iris.iloc[:,:-1]
y=iris.iloc[:,-1]


# In[69]:


#ploting to understand relationship between species and the other attributes
plt.xlabel("features")
plt.ylabel("species")

pltx=iris.loc[:,"SepalLengthCm"]
plty=iris.loc[:,"Species"]
plt.scatter(pltx,plty,label='SepalLengthCm')

pltx=iris.loc[:,"SepalWidthCm"]
plty=iris.loc[:,"Species"]
plt.scatter(pltx,plty,label='SepalWidthCm')

pltx=iris.loc[:,"PetalLengthCm"]
plty=iris.loc[:,"Species"]
plt.scatter(pltx,plty,label='PetalLengthCm')

pltx=iris.loc[:,"PetalWidthCm"]
plty=iris.loc[:,"Species"]
plt.scatter(pltx,plty,label='PetalWidthCm')
plt.legend()


# In[70]:


#spliting the data set into test and training
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


# In[71]:


#model training using logistic regression
model=LogisticRegression()


# In[72]:


model.fit(xtrain,ytrain)


# In[73]:


xtrain_prediction=model.predict(xtrain)
training_data_accuracy=accuracy_score(ytrain,xtrain_prediction)
print(training_data_accuracy)


# In[74]:


xtest_prediction=model.predict(xtest)
test_data_accuracy=accuracy_score(ytest,xtest_prediction)
print(test_data_accuracy)


# In[75]:


#using hyper parameters
parameter_grid=[
    {'penalty':['l1', 'l2', 'elasticnet', 'none'], 
     'C': np.logspace(-4,4,20),
     'solver':['lbfgs','newton-cg', 'liblinear', 'sag', 'saga'],
     'max_iter':[100,1000,2500,5000]
    }
]


# In[76]:


clf=GridSearchCV(model,param_grid=parameter_grid,cv=3,verbose=1,n_jobs=-1)


# In[77]:


best_clf=clf.fit(xtrain,ytrain)


# In[78]:


#finding the best hyper parameter usage
best_clf.best_estimator_


# In[79]:


#after using hyper parameters
#fr train
xtrain__predictions= best_clf.predict(xtrain)
trainingdata_accuracy=accuracy_score(ytrain,xtrain__predictions)
print(trainingdata_accuracy)


# In[80]:


#after using hyper parameters
#for test
xtest__predictions= best_clf.predict(xtest)
testdata_accuracy=accuracy_score(ytest,xtest__predictions)
print(testdata_accuracy)


# In[81]:


#model testing
expected=ytest
predictions=best_clf.predict(xtest)
print(predictions)


# In[82]:


print(classification_report(expected,predictions))


# In[83]:


print(accuracy_score(expected,predictions))


# In[84]:


#checking the model predictions
input_data=(102,5.8,2.7,5.1,1.9)
data_numpyarray=np.asarray(input_data)


# In[85]:


#reshaping as 1 datapoint
reshaped_input=data_numpyarray.reshape(1,-1)


# In[86]:


predicted=best_clf.predict(reshaped_input)


# In[87]:


if(predicted==0):
    print("The flower is Iris-setosa")
elif(predicted==1):
    print("The flower is Iris-versicolor")
else:
    print("The flower is Iris-virginica")


# In[ ]:




