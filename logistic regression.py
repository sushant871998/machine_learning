
# coding: utf-8

# In[119]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('C:\\Users\\Sushant\\Desktop\\bank.csv',header=0,sep=';')
data=data.dropna()
print(data)


# In[88]:


data['education'].unique()


# In[120]:


data.groupby('job').mean()


# In[51]:


pr=pd.crosstab(data.job,data.y)
pr.plot(kind='bar')
plt.show()


# In[121]:


cat_vars=['job','marital','education','default','housing','loan','contact','day','month','poutcome' ]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list=pd.get_dummies(data[var],prefix=var)
    print(var,"-----",cat_list.columns)
    data1=data.join(cat_list)
    data=data1
   
    


# In[98]:


cat_vars=['job','marital','education','default','housing','loan','contact','day','month','poutcome' ]
data_vars=data.columns.values.tolist()
to_keep=[i for i in  data_vars if i not in cat_vars]


# In[122]:


data_final=data[to_keep]

data_final.columns.values


# In[133]:


data_final_val=data_final.columns.values.tolist()
Y=['y']
X=[i for i in data_final_val if i not in Y]


# In[136]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
rfe=RFE(logreg,18)
rfe=rfe.fit(data_final[X],data_final[Y])
print(rfe.ranking_)
print(rfe.support_)


# In[139]:


pref_indexes=list(np.where(rfe.ranking_==1)[0])
cols=list(np.asarray(X)[pref_indexes])
X=data_final[cols]
y=data_final['y']


# In[140]:


len(cols)


# In[142]:


print(cols)


# In[144]:


from sklearn.model_selection import train_test_split


# In[175]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=0)


# In[176]:


from sklearn import metrics
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
Y_pred=logreg.predict(X_test)
print(Y_pred)


# In[177]:


print(logreg.score(X_test,y_test))

