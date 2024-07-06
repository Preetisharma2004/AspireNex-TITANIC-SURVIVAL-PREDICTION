#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt


# In[30]:


titanic_data=pd.read_csv('train.csv')


# In[31]:


titanic_data.head()


# In[32]:


titanic_data.describe()


# In[33]:


import seaborn as sns

sns.heatmap(titanic_data.corr(),cmap="YlGnBu")
plt.show()


# In[34]:


from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2)
for train_indices,test_indices in split.split(titanic_data,titanic_data[["Survived","Pclass","Sex"]]):
    strat_train_set =titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[35]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()
plt.subplot(1,2,2)
strat_test_set['Survived'].hist()
strat_test_set['Pclass'].hist()

plt.show()


# In[36]:


strat_train_set.info()


# In[37]:


from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        imputer =SimpleImputer(strategy="mean")
        X['Age']=imputer.fit_transform(X[['Age']])
        return X


# In[38]:


from sklearn.preprocessing import OneHotEncoder
class FeatureEncoder(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        encoder=OneHotEncoder()
        matrix=encoder.fit_transform(X[['Embarked']]).toarray()
        
        column_names=["C","S","Q","N"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]
                       
        matrix=encoder.fit_transform(X[['Sex']]).toarray()
        column_names =["Female","Male"]
        
        for i in range(len(matrix.T)):
            X[column_names[i]]=matrix.T[i]        
            
        return X


# In[39]:


class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"],axis=1,errors="ignore")


# In[40]:


from sklearn.pipeline import Pipeline

pipeline =Pipeline([("ageimputer",AgeImputer()),
                  ("featureencoder",FeatureEncoder()),
                   ("featuredropper",FeatureDropper())])


# In[41]:


strat_train_set =pipeline.fit_transform(strat_train_set)


# In[42]:


strat_train_set


# In[43]:


strat_train_set.info()


# In[44]:


from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


X = strat_train_set.drop(['Survived'], axis=1)
y = strat_train_set['Survived']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


scaler = StandardScaler()
X_data = scaler.fit_transform(X_imputed)
y_data = y.to_numpy()


X_data = np.nan_to_num(X_data, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)


clf = RandomForestClassifier()

# Define th
param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

# Perform GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)


# In[45]:


final_clf=grid_search.best_estimator_


# In[46]:


final_clf


# In[47]:


strat_test_set=pipeline.fit_transform(strat_test_set)


# In[48]:


strat_test_set


# In[49]:


X_test =strat_test_set.drop(['Survived'],axis=1)
y_test = strat_test_set['Survived']

scaler = StandardScaler()
X_data_test =scaler.fit_transform(X_test)
y_data_test =y_test.to_numpy()


# In[50]:


final_clf.score(X_data_test,y_data_test)


# In[51]:


final_data =pipeline.fit_transform(titanic_data)
final_data


# In[52]:


X_final = final_data.drop(['Survived'], axis=1)
y_final = final_data['Survived']

# Combine X_final and y_final to ensure consistent cleaning
combined = pd.concat([X_final, y_final], axis=1)

# Check and handle NaNs and infinities in the combined DataFrame
combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

# Separate the cleaned data back into X_final and y_final
X_final = combined.drop(['Survived'], axis=1)
y_final = combined['Survived']

# Normalize data
scaler = StandardScaler()
x_data_final = scaler.fit_transform(X_final)
y_data_final = y_final.to_numpy()

# Fit classifier
prod_clf = RandomForestClassifier()

param_grid = [
    {"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}
]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(x_data_final, y_data_final)


# In[53]:


prod_final_clf=grid_search.best_estimator_
prod_final_clf


# In[54]:


titanic_test_data =pd.read_csv("test.csv")


# In[55]:


final_test_data =pipeline.fit_transform(titanic_test_data)


# In[56]:


final_test_data.info()


# In[59]:


x_final_test=final_test_data
x_final_test = x_final_test.fillna(method="ffill")

scaler=StandardScaler()
X_data_final_test =scaler.fit_transform(x_final_test)


# In[60]:


prediction =prod_final_clf.predict(X_data_final_test)


# In[63]:


final_df =pd.DataFrame(titanic_test_data['PassengerId'])
final_df['Survived']=prediction
final_df.to_csv("prediction.csv",index=False)


# In[64]:


final_df


# In[ ]:




