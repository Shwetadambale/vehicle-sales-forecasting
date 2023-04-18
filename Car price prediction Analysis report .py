#!/usr/bin/env python
# coding: utf-8

# In[35]:


import warnings


# In[36]:


warnings.filterwarnings('ignore')


# In[37]:


import pandas as pd 


# In[38]:


data = pd.read_csv('car data.csv')


# # 1. Display top 5 rows of the dataset 

# In[39]:


data.head()


# In[40]:


data.tail()


# # 2.Find the shape of our dataset (Number of rows and Number of colums )

# In[41]:


data.shape


# In[42]:


print ("Number of rows",data.shape[0])
print ("Number of colmuns",data.shape[1])


# # 3.Get the information about the dataset like the total Numbers of rows , Numbers of columns,datatype of each columns and memory requirements  

# In[43]:


data.info()


# In[20]:


#Find the missing value of the dataset 
data.isnull().sum()


# In[22]:


# Get the over all statastic value of the dataset 
data.describe()


# # 4.Data processing

# In[44]:


data.head()


# In[45]:


import datetime
date_time = datetime.datetime.now()


# In[46]:


date_time.year


# #Adding a new feature (that is Age)

# In[47]:


data['Age'] = date_time.year - data['Year']


# In[48]:


data.head()


# # Outlier Removal

# In[49]:


import seaborn as sns


# In[50]:


sns.boxplot (data['Selling_Price'])


# In[51]:


sorted(data['Selling_Price'],reverse= True)


# In[52]:


data[(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]


# In[55]:


data=data[~(data['Selling_Price']>=33.0) & (data['Selling_Price']<=35.0)]


# In[56]:


#For check wheather the outlier remove or not we have to recheck
data.shape


# # 5.Encoding a catogorical column 

# In[57]:


data.head()


# In[58]:


data['Fuel_Type'].unique()


# In[59]:


data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})


# In[60]:


data['Fuel_Type'].unique()


# In[61]:


data['Seller_Type'].unique()


# In[62]:


data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0,'Individual':1,})


# In[63]:


data['Seller_Type'].unique()


# In[64]:


data['Transmission'].unique()


# In[65]:


data['Transmission'] = data['Transmission'].map({'Manual':0,'Automatic':1,})


# In[66]:


data['Seller_Type'].unique()


# In[67]:


# for reacheck of wheather the data is encode or not 
data.head()


# In[100]:


X = data.drop(['Car_Name','Selling_Price'],axis=1)
y = data['Selling_Price']


# In[101]:


# Machine learning Algorithum 


# # Steps by step 
# Split the data 
# 1 step : Create a object 
# 2 step : Fit the machine learning model on train data set 
# 3 step : Predict the test data set using trained machine learning model .
# 4 step Estimate the score using performance matrix . 
# 

# In[102]:


X


# In[103]:


y


# # 5. Splitting the dataset into the traing dataset and test dataset 

# In[104]:


from sklearn.model_selection import train_test_split


# In[105]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[106]:


data.head()


# In[107]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# # Train the model 

# In[108]:



lr = LinearRegression()
lr.fit(X_train,y_train)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)

xgb = GradientBoostingRegressor()
xgb.fit(X_train,y_train)

xg = XGBRegressor()
xg.fit(X_train,y_train)


# # 7. Prediction on Test Data

# In[109]:


y_pred1 = lr.predict(X_test)
y_pred2 = rf.predict(X_test)
y_pred3 = xgb.predict(X_test)
y_pred4 = xg.predict(X_test)


# # 8.Evaluating the Algorithm
# 

# In[110]:


from sklearn import metrics


# In[111]:


score1 = metrics.r2_score(y_test,y_pred1)
score2 = metrics.r2_score(y_test,y_pred2)
score3 = metrics.r2_score(y_test,y_pred3)
score4 = metrics.r2_score(y_test,y_pred4)


# In[112]:


print(score1,score2,score3,score4)


# In[113]:


final_data = pd.DataFrame({'Models':['LR','RF','GBR','XG'],
             "R2_SCORE":[score1,score2,score3,score4]})


# In[114]:


final_data


# In[115]:


sns.barplot(final_data['Models'],final_data['R2_SCORE'])


# # Save The Model

# In[116]:


xg = XGBRegressor()
xg_final = xg.fit(X,y)


# In[117]:


import joblib


# In[118]:


joblib.dump(xg_final,'car_price_predictor')


# # Prediction on New Data

# In[122]:


import pandas as pd


# In[123]:


data_new = pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':0,
    'Age':8
},index=[0])


# In[127]:


model.predict(data_new)


# In[ ]:





# In[ ]:




