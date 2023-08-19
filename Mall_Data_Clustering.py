#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


customer_data = pd.read_csv('C:\\Users\\kunal.singh\\Desktop\\Learnings\\datasets\\marketing_campaign.csv')


# In[3]:


customer_data = customer_data.dropna()


# In[4]:


customer_data.info()


# In[5]:


customer_data = customer_data.drop(labels = ['ID','Education','Marital_Status', 'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3',
                                  'AcceptedCmp4','AcceptedCmp5','Complain','Response'], axis = 1)


# In[6]:


customer_data.info()


# In[7]:


customer_data.head()


# In[8]:


customer_data.info()


# In[9]:


customer_data['Dt_Customer'] = pd.to_datetime(customer_data['Dt_Customer'], format = '%d/%m/%y')


# In[10]:


customer_data.info()


# In[11]:


customer_data.isnull().sum()


# In[12]:


customer_data['Total_amount_spent'] = customer_data['MntWines'] + customer_data['MntFruits'] + customer_data['MntMeatProducts'] + customer_data['MntFishProducts'] + customer_data['MntSweetProducts'] + customer_data['MntGoldProds']
    
customer_data['Total_children'] = customer_data['Kidhome'] + customer_data['Teenhome']

customer_data['Customer_purchases_webvisits'] = customer_data['NumDealsPurchases'] + customer_data['NumWebPurchases'] + customer_data['NumCatalogPurchases'] + customer_data['NumStorePurchases']


# In[13]:


customer_data.head()


# In[14]:


from datetime import datetime


# In[15]:


dt = datetime.today().date()


# In[16]:


dt = pd.to_datetime(dt, format = '%Y-%m-%d')


# In[17]:


customer_data['length_of_association'] = (dt - customer_data['Dt_Customer']).dt.days


# In[18]:


customer_data['length_of_association'] = ((customer_data['length_of_association'])/30).astype('int')


# In[19]:


customer_data.head().T


# In[20]:


customer_data['Age'] = (dt.year - customer_data['Year_Birth'])


# In[21]:


customer_data.head().T


# In[22]:


customer_data.describe().T


# In[23]:


customer_data = customer_data.drop(labels = ['Kidhome', 'Teenhome', 'Z_CostContact', 'Z_Revenue', 'Total_children' ], axis = 1)


# In[24]:


customer_data.head().T


# In[25]:


#Box plot to identify the data further and break it down further

fig, axs = plt.subplots(figsize = (12,8), nrows = 2, ncols = 3, sharey = False)

sns.boxplot(y = customer_data['Income'], ax = axs[0][0])

sns.boxplot(y = customer_data['Total_amount_spent'], ax = axs[0][1])

sns.boxplot(y = customer_data['Customer_purchases_webvisits'], ax = axs[0][2])

sns.boxplot(y = customer_data['Recency'], ax = axs[1][0])

sns.boxplot(y = customer_data['length_of_association'], ax = axs[1][1])

sns.boxplot(y = customer_data['Age'], ax = axs[1][2])


# In[26]:


customer_data.head().T


# In[27]:


customer_data ['Customer_purchases_webvisits'].clip( 
    lower = customer_data['Customer_purchases_webvisits'].quantile(0.5),
    upper = customer_data['Customer_purchases_webvisits'].quantile(0.95), inplace = True)

customer_data ['Total_amount_spent'].clip( 
    lower = customer_data['Total_amount_spent'].quantile(0.5),
    upper = customer_data['Total_amount_spent'].quantile(0.95), inplace = True)

customer_data ['Income'].clip( 
    lower = customer_data['Income'].quantile(0.5),
    upper = customer_data['Income'].quantile(0.95), inplace = True)

customer_data ['Age'].clip( 
    lower = customer_data['Age'].quantile(0.5),
    upper = customer_data['Age'].quantile(0.95), inplace = True)


# In[28]:


fig, axs = plt.subplots(figsize = (12,8), nrows = 2, ncols = 3, sharey = False)

sns.boxplot(y = customer_data['Income'], ax = axs[0][0])

sns.boxplot(y = customer_data['Total_amount_spent'], ax = axs[0][1])

sns.boxplot(y = customer_data['Customer_purchases_webvisits'], ax = axs[0][2])

sns.boxplot(y = customer_data['Recency'], ax = axs[1][0])

sns.boxplot(y = customer_data['length_of_association'], ax = axs[1][1])

sns.boxplot(y = customer_data['Age'], ax = axs[1][2])


# In[29]:


plt.figure(figsize = (12,8))

sns.scatterplot(x = 'Income', y = 'Total_amount_spent', data = customer_data, color = 'r')


# In[30]:


plt.figure(figsize = (12,8))

sns.scatterplot(x = 'Income', y = 'Total_amount_spent', data = customer_data, hue = 'length_of_association')


# In[31]:


plt.figure(figsize = (12,8))

sns.scatterplot(x = 'Income', y = 'length_of_association', data = customer_data, color = 'y')


# In[32]:


import plotly.express as px


# In[33]:


fig = px.scatter_3d(customer_data, x= 'length_of_association', y = 'Income', z = 'Total_amount_spent')
fig.show()


# In[34]:


customer_data_p = customer_data[['Income', 'Total_amount_spent', 'length_of_association']] #saving the pre processed data separately


# In[35]:


customer_data_p


# In[36]:


pip install plotly


# In[37]:


import numpy as np
import pandas as pd

import matplotlib.pyplot  as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize


# In[38]:


customer_data_p.describe()


# In[39]:


plt.figure(figsize = (6,4))
sns.boxplot(data= customer_data_p, y= 'Income')


# In[40]:


#standardizing the data

scaler = StandardScaler()

X_std = pd.DataFrame(data = scaler.fit_transform(customer_data_p), columns = customer_data_p.columns)

X_std.describe()


# In[41]:



X = pd.DataFrame(data=normalize(X_std, norm='l2'), columns=X_std.columns)
X.head()


# In[42]:


plt.figure(figsize = (12,8))

sns.scatterplot(data = X, x = 'Income', y = 'length_of_association', hue = 'Total_amount_spent')


# In[43]:


#Identifying own the correct set of cluster needed #Elbow Method

weighted_cluster_sum_of_squares = []
for k in range(1,20):
    kmeans_model = KMeans (n_clusters = k, random_state = 123)
    kmeans_model.fit(X)
    weighted_cluster_sum_of_squares.append(kmeans_model.inertia_)

plt.figure(figsize=(12,8))
sns.lineplot(x= range(1,20), y = weighted_cluster_sum_of_squares, linewidth = 2, color = 'blue', marker = '8')

plt.xlabel('Number Of Cluster')
plt.xticks(np.arange(1,20,1))

plt.ylabel('WCSS')
plt.show()


# In[44]:


silhouette_avg = [] #Using Silhouette score to understand number of clusters needed.

for num_clusters  in range (2,20):
    kmeans_model_1 = KMeans (n_clusters = num_clusters, random_state = 123)
    kmeans_model_1.fit(X)
    cluster_labels = kmeans_model_1.labels_
    silhouette_avg.append(silhouette_score(X, cluster_labels))
    

silhouette_avg


# In[45]:


plt.figure(figsize = (12,8)) 

sns.lineplot(x = range(2,20), y = silhouette_avg, linewidth = 2, color = 'green', marker = '8')

plt.xlabel('Number of Cluster')
plt.ylabel('Silhouette Score')
plt.show()


# In[46]:


kmeans_model = KMeans(n_clusters = 4, random_state = 123)
kmeans_model.fit(X)
X['cluster_labels'] = kmeans_model.labels_

X.head()


# In[47]:


X.groupby('cluster_labels').mean() #this show the centroid of each cluster


# In[48]:


#putting the clusters back into the original data

customer_data_p['cluster_labels'] = kmeans_model.labels_
customer_data_p.groupby('cluster_labels').mean()


# In[49]:


plt.figure(figsize=(12,8))

sns.scatterplot(x = 'Income', y = 'length_of_association', size = 'Total_amount_spent', hue = 'cluster_labels', data = X
               , palette = 'coolwarm_r')


# In[50]:


plt.figure(figsize=(12,8)) #No clear clustering with amount spend 

sns.scatterplot(x = 'Income', y = 'Total_amount_spent', size = 'length_of_association', hue = 'cluster_labels', data = X
               , palette = 'coolwarm_r')


# In[51]:


plt.figure(figsize=(12,8)) #No clear pattern in the data however clusters are visible

sns.scatterplot(x = 'Income', y = 'length_of_association', size = 'Total_amount_spent', hue = 'cluster_labels', data = customer_data_p
               , palette = 'coolwarm_r')


# In[52]:


fig = px.scatter_3d(X, x= 'length_of_association', y = 'Income', z = 'Total_amount_spent', color = 'cluster_labels')
fig.show() #3d chart shows different clusters in 3 dimentional space


# In[ ]:




