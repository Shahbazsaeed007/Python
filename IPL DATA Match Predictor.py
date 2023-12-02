#!/usr/bin/env python
# coding: utf-8

# ## Importing the neccessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


delivery = pd.read_csv('deliveries.csv')
delivery


# In[3]:


match = pd.read_csv('matches.csv')


# In[ ]:





# In[4]:


delivery.shape


# ## Using Group by to fetch each inning total runs with respect to match_id and storing it in a variable

# In[5]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df


# ## Filtering 

# In[6]:


total_score_df = total_score_df[total_score_df['inning']==1]
total_score_df


# ## Joining two DataFrames with the help Merge function

# In[7]:


match_df = pd.merge(match,total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df


# ## Preprocessing the data for our further analysis

# In[8]:


match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')


# In[9]:


match_df['team1']=match['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# ## Creating a list teams who playes IPL currently ( some of the teams dont play IPL now)

# In[10]:


teams=['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore','Kolkata Knight Riders','Kings XI Punjab',
      'Chennai Super Kings','Rajasthan Royals','Delhi Capitals']


# In[11]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[12]:


match_df.shape


# ## Fetching those matches when it did not rain

# In[13]:


match_df= match_df[match_df['dl_applied']==0]
match_df.shape


# ## Feature selection on which basis we create our model

# In[14]:


match_df = match_df[['match_id','city','winner','total_runs']]
match_df


# In[15]:


delivery_df = pd.merge(delivery,match_df,on='match_id')
delivery_df


# In[16]:


delivery_df.columns.values


# In[17]:


delivery_df = delivery_df[delivery_df['inning']==2]
delivery_df.shape


# In[18]:


delivery_df[['match_id','total_runs_x','total_runs_y']]


# ## Creating new column called current score using cumulativesum function

# In[19]:


delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_x'].cumsum()


# In[20]:


delivery_df[['current_score']]


# ## Calculating runs_left

# In[21]:


delivery_df['runs_left'] = delivery_df['total_runs_y']-delivery_df['current_score']


# In[22]:


delivery_df


# ## Calculating Balls Left

# In[23]:


delivery_df['balls_left'] = 126-(delivery_df['over']*6+delivery_df['ball'])


# In[24]:


delivery_df[['over','ball','balls_left']].head(12)


# ## Filling Nan values with 0 where played did not out 

# In[25]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)


# In[26]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x==0 else 1)


# In[27]:


delivery_df[['player_dismissed','balls_left']].head(42)


# ## Calculating wickets_left

# In[28]:


wickets =delivery_df.groupby('match_id')['player_dismissed'].cumsum().values


# In[29]:


delivery_df['wickets_left'] = 10 - wickets


# In[30]:


delivery_df.head(10)


# ## Calculating current run rate

# In[31]:


delivery_df['cur_run_rate'] = (delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[32]:


delivery_df[['cur_run_rate','current_score','balls_left','runs_left']]


# ## Calculating required run rate

# In[33]:


delivery_df['req_run_rate'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[34]:


delivery_df.rename(columns={'wickets':'wickets_left'})


# ## Create a function which calculates result of the match

# In[35]:


def result(x):
    return 1 if x['batting_team']==x['winner'] else 0


# In[36]:


delivery_df.apply(result,axis=1)


# In[37]:


delivery_df['result']=delivery_df.apply(result,axis=1)


# In[38]:


delivery_df


# In[39]:


delivery_df.columns.values


# In[40]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_y','cur_run_rate','req_run_rate','result']]


# ## Selected attributes

# In[41]:


final_df


# ## SHUFFLING THE ROWS

# In[42]:


final_df = final_df.sample(final_df.shape[0])


# In[43]:


final_df.sample()


# In[44]:


final_df = final_df[~(final_df['balls_left']==0)]


# In[61]:


final_df.isnull().sum()


# In[62]:


final_df.dropna(inplace=True)


# ### Dividing the data into X and y 

# In[63]:


X = final_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_y','cur_run_rate','req_run_rate']]


# In[64]:


y = final_df['result']


# In[ ]:





# In[ ]:





# In[65]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[66]:


X_train


# In[67]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trans_df= ColumnTransformer([('trans_df',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],remainder='passthrough')


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# In[69]:


pipe = Pipeline(steps=[('step1',trans_df),('step2',LogisticRegression(solver='liblinear'))])


# In[70]:


pipe.fit(X_train,y_train)


# In[77]:


y_pred = pipe.predict(X_test)


# In[79]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:





# In[87]:


pipe.predict_proba(X_test)


# In[ ]:





# In[107]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets_left','total_runs_y','cur_run_rate','req_run_rate']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_y'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[160]:


temp_df,target = match_progression(delivery_df,74,pipe)
temp_df


# In[161]:


plt.figure(figsize=(20,10))
sns.lineplot(x= temp_df['end_of_over'],y=temp_df['wickets_in_over'],color='yellow',linewidth=3)
sns.lineplot(x=temp_df['end_of_over'],y=temp_df['win'],color='green',linewidth=4)
sns.lineplot(x=temp_df['end_of_over'],y=temp_df['lose'],color='red',linewidth=4)
sns.barplot(x=temp_df['end_of_over'],y=temp_df['runs_after_over'])
plt.title('Target-'+str(target))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




