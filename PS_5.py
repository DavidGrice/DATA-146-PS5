
# coding: utf-8

# In[1]:


# Libraries imported
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import math
from statsmodels.formula.api import ols
from statsmodels.stats import anova
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# # Question 1 -> 4

# In[2]:


# import data
classy = pd.read_csv("class_data.txt", header = None, sep = "\t", names = ["Standing", "Miles", "Coordinates Lower", "Coordinates Upper"])


# In[3]:


# Read data
classy


# In[4]:


# Q2 + Q3 creating graphs representing data
standing = ['Senior', 'Junior', 'Sophomore']
colors = ['blue', 'red', 'purple']
f, ax = plt.subplots(2,2, figsize = (6,6), sharex=True, sharey=True)
plt.subplots_adjust(hspace=.5)
for j,a,c in zip(standing, ax.flat,colors):
    a.set_title(j)
    sns.distplot(classy.groupby('Standing').get_group(j)['Miles'], kde=False, ax= a, color=c)


# In[5]:


# Q4 Levene test
g = classy.groupby('Standing')
data_list = [item['Miles'] for name,item in g]
variances = [i.var(ddof=1) for i in data_list]
print(variances)
ss.levene(data_list[0],data_list[1],data_list[2])


# In[6]:


classy['Standing'] = pd.Categorical(classy['Standing'], ordered=True, categories=['Sophomore','Junior','Senior'])


# In[7]:


# ANOVA portion
#print('Grand mean:',classy['Standing'].mean())
print('Group means:',classy.groupby('Standing').mean()['Miles'].values)


# In[8]:


# ANOVA graphing
plt.figure(figsize=[8,8])
sns.stripplot(y = classy['Miles'], x = classy['Standing'], orient='v')
minx = plt.xlim()[0]
maxx = plt.xlim()[1]
xmins = [minx + i*(maxx-minx)/4 for i in range(4)]
xmaxs = [maxx - i*(maxx-minx)/4 for i in range(3,-1,-1)]
plt.hlines(xmin = plt.xlim()[0], xmax = plt.xlim()[1], y=classy['Miles'].mean(), color='black')
plt.hlines(xmin = xmins, xmax = xmaxs, y = classy.groupby('Standing').mean()['Miles'].values, colors = ['blue','orange','green','red'] )


# In[9]:


# QUESTION 5
import sklearn.manifold as skm
import scipy
#import folium
import seaborn as sns


# In[10]:


scotus = pd.read_csv('SCOTUS.txt', index_col = 0, sep = "\t")
dis = scotus.as_matrix()
sns.heatmap(dis, cmap = 'BuPu',xticklabels=scotus.index, yticklabels=scotus.index)


# In[11]:


embedding = skm.MDS(n_components=2, dissimilarity='precomputed')
X_transformed = embedding.fit_transform(dis)
print(X_transformed)


# In[12]:


z = zip(X_transformed[:,0], X_transformed[:,1])
judges = scotus.index
ax = sns.regplot(x=X_transformed[:,0], y=X_transformed[:,1] ,fit_reg=False)
for i,zi in enumerate(z):
    ax.annotate(s = judges[i], xy =zi )


# In[13]:


import pandas as pd
import numpy as np
import scipy.spatial.distance as ssd
import seaborn as sns
import sklearn.manifold.mds as skm
import matplotlib.pyplot as plt


# In[14]:


dist = ssd.pdist(scotus.as_matrix(), metric='euclidean')
disq = ssd.squareform(dist)
dist[0] #dist[0:,]


# In[15]:


embedding = skm.MDS(n_components=2, dissimilarity='precomputed', metric = True)
X_transformed = embedding.fit_transform(disq)


# In[16]:


scotusmeta = pd.read_csv('SCOTUS_metadata.txt', sep = "\t")


# In[17]:


scotusmeta


# In[18]:


#arr = scotusmeta.as_matrix()
#embedding = skm.MDS(n_components=3, dissimilarity='precomputed', metric = True)
#X_transformed = embedding.fit_transform(arr)


# In[30]:


z = zip(X_transformed[:,0], X_transformed[:,1])
names = scotusmeta['ID']
party = scotusmeta['Ideology']
colors = {'Conservative':'red','Liberal':'blue','Center':'green'}
plot_col = scotusmeta['Ideology'].apply(lambda x: colors[x])
ax = sns.regplot(x=X_transformed[:,0], y=X_transformed[:,1], fit_reg=False)
plt.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=plot_col)
for i,zi in enumerate(z):
    ax.annotate(s = names[i], xy = zi, color = colors[party[i]])


# In[20]:


scotusmeta


# In[21]:


scotus


# In[28]:


z2 = zip(X_transformed[:,0], X_transformed[:,1])
names = scotusmeta['ID']
gender = scotusmeta['Gender']
colors2 = {'Female':'purple','Male':'blue'}
plot_col = scotusmeta['Ideology'].apply(lambda x: colors[x])
ax2 = sns.regplot(x=X_transformed[:,0], y=X_transformed[:,1], fit_reg=False)
plt.scatter(x=X_transformed[:,0], y=X_transformed[:,1], c=plot_col)
for i,zi in enumerate(z2):
    ax2.annotate(s = names[i], xy = zi, color = colors2[gender[i]])


# In[23]:


south = pd.read_csv("South_with_category.csv")


# In[24]:


south


# In[25]:


immune = pd.read_csv("AS.txt", sep = "\t")


# In[26]:


immune


# In[35]:


# Q17->20 creating graphs representing data
condition = ['AS', 'RA', 'Ctrl']
colors4 = ['blue', 'red', 'green']
f, ax = plt.subplots(2,2, figsize = (6,6), sharex=True, sharey=True)
plt.subplots_adjust(hspace=.5)
for j,a,c in zip(condition, ax.flat,colors4):
    a.set_title(j)
    sns.distplot(immune.groupby('Condition').get_group(j)['Clostridium'], kde=False, ax= a, color=c)


# In[36]:


# ANOVA portion
#print('Grand mean:',classy['Standing'].mean())
print('Group means:',immune.groupby('Condition').mean()['Clostridium'].values)


# In[55]:


death = immune.groupby('Condition')
death_list = [item['Clostridium'] for name,item in death]
variancesofdeath = [i.var(ddof=1) for i in death_list]
print(variancesofdeath)
ss.levene(death_list[0],death_list[1],death_list[2])


# In[39]:


# ANOVA graphing
plt.figure(figsize=[8,8])
sns.stripplot(y = immune['Clostridium'], x = immune['Condition'], orient='v')
minx = plt.xlim()[0]
maxx = plt.xlim()[1]
xmins = [minx + i*(maxx-minx)/4 for i in range(4)]
xmaxs = [maxx - i*(maxx-minx)/4 for i in range(3,-1,-1)]
plt.hlines(xmin = plt.xlim()[0], xmax = plt.xlim()[1], y=immune['Clostridium'].mean(), color='black')
plt.hlines(xmin = xmins, xmax = xmaxs, y = immune.groupby('Condition').mean()['Clostridium'].values, colors = ['blue','orange','green'] )


# In[56]:


states = pd.read_csv("South_With_Category_Ints.csv")
namestate = states[rows = "Arkansas", "Louisiana", "Oklahoma", "Texas", "Alabama", "Kentucky", "Mississippi", "Tennesse", "Florida", "Georgia", "North.Carolina", "South.Carolina", "Virginia", "West.Virginia", "Delaware", "Arizona", "New.Mexico", "Colorado", "Kansas", "Missouri", "Illinois", "Indiana", "Ohio", "Pennsylvania"]


# In[52]:


table_1 = pd.crosstab(states.index, states['category'])
table_1


# In[53]:


sns.heatmap(table_1, cmap = 'BuPu', xticklabels=['A_Lot','Not_At_All','Not_Much',                                                'Some'])
plt.xticks(rotation = 45)

