
# coding: utf-8

# In[108]:


import pandas as pd


# In[109]:


df=pd.read_csv("haberman.csv",header=None)
df.columns=['age','operation_at_age','number_auxilary_nodes','survival_status']
df.shape


# In[110]:


df.columns


# In[111]:


print("Number of points is %d" %(df.shape[0]))


# In[112]:


print("Number of features is %d" %(df.shape[1]-1))


# In[113]:


len(df['survival_status'].unique())


# In[114]:


print("Number of different types of classes are %d"%(len(df['survival_status'].unique())))
print("which are")
print(df['survival_status'].unique())


# In[115]:


#chnage level 2 to 0
def change_2_to_0(x):
    if x==2:
        return 0
    else:
        return 1
#print(df['survival_status'][7])
df['survival_status']=df['survival_status'].apply(change_2_to_0)
#print(df['survival_status'][7])


# In[116]:


df['survival_status'].value_counts()


# In[117]:


print("This is a imbalance data set as the ratio of positive to negative points is %d : %d" %(len(df[df['survival_status']==1]),len(df[df['survival_status']==0])))


# In[118]:


import matplotlib.pyplot as plt
df.plot.scatter(x='age',y='operation_at_age')
plt.show()


# In[120]:


import seaborn as sns
sns.set_style("whitegrid")
sns.FacetGrid(df,hue='survival_status').map(plt.scatter,'number_auxilary_nodes','age').add_legend()
plt.show()


# In[121]:


## 3d Scatter plot


# In[122]:


import plotly as py
import plotly.graph_objs as go
py.offline.init_notebook_mode(connected=True)


# In[123]:


trace1 = go.Scatter3d(
    x=df['age'],
    y=df['operation_at_age'],
    z=df['number_auxilary_nodes'],
    mode='markers',
    marker=dict(
        size=12,
        color=df['survival_status'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='3d-scatter-colorscale')


# In[126]:


df.columns


# In[148]:


#pair plots


# In[149]:


sns.pairplot(df,hue='survival_status')


# In[150]:


print("From 3d scatter plot and pair plot i assume that this data set is inseparable. Lets see what happens")


# In[158]:


# PDF of sll 3 featured


# In[159]:


sns.FacetGrid(df,hue='survival_status',height=6).map(sns.distplot,'age').add_legend()


# In[160]:


sns.FacetGrid(df,hue='survival_status',height=6).map(sns.distplot,'operation_at_age').add_legend()


# In[161]:


sns.FacetGrid(df,hue='survival_status',height=6).map(sns.distplot,'number_auxilary_nodes').add_legend()


# In[195]:


#plotting pdf and cdf


# In[200]:


import numpy as np
count,bin_edges=np.histogram(df['operation_at_age'],density=True,bins=10)
pdf=count/sum(count)
#print(pdf)
cdf=np.cumsum(pdf)
print(cdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# In[201]:


import numpy as np
count,bin_edges=np.histogram(df['age'],density=True,bins=10)
pdf=count/sum(count)
#print(pdf)
cdf=np.cumsum(pdf)
print(cdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# In[202]:


import numpy as np
count,bin_edges=np.histogram(df['number_auxilary_nodes'],density=True,bins=10)
pdf=count/sum(count)
#print(pdf)
cdf=np.cumsum(pdf)
print(cdf)
print(bin_edges)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)


# In[203]:


df.columns


# In[204]:


sns.boxplot(data=df,x='survival_status',y='age')
plt.show()


# In[205]:


sns.boxplot(data=df,x='survival_status',y='operation_at_age')
plt.show()

# In[206]:


sns.boxplot(data=df,x='survival_status',y='number_auxilary_nodes')
plt.show()


# In[210]:


sns.violinplot(data=df,x='survival_status',y='age')
plt.show()


# In[211]:


sns.violinplot(data=df,x='survival_status',y='operation_at_age')
plt.show()

# In[212]:


sns.violinplot(data=df,x='survival_status',y='number_auxilary_nodes')
plt.show()
