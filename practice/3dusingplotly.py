# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:12:03 2018

@author: prabhudayala
"""

print("helo")

import numpy as np
x1=np.array([1,2,3])
x2=np.array([11,12,13])
x3=np.array([21,22,23])
x4=np.array([31,32,33])
x=np.vstack((x1,x2,x3,x4))
import pandas as pd
df=pd.DataFrame(x,columns=['one','two','three'])
print(df.shape)
y=np.array([0,0,1,1])
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x,y)
print(model.coef_)
print(model.intercept_)
xq=np.array([15,16,17])
xq=xq.reshape((1,3))
print(model.predict(xq))

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
trace1 = go.Scatter3d(
    x=df['one'],
    y=df['two'],
    z=df['three'],
    mode='markers',
    marker=dict(
        size=12,
        color=y,                # set color to an array/list of desired values
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
plot(go.Figure(data=data, layout=layout))
