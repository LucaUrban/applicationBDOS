import streamlit as st
import time
import base64
import os
from urllib.parse import quote as urlquote
from urllib.request import urlopen
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import json
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# importing the table and all other necessary files 
table = pd.read_csv("https://raw.github.com/LucaUrban/applicationBDOS/main/train_streamlit.csv")

# selection boxes columns
col_an = [col for col in list(table) if len(table[col].unique()) < 10 or is_numeric_dtype(table[col])]
col_mul = [col for col in list(table) if is_numeric_dtype(table[col])]
lis_check = [{'label': col, 'value': col} for col in col_mul if col != col_mul[0]]

# showing the table with the data
st.write("Data contained into the dataset:", table)

# mono variable analysis part
st.header("Monovariable Analysis")

st.sidebar.subheader("Monovariable Area")
monoVar_col = st.sidebar.selectbox("select the monovariable feature", col_an, 0)

if len(table[monoVar_col].unique()) > 10:
    monoVar_plot = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = table[monoVar_col].mean(),
        delta = {"reference": 2 * table[monoVar_col].mean() - table[monoVar_col].quantile(0.95)},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {'axis': {'range': [table[monoVar_col].min(), table[monoVar_col].max()]},
                 'steps' : [
                     {'range': [table[monoVar_col].min(), table[monoVar_col].quantile(0.05)], 'color': "lightgray"},
                     {'range': [table[monoVar_col].quantile(0.95), table[monoVar_col].max()], 'color': "gray"}],},
        title = {'text': "Gauge plot for the variable: " + monoVar_col}))
else:
    monoVar_plot = px.pie(table, names = monoVar_col, title = "Pie chart for the variable: " + monoVar_col)

st.plotly_chart(monoVar_plot, use_container_width=True)

# multi variable analysis part
st.header("Multivariable Analysis")

st.sidebar.subheader("Multivariable Area")
multi_index = st.sidebar.selectbox("multivariable index col", table.columns, 1)
multi_time = st.sidebar.selectbox("multivariable time col", table.columns, 3)
multiXax_col = st.sidebar.selectbox("multivariable X axis col", col_mul, 1)
multiYax_col = st.sidebar.selectbox("multivariable Y axis col", col_mul, 2)
multiSlider = st.sidebar.slider("multivarible time value", int(table[multi_time].min()), int(table[multi_time].max()), int(table[multi_time].min()))

def myFun():
    return 1

def create_time_series(dff, title, id_col, time_col):
    fig = go.Figure()
    if dff.shape[0] != 0:
        x_bar = []
        for inst in table[id_col].unique():
            inst_data = table[table[id_col] == inst][list(dff)[1]]
            if inst_data.count() != 0:
                x_bar.append(inst_data.mean())

        x_barbar = round(sum(x_bar)/len(x_bar), 3)

        x_LCL = x_barbar - (1.88 * (dff[list(dff)[1]].quantile(0.95) - dff[list(dff)[1]].quantile(0.05)))
        x_UCL = x_barbar + (1.88 * (dff[list(dff)[1]].quantile(0.95) - dff[list(dff)[1]].quantile(0.05)))

        x_el = [i for i in range(int(dff[time_col].min()), int(dff[time_col].max()) + 1)]
        fig.add_trace(go.Scatter(x = dff[time_col], y = dff[list(dff)[1]], mode = 'lines+markers', name = "Value"))
        fig.add_trace(go.Scatter(x = x_el, y = [x_UCL for _ in range(len(x_el))], mode = "lines", name = "Upper Bound"))
        fig.add_trace(go.Scatter(x = x_el, y = [x_LCL for _ in range(len(x_el))], mode = "lines", name = "Lower Bound"))
        fig.update_xaxes(showgrid=False)
        fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                           xref='paper', yref='paper', showarrow=False, align='left',
                           bgcolor='rgba(255, 255, 255, 0.5)', text = title)
        fig.update_layout(xaxis_title = time_col, yaxis_title = list(dff)[1])
        fig.update_layout(height = 245, margin = {'l': 20, 'b': 30, 'r': 10, 't': 10})
    return fig

fig_tot = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]])

dff = table[table[multi_time] == multiSlider]
multi_plot = px.scatter(x = dff[multiXax_col], y = dff[multiYax_col], hover_name = dff[multi_index])
multi_plot.update_traces(customdata = dff[multi_index])
multi_plot.update_xaxes(title = multiXax_col)
multi_plot.update_yaxes(title = multiYax_col)
multi_plot.update_layout(clickmode = 'event')

#fig_tot.add_trace(multi_plot, row=1, col=1)
#fig_tot.add_trace(create_time_series(dff, "", multi_index, multi_time), row=1, col=2)
#fig_tot.add_trace(create_time_series(dff, "", multi_index, multi_time), row=2, col=2)

st.plotly_chart(multi_plot, use_container_width=True)
a = multi_plot.data[0].on_click(myFun)
st.write(a)
