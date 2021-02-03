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
multiXax_col = st.sidebar.selectbox("multivariable X axis col", col_mul, 1)
multiYax_col = st.sidebar.selectbox("multivariable Y axis col", col_mul, 2)

fig_tot = make_subplots(rows=2, cols=2, specs=[[{"rowspan": 2}, {}], [None, {}]])

multi_plot = px.scatter(x = table[multiXax_col], y = table[multiYax_col], hover_name = table[multi_index])
multi_plot.update_traces(customdata = table[multi_index])
multi_plot.update_xaxes(title = multiXax_col)
multi_plot.update_yaxes(title = multiYax_col)
multi_plot.update_layout(clickmode = 'event')
