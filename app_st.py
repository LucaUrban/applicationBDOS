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

st.plotly_chart(multi_plot, use_container_width=True)

# pareto chart with feature importance on huber regressor
st.header("Feature Importance Analysis")

fea_Imp_features = st.multiselect("Feature Importance multiselectin box:", col_mul)
scaler = StandardScaler(); train_nm = scaler.fit_transform(table[fea_Imp_features])
Alpha = [.1, 1, 10, 100]; titles = tuple("Feature importance for alpha = " + str(alpha) for alpha in Alpha)
Alpha = [[.1, 1], [10, 100]]

# Create figure with secondary y-axis
fig_tot = make_subplots(rows = 2, cols = 2, 
                        specs = [[{"secondary_y": True}, {"secondary_y": True}], 
                                 [{"secondary_y": True}, {"secondary_y": True}]], 
                        subplot_titles = titles)

for num_row in range(2):
    for num_col in range(2):
        clf = Ridge(alpha = Alpha[num_row][num_col])
        clf.fit(train_nm, table["Expected"])

        importance = clf.coef_
        for i in range(len(importance)):
            if importance[i] < 0:
                importance[i] *= -1
        dict_fin = {list(train)[i]: importance[i] for i in range(len(importance))}
        dict_fin = {k: v for k, v in sorted(dict_fin.items(), key=lambda item: item[1], reverse = True)}
        dict_fin_per = {list(train)[i]: (importance[i]/sum(importance))*100 for i in range(len(importance))}
        dict_fin_per = {k: v for k, v in sorted(dict_fin_per.items(), key=lambda item: item[1], reverse = True)}
        lis_final = []; res_par = 0
        for value in dict_fin_per.values():
            res_par += value; lis_final.append(res_par)

        fig_tot.add_trace(
            go.Bar(x = list(dict_fin_per.keys()), y = list(dict_fin_per.values()), 
                   marker_color = 'rgb(158,202,225)', marker_line_color = 'rgb(8,48,107)', 
                   marker_line_width = 1.5, opacity = 0.6, name = 'Value'),
            row = num_row + 1, col = num_col + 1, secondary_y = False
        )

        fig_tot.add_trace(
            go.Scatter(x = list(dict_fin_per.keys()), y = lis_final, line_color = 'rgb(255, 150, 0)'),
            row = num_row + 1, col = num_col + 1, secondary_y = True
        )

        # Add figure title
        fig_tot.update_layout(
            title_text = "Feature importances", showlegend = False
        )

        # Set x-axis title
        fig_tot.update_xaxes(title_text = "Variables")

        # Set y-axes titles
        fig_tot.update_yaxes(title_text="<b>Value</b> of importance", secondary_y=False)
        fig_tot.update_yaxes(title_text="<b>%</b> of importance", secondary_y=True)

fig_tot.update_layout(height = 600)
st.plotly_chart(fig_tot, use_container_width=True)
