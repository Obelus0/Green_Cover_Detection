import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

st.title('Comparison of the environment over 10 years')

col1, col2 = st.beta_columns(2)

selection = col1.radio('',('Groundwater','Pollution','Rainfall'))

df1 = pd.read_excel('environment_data/'+selection.lower()+'_2011.xlsx')
df2 = pd.read_excel('environment_data/'+selection.lower()+'_2016.xlsx')
df3 = pd.read_excel('environment_data/'+selection.lower()+'_2020.xlsx')

district = col2.selectbox('Choose District',sorted(list(df1['DISTRICT'])))

df_list = [df[df['DISTRICT']==district] for df in [df1,df2,df3]]
plot_df = pd.concat(df_list,axis=0)
plot_df.index=['2011','2016','2020']
plot_df=plot_df.T[1:]
# st.table(plot_df)

colors = px.colors.qualitative.Plotly
fig = go.Figure()
fig.add_traces(go.Scatter(x=plot_df.index, y = plot_df['2011'], mode = 'lines+markers', line=dict(color=colors[0],width=3), name='2011'))
fig.add_traces(go.Scatter(x=plot_df.index, y = plot_df['2016'], mode = 'lines+markers', line=dict(color=colors[1],width=3), name='2016'))
fig.add_traces(go.Scatter(x=plot_df.index, y = plot_df['2020'], mode = 'lines+markers', line=dict(color=colors[2],width=3), name='2020'))
fig.update_layout(paper_bgcolor='rgba(50,50,50,0.5)',plot_bgcolor='rgba(0,0,0,0)')
fig.update_layout(hovermode='x unified')

st.plotly_chart(fig,use_container_width=True)