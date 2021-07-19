import streamlit as st
import pandas as pd

duration = st.selectbox('Select Duration:',('jan20-jul20','jul20-dec20','jan21-jul21'))
feature = st.selectbox('Select Feature:',('PM2.5','PM10','NO','NO2','NOx','NH3','SO2','CO','Ozone'))#,'Benzene','RH','WS','WD','SR','BP','VWS')
df = pd.read_excel('pollution_data/'+duration+'.xlsx')
df.dropna()
st.line_chart(df[feature],width=500,height=350)