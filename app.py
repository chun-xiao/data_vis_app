"""
Created on Fri Aug 13 21:48:35 2021

@author: Chun
"""
import streamlit as st

import pandas as pd



st.title("Chun's Data Visualization app")


upload_file = st.file_uploader('', type="csv", accept_multiple_files=False)

if upload_file:
    file_df = pd.read_csv(upload_file)
    st.write(file_df.head())
    st.write("Success")