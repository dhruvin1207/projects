import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

model = pickle.load(open('model.pkl','rb'))
st.title("Prediction Model")
tv= st.text_input('Enter a TV Sales')
radio= st.text_input('Enter a Radio Sales')
newspaper= st.text_input('Enter a newspaper Sales')

if st.button("predict"):
    features = np.array([[tv,radio,newspaper]],dtype=np.float64)
    result = model.predict(features).reshape(1,-1)
    st.write('sales prediction is ::',result[0])