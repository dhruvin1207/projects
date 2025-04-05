import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

from sklearn.preprocessing import LabelEncoder,StandardScaler
le = LabelEncoder()
sc= StandardScaler()


model = pickle.load(open('churn.pkl','rb'))


st.title("Churn Prediction Web App")
gender = st.selectbox('Select Gender:',options=['Male','Female'])
SeniorCitizen = st.selectbox(' You Are SeniorCitizen ?',options=['Yes','No'])
Partner = st.selectbox(' Do you have a partner ?',options=['Yes','No'])
Dependents = st.selectbox('Are you Dependents on others ?',options=['Yes','No'])
tenure = st.text_input('Enter your tenure')
PhoneService= st.selectbox('Do you have PhoneService ?',options=['Yes','No'])
MultipleLines = st.selectbox('Do you have MultipleLines Services ?',options=['Yes','No','No one'])
Contract= st.selectbox('Your Contract ',options=['one year','Two year','Month-to-month'])
TotalCharges = st.text_input('Enter your total charges')



def  predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges):
  data={"gender":[gender],
       "SeniorCitizen":[SeniorCitizen],
       "Partner" :[Partner],
        "Dependents" :[Dependents],
          "tenure" :[tenure],
"PhoneService" :[PhoneService],
"MultipleLines" :[MultipleLines],
"Contract" : [Contract],
"TotalCharges" :[TotalCharges]}
  df1 = pd.DataFrame(data)

  co = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','Contract','TotalCharges']
  for col in co:
    df1[col] = le.fit_transform(df1[col])
  
  df1 = sc.fit_transform(df1)
  result = model.predict(df1).reshape(1,-1)
  return result




if st.button("predict"):
    results =predictive(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,Contract,TotalCharges)
    if results==0:
           st.write("Not Churn")
    else:
       st.write("Churn")






