import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets

st.set_page_config(page_title='Simple Iris Classification', page_icon="./f.png")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
  sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
  sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
  petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
  petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
  data = {'sepal_length': sepal_length,
          'sepal_width': sepal_width,
          'petal_length': petal_length,
          'petal_width': petal_width}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()

clf = pickle.load(open('iris_clf.pkl', 'rb'))
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# This app repository

st.write("""
---
## App repository

[Github](https://github.com/ftarantuviez/Classification-Iris)
""")
# / This app repository