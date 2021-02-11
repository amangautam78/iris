import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the Iris flower type!
""")

st.sidebar.header('User Input Paramters')

def userInputFeatures():
    sepal_length = st.sidebar.slider('sepal length', 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('sepal width', 2.0,4.4,3.4)
    petal_length = st.sidebar.slider('petal length',1.0,6.9,1.3)
    petal_width = st.sidebar.slider('petal width',0.1,2.5,0.2)

    data = {'sepal length':sepal_length,
            'sepal width':sepal_width,
            'petal_length':petal_length,
            'petal_width':petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
df = userInputFeatures()
st.subheader('User Input Parameters')
st.write(df)

data = datasets.load_iris()
X = data.data
Y = data.target

clf = RandomForestClassifier()
clf.fit(X,Y)

prediction = clf.predict(df)
prob = clf.predict_proba(df)

st.subheader('Target label and their corresponding number')
st.write(data.target_names)

st.subheader('Actual representation of iris flowers')
st.image('./iris_pic2.jpg')

st.subheader('Prediction')
st.write(data.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prob)
