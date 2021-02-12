import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pickle

st.write("""
# Simple Iris Flower Prediction App

This app predicts the Iris flower type!
""")

st.sidebar.markdown("""
[Example CSV input file](https://github.com/amangautam78/iris/blob/main/Sample_dataset.csv)
""")
st.sidebar.header('User Input Paramters')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["CSV"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    def userInputFeatures():
        sepal_length = st.sidebar.slider('Sepal length', 4.9,7.9,5.4)
        sepal_width = st.sidebar.slider('Sepal width',2.0,4.4,3.1)
        petal_length = st.sidebar.slider('Petal length',1.0,6.9,5.5)
        petal_width = st.sidebar.slider('Petal width',0.1,2.5,2.2)
        data = {'sepal length':sepal_length,
                'sepal width':sepal_width,
                'petal length':petal_length,
                'petal width':petal_width}
        features = pd.DataFrame(data, index = [0])
        return features
    df = userInputFeatures()
st.subheader('User Input Parameters')
st.write(df)
data = datasets.load_iris()
X = data.data
Y = data.target

clf = pickle.load(open('iris_clf.pkl','rb'))

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
