#import package
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

#import the data
data = pd.read_csv("Data Clean.csv")
image = Image.open("house.jpg")
st.title("Aplikasi Prediksi Harga Rumah")
st.image(image, use_column_width=True)

#checking the data
st.write("Aplikasi ini digunakan untuk memprediksi harga rumah dan rentang harga rumah di kawasan Jakarta Selatan!")
check_data = st.checkbox("Lihat contoh data ?")
if check_data:
    st.write(data[1:10])
st.write("Silahkan isi data-data berikut !")

#input the numbers
luas_tanah = st.slider("Berapa luas tanah ?",int(data.LT.min()),int(data.LT.max()),int(data.LT.mean()))
luas_bangunan = st.slider("Berapa Luas Bangunan?",int(data.LB.min()),int(data.LB.max()),int(data.LB.mean()))
jml_kamar_tidur     = st.slider("Berapa Banyak Kamar Tidur?",int(data.JKT.min()),int(data.JKT.max()),int(data.JKT.mean()))
jml_kamar_mandi = st.slider("Berapa jumlah kamar mandi ?",int(data.JKM.min()),int(data.JKM.max()),int(data.JKM.mean()) )
garasi   = st.slider("Apakah rumah anda ada garasi? (0= Ya, 1=Tidak)",int(data.GRS.min()),int(data.GRS.max()),int(data.GRS.mean()) )


from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train, df_test = train_test_split(data, test_size = 0.2, random_state = 45)

y_train = df_train.pop('HARGA')
X_train = df_train

y_test = df_test.pop('HARGA')
X_test = df_test

#modelling step
#Linear Regression model
#import your model
model=LinearRegression()
#fitting and predict your model
model.fit(X_train, y_train)
model.predict(X_test)
errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions = model.predict([[luas_tanah, luas_bangunan, jml_kamar_tidur, jml_kamar_mandi, garasi]])[0]
akurasi= np.sqrt(r2_score(y_test,model.predict(X_test)))


#checking prediction house price
if st.button("Run me!"):
    st.header("Prediksi harga rumah anda adalah Rp {} M".format(round(predictions/1000000000)))
    st.subheader("Rentang harga rumah anda Rp {} M - Rp {} M".format(round((predictions-errors)/1000000000),round((predictions+errors)/1000000000)))
