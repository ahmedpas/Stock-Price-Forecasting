import pandas as pd
import streamlit as st
import numpy as np
from pymongo import MongoClient

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pandas_datareader as pdr
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go

key = "d1e0bf0b26e537200ebc6fce031449455a3f44e9"

ticker_symbols = []
nasdaq = pd.read_csv("nasdaq.csv")
ticker_symbols = nasdaq['Symbol']

st.set_page_config(
    page_title="Stocks Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: centre; color: #39FF14;'>STOCK PRICE PREDICTOR📈</h1>",
            unsafe_allow_html=True)

# Session state to store login status
if 'login_state' not in st.session_state:
    st.session_state.login_state = False

with st.sidebar:
    ml = st.selectbox('Select from below', ['Signup', 'Login', 'Forecast'])

if ml == 'Signup':
    st.title("Sign Up")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    conf = st.text_input("Confirm Password", type="password", key="login-password")
    email = st.text_input("Email")

    if password != conf:
        st.error("Passwords do not match")

    if st.button("Sign Up"):
        with st.spinner('Signing you up...'):
            try:
                client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
                database = client.Imstock
                collection = database.userdata
                user_to_ins = {'Username': username, 'Password': password, 'Email': email}
                collection.insert_one(user_to_ins)
                st.success('Signed up successfully')
            except Exception:
                st.error('😲 An unfortunate error occurred...check your connection')

elif ml == 'Login':
    st.title("Log In")
    usernamel = st.text_input("Username")
    passwordl = st.text_input("Password", type="password")

    if st.button('Log In'):
        with st.spinner('Connecting to database...'):
            try:
                client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
                database = client.Imstock
                collection = database.userdata
                if collection.find_one({'Username': usernamel, 'Password': passwordl}):
                    st.session_state.login_state = True
                    st.success('Successfully Logged In')
                else:
                    st.error('Invalid Username or Password')
            except Exception:
                st.error('😲 An unfortunate error occurred...check your connection')

elif ml == 'Forecast':
    if not st.session_state.login_state:
        st.error("Please login first to access the forecast feature.")
    else:
        col1, col2 = st.columns(2)
        col3, col4, col5, col6, col7 = st.columns(5)

        with col1:
            st.markdown("<h3 style='text-align: centre; color: #BF3EFF;'>Select Ticker Symbol</h3>", unsafe_allow_html=True)
            ticker_symbol = st.selectbox("Select Ticker Symbol", ticker_symbols)

        with col2:
            st.markdown("<h3 style='text-align: centre; color: #BF3EFF;'>No of Days to Forecast</h3>", unsafe_allow_html=True)
            forecast_days = st.slider("", min_value=1, max_value=30)

        with st.spinner('Connecting to database...'):
            try:
                client = MongoClient('mongodb+srv://Atif:XHMswoIrHKVzfIjo@stockcluster.m5bop17.mongodb.net/') 
                database = client.Imstock
                collection = database.stocks
            except Exception:
                st.error('😲 An unfortunate error occurred...check your connection')

        with col5:
            st.markdown("""
                <style>
                .stButton button {
                    background-color: black;
                    color: white;
                    font-size: 16px;
                    padding: 10px 20px;
                    border-radius: 5px;
                    border: none;
                }
                </style>
                """, unsafe_allow_html=True)
            predict_button = st.button("Forecast")

        if predict_button:
            query = {"tickerSym": ticker_symbol}
            results = collection.find(query)
            data = list(results)

            if not data:
                data = pdr.get_data_tiingo(ticker_symbol, api_key=key)
                data = data.reset_index()
                last_date = data.iloc[-1]['date']
                data = data.drop(['symbol', 'date', 'divCash', 'splitFactor'], axis=1)

                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)

                tx, ty = [], []
                n_past = 100
                for i in range(n_past, len(scaled_data)):
                    tx.append(scaled_data[i-n_past:i, :])
                    ty.append(scaled_data[i, 1])

                tx = np.array(tx)
                ty = np.array(ty)

                model = Sequential()
                model.add(LSTM(64, activation='relu', input_shape=(tx.shape[1], tx.shape[2]), return_sequences=True))
                model.add(LSTM(32, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')

                with st.spinner('Model is predicting... May take 2-3 minutes.'):
                    model.fit(tx, ty, epochs=10, batch_size=65, validation_split=0.1, verbose=1)

                sc = scaler.transform(data[-100:])
                sc = np.reshape(sc, (1, 100, data.shape[1]))
                predictions = []
                for _ in range(30):
                    prediction = model.predict(sc)[0][0]
                    predictions.append(prediction)
                    sc = np.roll(sc, -1, axis=1)
                    sc[0, -1, :] = prediction

                prediction_array = np.array(predictions).reshape(-1, 1)
                prediction_copies = np.repeat(prediction_array, data.shape[1], axis=-1)
                preds = scaler.inverse_transform(prediction_copies)[:, 0].tolist()

                collection.insert_one({"tickerSym": ticker_symbol, "last_date": last_date, "last_value": data.iloc[-1][1], "forecast": preds})
            else:
                document = collection.find_one({"tickerSym": ticker_symbol})
                last_date = document['last_date']
                preds = document['forecast']
                last_value = document['last_value']

            now = datetime(last_date.year, last_date.month, last_date.day)
            future_dates = [now + timedelta(days=i) for i in range(forecast_days)]

            to_plot = pd.DataFrame({
                'DATES': [d.date() for d in future_dates],
                'FUTURE PREDICTION OF HIGHEST STOCK PRICE': preds[:forecast_days]
            })

            col11, col12 = st.columns(2)
            with col11:
                st.plotly_chart(px.line(to_plot, x='DATES', y='FUTURE PREDICTION OF HIGHEST STOCK PRICE', title='PREDICTION FOR HIGH PRICE'))
            with col12:
                st.plotly_chart(px.area(to_plot, x='DATES', y='FUTURE PREDICTION OF HIGHEST STOCK PRICE'))

            st.plotly_chart(go.Figure(go.Indicator(
                mode="number+delta",
                value=preds[0],
                delta={'position': "top", 'reference': last_value},
                title={'text': 'HIGH VALUE FROM THE LAST DAY'}
            )))

            st.dataframe(to_plot)
            st.write(f'[LINK TO YOUR STOCK](https://finance.yahoo.com/quote/{ticker_symbol})')
