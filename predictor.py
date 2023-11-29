import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

st.title("Stock Price Analysis App")

project_folder = Path(os.getcwd() + "/dataset")

csv_files = [file for file in project_folder.glob("*.csv")]

selected_dataset = st.selectbox("Select the company: ", csv_files, index=0)

if selected_dataset is not None:
    data = pd.read_csv(selected_dataset)

    features = [
        "Open",
        "High",
        "Low",
        "Last",
        "VWAP",
        "Volume",
        "Turnover",
        "Trades",
        "Deliverable Volume",
        "%Deliverble",
    ]
    target = "Close"

    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train_scaled, y_train)

    st.subheader("Input Parameters for Prediction")

    open_price = st.number_input(
        "Open Price",
        min_value=float(X["Open"].min()),
        max_value=float(X["Open"].max()),
        value=float(X["Open"].mean()),
    )
    high_price = st.number_input(
        "High Price",
        min_value=float(X["High"].min()),
        max_value=float(X["High"].max()),
        value=float(X["High"].mean()),
    )
    low_price = st.number_input(
        "Low Price",
        min_value=float(X["Low"].min()),
        max_value=float(X["Low"].max()),
        value=float(X["Low"].mean()),
    )
    last_price = st.number_input(
        "Last Price",
        min_value=float(X["Last"].min()),
        max_value=float(X["Last"].max()),
        value=float(X["Last"].mean()),
    )
    vwap_price = st.number_input(
        "VWAP Price",
        min_value=float(X["VWAP"].min()),
        max_value=float(X["VWAP"].max()),
        value=float(X["VWAP"].mean()),
    )
    volume = st.number_input(
        "Volume",
        min_value=float(X["Volume"].min()),
        max_value=float(X["Volume"].max()),
        value=float(X["Volume"].mean()),
    )
    turnover = st.number_input(
        "Turnover",
        min_value=float(X["Turnover"].min()),
        max_value=float(X["Turnover"].max()),
        value=float(X["Turnover"].mean()),
    )
    trades = st.number_input(
        "Trades",
        min_value=float(X["Trades"].min()),
        max_value=float(X["Trades"].max()),
        value=float(X["Trades"].mean()),
    )
    deliverable_volume = st.number_input(
        "Deliverable Volume",
        min_value=float(X["Deliverable Volume"].min()),
        max_value=float(X["Deliverable Volume"].max()),
        value=float(X["Deliverable Volume"].mean()),
    )
    percent_deliverable = st.number_input(
        "% Deliverable",
        min_value=float(X["%Deliverble"].min()),
        max_value=float(X["%Deliverble"].max()),
        value=float(X["%Deliverble"].mean()),
    )

    user_input = pd.DataFrame(
        {
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Last": [last_price],
            "VWAP": [vwap_price],
            "Volume": [volume],
            "Turnover": [turnover],
            "Trades": [trades],
            "Deliverable Volume": [deliverable_volume],
            "%Deliverble": [percent_deliverable],
        }
    )

    scaled_user_input = scaler.transform(user_input)
    user_prediction = model.predict(scaled_user_input)
    st.subheader("Prediction Result")
    st.write(f"Predicted Closing Rate: {user_prediction[0]}")
