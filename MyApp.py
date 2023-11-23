import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


st.title("Stock Price Prediction App")

project_folder = Path(os.getcwd()+"/dataset")

csv_files = [file for file in project_folder.glob("*.csv")]

selected_dataset = st.selectbox("Select the company: ", csv_files, index=0)

if selected_dataset is not None:
    data = pd.read_csv(selected_dataset)

    features = [
        "Open",
        "High",
        "Low",
        "Last",
        "Close",
        "VWAP",
        "Volume",
        "Turnover",
        "Trades",
        "Deliverable Volume",
        "%Deliverble",
    ]
    target = "Close"

    data = data.dropna()

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    st.subheader(f"Mean Squared Error: {mse}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["Date"].iloc[-len(y_test):], y_test.values, label="Actual Closing Rate", marker="o")
    ax.plot(data["Date"].iloc[-len(y_test):], y_pred, label="Predicted Closing Rate", marker="o")
    ax.legend()
    ax.set_title(Path(selected_dataset).stem)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Rate")
    ax.tick_params(rotation=45)
    st.pyplot(fig)

