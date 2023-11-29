import streamlit as st
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.title("Stock Price Analysis App")

project_folder = Path(os.getcwd() + "/dataset")

csv_files = [file for file in project_folder.glob("*.csv")]

selected_dataset = st.selectbox("Select the company: ", csv_files, index=0)

if selected_dataset is not None:
    data = pd.read_csv(selected_dataset)

    features = [
        "Date",
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
        "%Deliverble"
    ]
    target = "Close"

    data = data.dropna()
    data["Date"] = pd.to_datetime(data["Date"])

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train.iloc[:, 1:] = scaler.fit_transform(X_train.iloc[:, 1:])
    X_test.iloc[:, 1:] = scaler.transform(X_test.iloc[:, 1:])

    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X_train.iloc[:, 1:], y_train)

    y_pred = model.predict(X_test.iloc[:, 1:])

    mse = mean_squared_error(y_test, y_pred)
    st.subheader(f"Mean Squared Error: {mse}")

    X_test_sorted = X_test.sort_values(by="Date")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(X_test["Date"], y_test.values, label="Actual Closing Rate", marker="o")
    ax.plot(X_test["Date"], y_pred, label="Predicted Closing Rate", marker="o")
    ax.legend()
    ax.set_title(Path(selected_dataset).stem)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Rate")
    ax.tick_params(rotation=45)
    ax.set_xticklabels([])
    ax.set_xticks([])
    st.pyplot(fig)
