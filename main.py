import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv(sys.argv[1])

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor(objective="reg:squarederror")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

plt.figure(figsize=(10, 6))
plt.plot(
    data["Date"].iloc[-len(y_test) :],
    y_test.values,
    label="Actual Closing Rate",
    marker="o",
)
plt.plot(
    data["Date"].iloc[-len(y_test) :],
    y_pred,
    label="Predicted Closing Rate",
    marker="o",
)
plt.legend()
plt.title(Path(sys.argv[1]).stem)
plt.xlabel("Date")
plt.ylabel("Closing Rate")
plt.xticks(rotation=45)
plt.show()
