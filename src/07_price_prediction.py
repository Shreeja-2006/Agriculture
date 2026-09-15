import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# --------------------------------------------------
# 1. LOAD MARKET DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/market_clean.csv")

df["Date"] = pd.to_datetime(df["Date"])

# Create useful date features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month


# --------------------------------------------------
# 2. CREATE MONTHLY/YEARLY FEATURES
# --------------------------------------------------

# Sort before creating previous-price feature
df = df.sort_values(
    ["State", "District", "Crop", "Date"]
)

# Previous month's modal price
df["Previous_Modal_Price"] = df.groupby(
    ["State", "District", "Crop"]
)["Modal_Price_Rs_per_Quintal"].shift(1)

# Remove first record of each crop/location
df = df.dropna(subset=["Previous_Modal_Price"])


# --------------------------------------------------
# 3. SELECT FEATURES
# --------------------------------------------------

features = [
    "Year",
    "Month",
    "Previous_Modal_Price"
]

target = "Modal_Price_Rs_per_Quintal"

X = df[features]
y = df[target]


# --------------------------------------------------
# 4. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 5. TRAIN MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== PRICE MODEL PERFORMANCE ==========")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))


# --------------------------------------------------
# 8. ACTUAL VS PREDICTED PRICE
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Market Price")
plt.ylabel("Predicted Market Price")
plt.title("Actual vs Predicted Market Price")

plt.tight_layout()

plt.savefig(
    "data/processed/actual_vs_predicted_price.png"
)

plt.show()


# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n========== PRICE MODEL FEATURES ==========")
print(importance)


# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "models/price_prediction_model.pkl"
)

print("\nPrice prediction model saved successfully!")
print("Location: models/price_prediction_model.pkl")