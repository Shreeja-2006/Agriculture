import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# --------------------------------------------------
# 1. LOAD ML-READY DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/ml_ready_data.csv")

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Yield_Tonnes_Per_Hectare", axis=1)
y = df["Yield_Tonnes_Per_Hectare"]


# --------------------------------------------------
# 3. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. TRAIN RANDOM FOREST
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. EVALUATE MODEL
# --------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# --------------------------------------------------
# 7. ACTUAL VS PREDICTED
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual vs Predicted Crop Yield")

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "data/processed/actual_vs_predicted_yield.png"
)

plt.show()


# --------------------------------------------------
# 8. FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP FEATURES ==========")
print(importance.head(10))


# --------------------------------------------------
# 9. FEATURE IMPORTANCE GRAPH
# --------------------------------------------------

top_features = importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features for Crop Yield Prediction")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png"
)

plt.show()


# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    model,
    "models/yield_prediction_model.pkl"
)

print("\nModel saved successfully!")
print("Location: models/yield_prediction_model.pkl")