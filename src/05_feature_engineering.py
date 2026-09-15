import pandas as pd

# Load integrated dataset
df = pd.read_csv("data/processed/agri_master.csv")

print("Original shape:", df.shape)

# Select features
features = [
    "State",
    "District",
    "Crop",
    "Year",
    "Area_Hectares",
    "Annual_Rainfall_mm",
    "Avg_Temperature_C",
    "Avg_Humidity_Percent",
    "Avg_Modal_Price"
]

target = "Yield_Tonnes_Per_Hectare"

data = df[features + [target]].copy()

# Create additional features
data["Rainfall_per_Hectare"] = (
    data["Annual_Rainfall_mm"] / data["Area_Hectares"]
)

data["Price_Yield_Ratio"] = (
    data["Avg_Modal_Price"] / data["Yield_Tonnes_Per_Hectare"]
)

data["Years_From_2020"] = data["Year"] - 2020

# Handle missing values
numeric_columns = data.select_dtypes(include="number").columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].median()
)

# Encode categorical features
data_encoded = pd.get_dummies(
    data,
    columns=["State", "District", "Crop"],
    drop_first=True
)

# Save ML-ready dataset
data_encoded.to_csv(
    "data/processed/ml_ready_data.csv",
    index=False
)

print("\n========== FEATURE ENGINEERING ==========")
print("ML-ready shape:", data_encoded.shape)

print("\nML-ready columns:")
print(data_encoded.columns.tolist())

print("\nFirst 5 rows:")
print(data_encoded.head())

print("\nFeature engineering completed successfully!")