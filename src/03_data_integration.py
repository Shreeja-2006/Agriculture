import pandas as pd

# Load cleaned datasets
crop = pd.read_csv("data/processed/crop_clean.csv")
weather = pd.read_csv("data/processed/weather_clean.csv")
market = pd.read_csv("data/processed/market_clean.csv")

# --------------------------------------------------
# 1. PREPARE WEATHER DATA
# --------------------------------------------------

# Convert monthly weather data into yearly statistics
weather_yearly = weather.groupby(
    ["State", "District", "Year"]
).agg(
    Annual_Rainfall_mm=("Rainfall_mm", "sum"),
    Avg_Temperature_C=("Temperature_C", "mean"),
    Avg_Humidity_Percent=("Humidity_Percent", "mean")
).reset_index()

print("Weather yearly data:", weather_yearly.shape)


# --------------------------------------------------
# 2. PREPARE MARKET DATA
# --------------------------------------------------

# Convert date to year
market["Date"] = pd.to_datetime(market["Date"])
market["Year"] = market["Date"].dt.year

# Calculate yearly average market price
market_yearly = market.groupby(
    ["State", "District", "Crop", "Year"]
).agg(
    Avg_Min_Price=("Min_Price_Rs_per_Quintal", "mean"),
    Avg_Max_Price=("Max_Price_Rs_per_Quintal", "mean"),
    Avg_Modal_Price=("Modal_Price_Rs_per_Quintal", "mean")
).reset_index()

print("Market yearly data:", market_yearly.shape)


# --------------------------------------------------
# 3. MERGE CROP + WEATHER
# --------------------------------------------------

master = crop.merge(
    weather_yearly,
    on=["State", "District", "Year"],
    how="left"
)

print("After crop + weather:", master.shape)


# --------------------------------------------------
# 4. MERGE WITH MARKET
# --------------------------------------------------

master = master.merge(
    market_yearly,
    on=["State", "District", "Crop", "Year"],
    how="left"
)

print("After market merge:", master.shape)


# --------------------------------------------------
# 5. CHECK FINAL DATASET
# --------------------------------------------------

print("\n========== FINAL DATASET ==========")

print("Shape:", master.shape)

print("\nColumns:")
print(master.columns.tolist())

print("\nFirst 5 rows:")
print(master.head())

print("\nMissing values:")
print(master.isnull().sum())


# --------------------------------------------------
# 6. SAVE MASTER DATASET
# --------------------------------------------------

master.to_csv(
    "data/processed/agri_master.csv",
    index=False
)

print("\nAgriIntelligence master dataset created successfully!")