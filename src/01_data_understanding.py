import pandas as pd

# Load datasets
crop = pd.read_csv("data/raw/crop_yield.csv")
weather = pd.read_csv("data/raw/weather.csv")
market = pd.read_csv("data/raw/market_prices.csv")

# -------------------------------
# 1. BASIC INFORMATION
# -------------------------------

print("\n========== CROP DATA ==========")
print("Shape:", crop.shape)
print("\nColumns:")
print(crop.columns.tolist())
print("\nFirst 5 rows:")
print(crop.head())

print("\n========== WEATHER DATA ==========")
print("Shape:", weather.shape)
print("\nColumns:")
print(weather.columns.tolist())
print("\nFirst 5 rows:")
print(weather.head())

print("\n========== MARKET DATA ==========")
print("Shape:", market.shape)
print("\nColumns:")
print(market.columns.tolist())
print("\nFirst 5 rows:")
print(market.head())


# -------------------------------
# 2. DATA TYPES
# -------------------------------

print("\n========== DATA TYPES ==========")

print("\nCrop:")
print(crop.dtypes)

print("\nWeather:")
print(weather.dtypes)

print("\nMarket:")
print(market.dtypes)


# -------------------------------
# 3. MISSING VALUES
# -------------------------------

print("\n========== MISSING VALUES ==========")

print("\nCrop:")
print(crop.isnull().sum())

print("\nWeather:")
print(weather.isnull().sum())

print("\nMarket:")
print(market.isnull().sum())


# -------------------------------
# 4. DUPLICATES
# -------------------------------

print("\n========== DUPLICATES ==========")

print("Crop duplicates:", crop.duplicated().sum())
print("Weather duplicates:", weather.duplicated().sum())
print("Market duplicates:", market.duplicated().sum())


# -------------------------------
# 5. UNIQUE VALUES
# -------------------------------

print("\n========== UNIQUE VALUES ==========")

print("\nCrops:")
print(crop["Crop"].unique())

print("\nStates:")
print(crop["State"].unique())

print("\nDistricts:")
print(crop["District"].unique())


# -------------------------------
# 6. YEAR RANGE
# -------------------------------

print("\n========== YEAR RANGE ==========")

print("Crop years:", crop["Year"].min(), "-", crop["Year"].max())
print("Weather years:", weather["Year"].min(), "-", weather["Year"].max())

market["Date"] = pd.to_datetime(market["Date"])
print("Market dates:", market["Date"].min(), "-", market["Date"].max())


# -------------------------------
# 7. STATISTICAL SUMMARY
# -------------------------------

print("\n========== CROP STATISTICS ==========")
print(crop.describe())

print("\n========== WEATHER STATISTICS ==========")
print(weather.describe())

print("\n========== MARKET STATISTICS ==========")
print(market.describe())