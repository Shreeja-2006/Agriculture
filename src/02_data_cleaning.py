import pandas as pd

# Load datasets
crop = pd.read_csv("data/raw/crop_yield.csv")
weather = pd.read_csv("data/raw/weather.csv")
market = pd.read_csv("data/raw/market_prices.csv")

print("Original shapes:")
print("Crop:", crop.shape)
print("Weather:", weather.shape)
print("Market:", market.shape)


# -------------------------------
# 1. REMOVE DUPLICATES
# -------------------------------

crop = crop.drop_duplicates()
weather = weather.drop_duplicates()
market = market.drop_duplicates()


# -------------------------------
# 2. HANDLE MISSING VALUES
# -------------------------------

# Numeric columns
crop_numeric = crop.select_dtypes(include="number").columns
weather_numeric = weather.select_dtypes(include="number").columns
market_numeric = market.select_dtypes(include="number").columns

crop[crop_numeric] = crop[crop_numeric].fillna(crop[crop_numeric].median())
weather[weather_numeric] = weather[weather_numeric].fillna(
    weather[weather_numeric].median()
)
market[market_numeric] = market[market_numeric].fillna(
    market[market_numeric].median()
)

# Text columns
for column in crop.select_dtypes(include="object").columns:
    crop[column] = crop[column].fillna("Unknown")

for column in weather.select_dtypes(include="object").columns:
    weather[column] = weather[column].fillna("Unknown")

for column in market.select_dtypes(include="object").columns:
    market[column] = market[column].fillna("Unknown")


# -------------------------------
# 3. STANDARDIZE TEXT
# -------------------------------

for df in [crop, weather, market]:
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].str.strip()


# -------------------------------
# 4. CONVERT DATA TYPES
# -------------------------------

crop["Year"] = crop["Year"].astype(int)
weather["Year"] = weather["Year"].astype(int)
weather["Month"] = weather["Month"].astype(int)

market["Date"] = pd.to_datetime(market["Date"])


# -------------------------------
# 5. CHECK FOR INVALID VALUES
# -------------------------------

print("\nNegative values check:")

print("Crop:")
print(crop.select_dtypes(include="number").lt(0).sum())

print("\nWeather:")
print(weather.select_dtypes(include="number").lt(0).sum())

print("\nMarket:")
print(market.select_dtypes(include="number").lt(0).sum())


# -------------------------------
# 6. SAVE CLEAN DATA
# -------------------------------

crop.to_csv("data/processed/crop_clean.csv", index=False)
weather.to_csv("data/processed/weather_clean.csv", index=False)
market.to_csv("data/processed/market_clean.csv", index=False)


# -------------------------------
# 7. FINAL CHECK
# -------------------------------

print("\nCleaning completed!")

print("\nFinal shapes:")
print("Crop:", crop.shape)
print("Weather:", weather.shape)
print("Market:", market.shape)

print("\nMissing values after cleaning:")
print("Crop:", crop.isnull().sum().sum())
print("Weather:", weather.isnull().sum().sum())
print("Market:", market.isnull().sum().sum())