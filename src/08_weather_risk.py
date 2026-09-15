import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. LOAD WEATHER DATA
# --------------------------------------------------

weather = pd.read_csv("data/processed/weather_clean.csv")

print("Weather dataset shape:", weather.shape)


# --------------------------------------------------
# 2. CREATE WEATHER RISK FEATURES
# --------------------------------------------------

# Calculate yearly weather statistics
weather_yearly = weather.groupby(
    ["State", "District", "Year"]
).agg(
    Annual_Rainfall_mm=("Rainfall_mm", "sum"),
    Avg_Temperature_C=("Temperature_C", "mean"),
    Avg_Humidity_Percent=("Humidity_Percent", "mean"),
    Max_Temperature_C=("Temperature_C", "max"),
    Min_Temperature_C=("Temperature_C", "min")
).reset_index()


# --------------------------------------------------
# 3. DEFINE RISK SCORE
# --------------------------------------------------

def calculate_risk(row):

    risk = 0

    # Rainfall risk
    if row["Annual_Rainfall_mm"] < 500:
        risk += 2
    elif row["Annual_Rainfall_mm"] < 700:
        risk += 1

    if row["Annual_Rainfall_mm"] > 1500:
        risk += 2
    elif row["Annual_Rainfall_mm"] > 1200:
        risk += 1

    # Temperature risk
    if row["Avg_Temperature_C"] > 32:
        risk += 2
    elif row["Avg_Temperature_C"] > 30:
        risk += 1

    # Humidity risk
    if row["Avg_Humidity_Percent"] > 85:
        risk += 1

    return risk


weather_yearly["Risk_Score"] = weather_yearly.apply(
    calculate_risk,
    axis=1
)


# --------------------------------------------------
# 4. CONVERT SCORE TO CATEGORY
# --------------------------------------------------

def risk_category(score):

    if score <= 1:
        return "Low"

    elif score <= 3:
        return "Moderate"

    else:
        return "High"


weather_yearly["Weather_Risk"] = weather_yearly[
    "Risk_Score"
].apply(risk_category)


# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== WEATHER RISK ==========")

print(
    weather_yearly[
        [
            "State",
            "District",
            "Year",
            "Annual_Rainfall_mm",
            "Avg_Temperature_C",
            "Avg_Humidity_Percent",
            "Risk_Score",
            "Weather_Risk"
        ]
    ].head(20)
)


# --------------------------------------------------
# 6. RISK DISTRIBUTION
# --------------------------------------------------

print("\n========== RISK DISTRIBUTION ==========")

print(
    weather_yearly["Weather_Risk"].value_counts()
)


# --------------------------------------------------
# 7. SAVE WEATHER RISK DATA
# --------------------------------------------------

weather_yearly.to_csv(
    "data/processed/weather_risk.csv",
    index=False
)


# --------------------------------------------------
# 8. CREATE RISK GRAPH
# --------------------------------------------------

risk_counts = weather_yearly[
    "Weather_Risk"
].value_counts()

plt.figure(figsize=(8, 6))

risk_counts.plot(kind="bar")

plt.title("Weather Risk Distribution")
plt.xlabel("Weather Risk")
plt.ylabel("Number of Locations")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "data/processed/weather_risk_distribution.png"
)

plt.show()


print("\nWeather risk analysis completed successfully!")