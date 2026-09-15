import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/processed/agri_master.csv")

weather_risk = pd.read_csv(
    "data/processed/weather_risk.csv"
)


# --------------------------------------------------
# 2. MERGE WEATHER RISK
# --------------------------------------------------

df = df.merge(
    weather_risk[
        [
            "State",
            "District",
            "Year",
            "Weather_Risk",
            "Risk_Score"
        ]
    ],
    on=["State", "District", "Year"],
    how="left"
)


# --------------------------------------------------
# 3. NORMALIZATION FUNCTION
# --------------------------------------------------

def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [0.5] * len(series),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


# --------------------------------------------------
# 4. CROP PERFORMANCE
# --------------------------------------------------

crop_summary = df.groupby(
    "Crop"
).agg(
    Average_Yield=(
        "Yield_Tonnes_Per_Hectare",
        "mean"
    ),
    Average_Price=(
        "Avg_Modal_Price",
        "mean"
    ),
    Average_Risk=(
        "Risk_Score",
        "mean"
    )
).reset_index()


# --------------------------------------------------
# 5. NORMALIZE EACH FACTOR
# --------------------------------------------------

crop_summary["Yield_Score"] = normalize(
    crop_summary["Average_Yield"]
)

crop_summary["Price_Score"] = normalize(
    crop_summary["Average_Price"]
)

crop_summary["Risk_Score_Normalized"] = normalize(
    crop_summary["Average_Risk"]
)


# --------------------------------------------------
# 6. CALCULATE RECOMMENDATION SCORE
# --------------------------------------------------

crop_summary["Recommendation_Score"] = (
    crop_summary["Yield_Score"] * 0.45
    +
    crop_summary["Price_Score"] * 0.30
    +
    (1 - crop_summary["Risk_Score_Normalized"]) * 0.25
)


# --------------------------------------------------
# 7. CONVERT TO PERCENTAGE
# --------------------------------------------------

crop_summary["Recommendation_Percentage"] = (
    crop_summary["Recommendation_Score"] * 100
)


# --------------------------------------------------
# 8. RANK CROPS
# --------------------------------------------------

crop_summary = crop_summary.sort_values(
    "Recommendation_Score",
    ascending=False
).reset_index(drop=True)

crop_summary["Rank"] = (
    crop_summary.index + 1
)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\n==========================================")
print("       AGRIINTELLIGENCE")
print("       CROP RECOMMENDATION")
print("==========================================")

print(
    crop_summary[
        [
            "Rank",
            "Crop",
            "Average_Yield",
            "Average_Price",
            "Average_Risk",
            "Recommendation_Percentage"
        ]
    ].round(2).to_string(index=False)
)


# --------------------------------------------------
# 10. BEST CROP
# --------------------------------------------------

best_crop = crop_summary.iloc[0]

print("\n==========================================")
print(
    "RECOMMENDED CROP:",
    best_crop["Crop"]
)

print(
    "Recommendation Score:",
    f"{best_crop['Recommendation_Percentage']:.2f}%"
)

print("==========================================")


# --------------------------------------------------
# 11. SAVE
# --------------------------------------------------

crop_summary.to_csv(
    "data/processed/crop_recommendations.csv",
    index=False
)

print(
    "\nRecommendation results saved successfully!"
)