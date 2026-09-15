import pandas as pd
import matplotlib.pyplot as plt

# Load master dataset
df = pd.read_csv("data/processed/agri_master.csv")

print("Dataset shape:", df.shape)

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== STATISTICS ==========")
print(df.describe())

print("\n========== AVERAGE YIELD BY CROP ==========")
crop_yield = df.groupby("Crop")["Yield_Tonnes_Per_Hectare"].mean().sort_values(ascending=False)
print(crop_yield)

print("\n========== AVERAGE YIELD BY STATE ==========")
state_yield = df.groupby("State")["Yield_Tonnes_Per_Hectare"].mean().sort_values(ascending=False)
print(state_yield)

print("\n========== AVERAGE PRICE BY CROP ==========")
crop_price = df.groupby("Crop")["Avg_Modal_Price"].mean().sort_values(ascending=False)
print(crop_price)


# --------------------------------------------------
# 1. AVERAGE YIELD BY CROP
# --------------------------------------------------

plt.figure(figsize=(10, 6))

crop_yield.plot(kind="bar")

plt.title("Average Crop Yield")
plt.xlabel("Crop")
plt.ylabel("Yield (Tonnes per Hectare)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/processed/average_yield_by_crop.png")
plt.show()


# --------------------------------------------------
# 2. AVERAGE YIELD BY STATE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

state_yield.plot(kind="bar")

plt.title("Average Crop Yield by State")
plt.xlabel("State")
plt.ylabel("Yield (Tonnes per Hectare)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/processed/average_yield_by_state.png")
plt.show()


# --------------------------------------------------
# 3. CROP PRICE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

crop_price.plot(kind="bar")

plt.title("Average Market Price by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Modal Price (Rs per Quintal)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("data/processed/average_price_by_crop.png")
plt.show()


# --------------------------------------------------
# 4. YIELD TREND OVER YEARS
# --------------------------------------------------

yearly_yield = df.groupby("Year")["Yield_Tonnes_Per_Hectare"].mean()

plt.figure(figsize=(10, 6))

yearly_yield.plot(kind="line", marker="o")

plt.title("Average Crop Yield Trend")
plt.xlabel("Year")
plt.ylabel("Average Yield (Tonnes per Hectare)")
plt.grid(True)
plt.tight_layout()

plt.savefig("data/processed/yield_trend.png")
plt.show()


# --------------------------------------------------
# 5. RAINFALL VS YIELD
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual_Rainfall_mm"],
    df["Yield_Tonnes_Per_Hectare"],
    alpha=0.6
)

plt.title("Rainfall vs Crop Yield")
plt.xlabel("Annual Rainfall (mm)")
plt.ylabel("Yield (Tonnes per Hectare)")
plt.grid(True)
plt.tight_layout()

plt.savefig("data/processed/rainfall_vs_yield.png")
plt.show()


print("\nEDA completed successfully!")
print("Graphs saved inside data/processed/")