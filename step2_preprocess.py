import pandas as pd

# Load the dataset
df = pd.read_csv("weather.csv")

# Convert "Yes"/"No" to 1/0
df["WillRain"] = df["WillRain"].map({"Yes": 1, "No": 0})

# Show the new values
print("✅ Converted 'WillRain' column:")
print(df["WillRain"].head())

# Save to new file
df.to_csv("weather_processed.csv", index=False)
print("\n💾 Saved processed dataset as 'weather_processed.csv'")
