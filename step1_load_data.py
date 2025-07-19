import pandas as pd

# Load the dataset
df = pd.read_csv("weather.csv")

# Show the first 5 rows
print("📊 Preview of dataset:")
print(df.head())

# Show basic info
print("\n🧾 Column names and types:")
print(df.info())

# Show class balance
print("\n📈 Rain value counts:")
print(df["WillRain"].value_counts())
