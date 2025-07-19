import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load the processed dataset
df = pd.read_csv("weather_processed.csv")

# Step 2: Split into input (X) and output (y)
X = df[["Temperature", "Humidity", "WindSpeed", "CloudCover"]]
y = df["WillRain"]

# Step 3: Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 5: Check model accuracy (optional but good to know)
accuracy = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Step 6: Save the trained model to a file
joblib.dump(model, "rain_predictor_model.pkl")
print("💾 Model saved as 'rain_predictor_model.pkl'")
