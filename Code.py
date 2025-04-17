import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load and inspect data
print("Loading and preprocessing data...")
df = pd.read_csv("housing.csv")
print("\nFirst 5 rows of data:")
print(df.head())

# Data preprocessing
print("\nHandling missing values...")
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

print("Converting categorical features...")
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

# Feature engineering
print("Creating new features...")
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]

# Prepare data for modeling
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
print("\nTraining and evaluating models...")
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "MSE": mean_squared_error(y_test, y_pred),
        "R²": r2_score(y_test, y_pred)
    }
    print(f"\n{name} Performance:")
    print(f"  MSE: ${results[name]['MSE']:,.2f}")
    print(f"  R² Score: {results[name]['R²']:.4f}")

# Visualization 1: Actual vs Predicted Prices
plt.figure(figsize=(14, 6))

# Linear Regression plot
plt.subplot(1, 2, 1)
lr_pred = models["Linear Regression"].predict(X_test_scaled)
plt.scatter(y_test, lr_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Linear Regression Predictions")
plt.grid(True)

# Random Forest plot
plt.subplot(1, 2, 2)
rf_pred = models["Random Forest"].predict(X_test_scaled)
plt.scatter(y_test, rf_pred, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Random Forest Predictions")
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualization 2: Model Comparison
metrics_df = pd.DataFrame(results).T
metrics_df[[ "MSE"]].plot(kind="bar", figsize=(10, 6), 
                              color=["skyblue", "salmon"],
                              title="Model Performance Comparison (Lower is Better)")
plt.ylabel("Error Value")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Feature Importance
print("\nRandom Forest Feature Importance:")
importances = models["Random Forest"].feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)
print(feature_importance.head(10))