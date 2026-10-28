
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Create sample house data
data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000,
             900, 1100, 1300, 1600, 1900, 2200,
             850, 1050, 1250, 1550, 1850, 2100,
             950, 1400],
    "bedrooms": [2, 2, 3, 3, 4, 4, 2, 3, 3, 3,
                 4, 5, 2, 3, 3, 4, 4, 5, 2, 3],
    "price": [40, 50, 60, 75, 90, 100,
              45, 55, 65, 80, 95, 110,
              42, 52, 62, 78, 92, 105,
              47, 70]
}

df = pd.DataFrame(data)

# 2. Select features and target
X = df[["size", "bedrooms"]]
y = df["price"]

# 3. Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# 4. Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=1
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions
predictions = model.predict(X_valid)

# 7. Evaluate the model
mae = mean_absolute_error(y_valid, predictions)

print("Actual prices:", y_valid.tolist())
print("Predicted prices:", predictions.round(2).tolist())
print("Mean Absolute Error:", round(mae, 2))

# 8. Predict a new house price
new_house = pd.DataFrame({
    "size": [1400],
    "bedrooms": [3]
})

predicted_price = model.predict(new_house)

print("Predicted price for new house:", round(predicted_price[0], 2))
