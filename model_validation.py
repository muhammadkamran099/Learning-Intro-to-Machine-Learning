import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = {
    "house_size": [
        800, 900, 1000, 1100, 1200,
        1300, 1400, 1500, 1600, 1700,
        1800, 1900, 2000, 2100, 2200
    ],
    "bedrooms": [
        2, 2, 2, 2, 3,
        3, 3, 3, 3, 4,
        4, 4, 4, 4, 5
    ],
    "price": [
        100000, 115000, 130000, 140000, 155000,
        170000, 185000, 200000, 215000, 230000,
        245000, 260000, 275000, 290000, 310000
    ]
}

df = pd.DataFrame(data)


X = df[["house_size", "bedrooms"]]
y = df["price"]


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=1
)


model = DecisionTreeRegressor(random_state=1)
model.fit(X_train, y_train)


predictions = model.predict(X_valid)


mae = mean_absolute_error(y_valid, predictions)

print("Actual prices:", y_valid.tolist())
print("Predicted prices:", predictions.tolist())
print("Mean Absolute Error:", mae)