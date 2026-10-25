import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Create sample data
data = {
    "house_size": [
        700, 800, 900, 1000, 1100,
        1200, 1300, 1400, 1500, 1600,
        1700, 1800, 1900, 2000, 2100,
        2200, 2300, 2400, 2500, 2600
    ],
    "price": [
        90000, 100000, 115000, 125000, 140000,
        150000, 165000, 175000, 190000, 205000,
        215000, 230000, 245000, 255000, 270000,
        285000, 295000, 310000, 325000, 340000
    ]
}

df = pd.DataFrame(data)

X = df[["house_size"]]
y = df["price"]

# 2. Split the data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# 3. Compare different tree depths
for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeRegressor(
        max_depth=depth,
        random_state=1
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    valid_predictions = model.predict(X_valid)

    train_mae = mean_absolute_error(
        y_train, train_predictions
    )
    valid_mae = mean_absolute_error(
        y_valid, valid_predictions
    )

    print(f"Depth: {depth}")
    print(f"Training MAE: {train_mae:.2f}")
    print(f"Validation MAE: {valid_mae:.2f}")
    print("-" * 30)