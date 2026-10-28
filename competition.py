
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Load competition data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 2. Choose features available in both datasets
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt"
]

X = train[features].copy()
y = train["SalePrice"]
X_test = test[features].copy()

# 3. Handle missing values using training-data medians
medians = X.median()
X = X.fillna(medians)
X_test = X_test.fillna(medians)

# 4. Split training data for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 5. Train a model and evaluate it
model = RandomForestRegressor(
    n_estimators=100,
    random_state=1
)

model.fit(X_train, y_train)
valid_predictions = model.predict(X_valid)

mae = mean_absolute_error(y_valid, valid_predictions)
print("Validation MAE:", round(mae, 2))

# 6. Train the final model using all labeled data
model.fit(X, y)

# 7. Predict prices for the test dataset
test_predictions = model.predict(X_test)

# 8. Create the required submission file
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Submission file created: submission.csv")
