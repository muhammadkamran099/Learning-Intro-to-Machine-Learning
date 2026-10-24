import pandas as pd
from sklearn.tree import DecisionTreeRegressor


data = {
    "house_size": [800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [2, 2, 3, 3, 4, 4],
    "price": [100000, 130000, 160000, 210000, 250000, 280000]
}

df = pd.DataFrame(data)


features = ["house_size", "bedrooms"]

X = df[features]
y = df["price"]


model = DecisionTreeRegressor(random_state=1)


model.fit(X, y)


new_house = pd.DataFrame({
    "house_size": [1600],
    "bedrooms": [3]
})

prediction = model.predict(new_house)

print("Predicted price:", prediction[0])