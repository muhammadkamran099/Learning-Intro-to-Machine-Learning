import pandas as pd
from sklearn.tree import DecisionTreeRegressor

data = {
    "size": [1200, 1400, 1600, 1800, 2000, 2200],
    "price": [180000, 200000, 220000, 240000, 260000, 280000]
}

df = pd.DataFrame(data)

x = df[["size"]]
y = df[["price"]]

model = DecisionTreeRegressor(random_state=1)

model.fit(x, y)

new_house = pd.DataFrame({"size": [1600]})

prediction = model.predict(new_house)

print("Predicted Price: ", prediction[0])