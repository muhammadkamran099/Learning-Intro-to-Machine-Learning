import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000],
    "price": [100000, 130000, 150000, 200000, 250000, 280000]
}

df = pd.DataFrame(data)

X = df[["size"]]
y = df["price"]

model = DecisionTreeRegressor(max_depth=2, random_state=1)
model.fit(X, y)

new_house = pd.DataFrame({"size": [1600]})

prediction = model.predict(new_house)

print("Predicted price:", prediction[0])

plt.figure(figsize=(12, 6))
plot_tree(
    model,
    feature_names=["size"],
    filled=True
)
plt.show()