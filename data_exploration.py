import pandas as pd

data = {
    "house_size": [800, 1000, 1200, 1500, 1800],
    "bedrooms": [2, 2, 3, 3, 4],
    "price": [100000, 130000, 160000, 210000, 260000]
}

df = pd.DataFrame(data)

print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.describe())