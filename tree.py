import pandas as pd
from sklearn.tree import DecisionTreeRegressor as tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


path = './resources/data.csv'
raw_data = pd.read_csv(path)
data = raw_data.dropna(axis=0)  # drop partial rows
print(data.columns)

# data.describe() stats on data
# data.columns available columns
y = data.Price
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Bedroom2', 'YearBuilt', 'Landsize']
X = data[features]

# X.head() first rows
# X.tail() last rows
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
model = tree(random_state=1)
model.fit(train_X, train_y)
predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
