from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from setup import Setup


path = './resources/data.csv'
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Bedroom2', 'YearBuilt', 'Landsize']
train_X, val_X, train_y, val_y = Setup(path=path, features=features).split_data()

forest_model = RandomForestRegressor(random_state=0)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
