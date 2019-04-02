from sklearn.tree import DecisionTreeRegressor as tree
from sklearn.metrics import mean_absolute_error
from setup import Setup


def mean_error(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = tree(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    return mean_absolute_error(val_y, prediction)


def generate_average_errors(nodes):
    results = {}
    for max_leaf_nodes in nodes:
        results[max_leaf_nodes] = mean_error(max_leaf_nodes, train_X, val_X, train_y, val_y)

    return results


def print_errors(average_errors):
    for key in average_errors:
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (key, average_errors[key]))


path = './resources/data.csv'
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Bedroom2', 'YearBuilt', 'Landsize']
train_X, val_X, train_y, val_y = Setup(path=path, features=features).split_data()
errors = generate_average_errors([5, 50, 500, 5000])
print_errors(errors)
