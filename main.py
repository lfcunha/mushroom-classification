from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from sklearn import metrics
import pickle
import pydot
from sklearn.externals.six import StringIO
import operator


# y = []
# x = []
#
# with open("data/data.csv", "r") as fh:
#     lines = fh.readlines()
#     for line in lines:
#         l = line.strip().split(",")
#         y.append(l[0])
#         x.append(l[1:])
#
#
# print(y)
# print(x)
# le = preprocessing.LabelEncoder()

# read data from remote file
data_url = 'https://s3.amazonaws.com/lfcunha-files/data.csv'
data = pd.read_csv(data_url)

#data = pd.read_csv('data/data.csv')
edible = data['edible']
features = data.drop('edible', axis=1)

# label encode categorial data
#le = LabelEncoder()
#data_encoded = data.apply(le.fit_transform)

encoder = defaultdict(LabelEncoder)
data_encoded = data.apply(lambda x: encoder[x.name].fit_transform(x))


## Inverse the encoded
#data_encoded.apply(lambda x: encoder[x.name].inverse_transform(x))

## Using the dictionary to label future data
#df.apply(lambda x: encoder[x.name].transform(x))


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data_encoded.iloc[:, 1:], data_encoded.iloc[:,0], test_size=0.20, random_state=1)


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # Calculate the performance score between 'y_true' and 'y_predict'
    return r2_score(y_true, y_predict)


def fit_model(X, y):
    rs = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    cv_sets = rs.get_n_splits(X)
    classifier = DecisionTreeClassifier(random_state=0)
    params = {"max_depth": range(1, 11)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(classifier, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    print(pd.DataFrame(grid.cv_results_))
    return grid.best_estimator_

reg = fit_model(X_train, y_train)
reg.fit(X_train, y_train)

Z = reg.predict(X_test)
s = pickle.dumps(reg)

print(metrics.confusion_matrix(y_test, Z))
print(metrics.classification_report(y_test, Z))

dot_data = StringIO()

export_graphviz(reg, out_file="dot.dot", feature_names=list(data)[1:], class_names=["edible", "poisonous"])
#export_graphviz(reg, out_file=dot_data, feature_names=list(data)[1:])
# graph_ = pydot.graph_from_dot_data(dot_data.getvalue())
# graph_.write_pdf("tree.pdf")

feature_importances = reg.feature_importances_

fi = dict(zip(feature_importances, list(data)[1:]))
fi_S = sorted(fi.items(), key=operator.itemgetter(0), reverse=True)

print(fi_S)
