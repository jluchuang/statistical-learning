import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error

# Importing the dataset
path_train = "input/train.csv"
data_train = pd.read_csv(path_train)

# Target Matrix
y_train = data_train.SalePrice

# Feature Matrix
model_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X_train = data_train[model_predictors]

path_test = "input/test.csv"

data_test = pd.read_csv(path_test)

X_test = data_test[model_predictors] 

params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

my_submission = pd.DataFrame({"Id": data_test.Id, "SalePrice": y_pred})
my_submission.to_csv("model_submission_GBDT.csv", index=False)

# mse = mean_squared_error(y_test, clf.predict(X_test))
# print("MSE: %.4f" % mse)
