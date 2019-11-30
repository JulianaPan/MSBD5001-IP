import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error

def RMSE(y_target, y_pred):
    rms = sqrt(mean_squared_error(y_target, y_pred))
    return rms

df_train = pd.read_csv('data/new_train.csv')
X = df_train.iloc[:,2:]
y = df_train.loc[:,'playtime_forever']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


tpot = TPOTRegressor(generations=20, verbosity=2, cv=10, scoring='neg_mean_squared_error', random_state=1) #迭代20次
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot20_exported_pipeline.py')

