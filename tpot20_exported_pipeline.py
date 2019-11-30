import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from sklearn.externals import joblib        # save model
from math import sqrt
from sklearn.metrics import mean_squared_error

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('data/new_train.csv')
features = tpot_data.drop('id', axis=1).drop('playtime_forever', axis=1)
training_features, testing_features, training_target, testing_target = train_test_split(features, tpot_data['playtime_forever'], test_size=0.33, random_state=3)

# Average CV score on the training set was: -123.75787025138598
exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.01),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.001, loss="lad", max_depth=2, max_features=0.7000000000000001, min_samples_leaf=9, min_samples_split=10, n_estimators=100, subsample=0.6500000000000001)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="epsilon_insensitive", penalty="elasticnet", power_t=0.0)),
    MinMaxScaler(),
    RandomForestRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=10, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
joblib.dump(exported_pipeline, "model/Model2.m")
def RMSE(y_target, y_pred):
    rms = sqrt(mean_squared_error(y_target, y_pred))
    return rms

print(RMSE(testing_target,results))
