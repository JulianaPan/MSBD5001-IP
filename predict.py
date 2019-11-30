import pandas as pd
from sklearn.externals import joblib

# load model
model = joblib.load("model/Model2.m")

# Load data
df_test = pd.read_csv('data/new_test.csv')

X_test = df_test.loc[:,df_test.columns!='id']
y_pre = model.predict(X_test)

# save data
df_result = pd.DataFrame()
df_result['id'] = df_test.loc[:,'id']
df_result['playtime_forever'] = y_pre
df_result.to_csv("result/result2.csv",index=False)

