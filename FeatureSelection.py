import pandas as pd # used for handling the dataset
from sklearn.feature_selection import VarianceThreshold  #移除低方差特征
from sklearn.model_selection import train_test_split



df_train = pd.read_csv('data/new_train.csv')
df_test = pd.read_csv('data/new_test.csv')


X = df_train.iloc[:,2:]       # independent variable set
y = df_train.loc[:,'playtime_forever']    # dependent variable set

X_test = df_test.loc[:,df_test.columns!='id']

sel_train = VarianceThreshold(threshold=0.2)
X_transformed=sel_train.fit_transform(X)
#print('去除低方差特征-train：\n',X_transformed.shape)
#print(sel_train.get_support(indices=False))


dfNew_train = pd.DataFrame()
dfNew_test = pd.DataFrame()
dfNew_train['id'] = df_train['id']
dfNew_train['playtime_forever'] = df_train['playtime_forever']
dfNew_test['id'] = df_test['id']
for ii in range(len(sel_train.get_support(indices=False))):
    if(sel_train.get_support(indices=False)[ii]):
        print(X.columns[ii])
        dfNew_train[X.columns[ii]] = df_train[X.columns[ii]]
        dfNew_test[X.columns[ii]] = df_test[X.columns[ii]]
print(dfNew_train.shape)
print(dfNew_test.shape)


''' save new train csv '''
dfNew_train.to_csv('data/feature_selection_train.csv',index=False)
dfNew_test.to_csv('data/feature_selection_test.csv',index=False)