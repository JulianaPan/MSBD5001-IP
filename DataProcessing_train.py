import pandas as pd # used for handling the dataset
import datetime


''' Loading Data '''
df_train = pd.read_csv('data/train.csv',parse_dates=['purchase_date','release_date'])


''' Checking missing data '''
print(df_train.isnull().any())


''' Dealing with missing data '''
mean_days = (df_train['purchase_date']-df_train['release_date']).mean().days
df_train['purchase_date'] = df_train['purchase_date'].fillna(df_train['release_date'] + datetime.timedelta(days = mean_days))
df_train['total_positive_reviews'] = df_train['total_positive_reviews'].fillna(df_train['total_positive_reviews'].mean())
df_train['total_negative_reviews'] = df_train['total_negative_reviews'].fillna(df_train['total_negative_reviews'].mean())


''' Dealing with string and date '''
# boolean变量转0/1
df_train['is_free'] = df_train['is_free'].astype('int')


genres = df_train["genres"].str.get_dummies(",")
genres.columns = genres.columns.map(lambda x: 'genres_'+str(x))
categories = df_train["categories"].str.get_dummies(",")
categories.columns = categories.columns.map(lambda x: 'categories_'+str(x))
tags = df_train["tags"].str.get_dummies(",")
tags.columns = tags.columns.map(lambda x: 'tags_'+str(x))



# 合并表
dfNew_train = pd.merge(df_train, genres, how='left', left_index=True, right_index=True)
dfNew_train = pd.merge(dfNew_train, categories, how='left', left_index=True, right_index=True)
dfNew_train = pd.merge(dfNew_train, tags, how='left', left_index=True, right_index=True)
print(dfNew_train.columns)

# 提取年月日到表
dfNew_train['purchase_year'] = dfNew_train['purchase_date'].apply(lambda x: x.year)
dfNew_train['purchase_month'] = dfNew_train['purchase_date'].apply(lambda x: x.month)
dfNew_train['purchase_day'] = dfNew_train['purchase_date'].apply(lambda x: x.day)
dfNew_train['release_year'] = dfNew_train['release_date'].apply(lambda x: x.year)
dfNew_train['release_month'] = dfNew_train['release_date'].apply(lambda x: x.month)
dfNew_train['release_day'] = dfNew_train['release_date'].apply(lambda x: x.day)


''' Deleting non-using data '''
dfNew_train.drop(['genres'],axis=1,inplace=True)
dfNew_train.drop(['categories'],axis=1,inplace=True)
dfNew_train.drop(['tags'],axis=1,inplace=True)
dfNew_train.drop(['purchase_date'],axis=1,inplace=True)
dfNew_train.drop(['release_date'],axis=1,inplace=True)

''' save new train csv '''
dfNew_train.to_csv('data/new_train.csv',index=False)