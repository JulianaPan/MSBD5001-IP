# main libraries
import pandas as pd # used for handling the dataset
import datetime


''' Loading Data '''
df_test = pd.read_csv('data/test.csv',parse_dates=['purchase_date','release_date'])


''' Checking missing data '''
print(df_test.isnull().any())


''' Dealing with missing data '''
mean_days = (df_test['purchase_date']-df_test['release_date']).mean().days
df_test['purchase_date'] = df_test['purchase_date'].fillna(df_test['release_date'] + datetime.timedelta(days = mean_days))
df_test['total_positive_reviews'] = df_test['total_positive_reviews'].fillna(df_test['total_positive_reviews'].mean())
df_test['total_negative_reviews'] = df_test['total_negative_reviews'].fillna(df_test['total_negative_reviews'].mean())


''' Dealing with string and date '''
# 处理游戏genres
count_genre = 10
count = 20
genres_set = set()
for val in df_test['genres'].str.split(','):
    genres_set.update(val)
genres2int = {val:ii+1 for ii, val in enumerate(genres_set)}

genres = df_test['genres'].str.split(',',expand=True)
for ii in range(len(genres.columns),count_genre):
    genres[ii] = None
genres.columns = genres.columns.map(lambda x: 'genres'+str(x))
for key in genres:
    genres[key] = genres[key].map(genres2int)
    genres[key] = genres[key].fillna(0)

# 处理游戏categories
categories_set = set()
for val in df_test['categories'].str.split(','):
    categories_set.update(val)
categories2int = {val:ii+1 for ii, val in enumerate(categories_set)}

categories = df_test['categories'].str.split(',',expand=True)
for ii in range(len(categories.columns),count):
    categories[ii] = None
categories.columns = categories.columns.map(lambda x: 'categories'+str(x))
for key in categories:
    categories[key] = categories[key].map(categories2int)
    categories[key] = categories[key].fillna(0)

# 游戏tags转数字字典
tags_set = set()
for val in df_test['tags'].str.split(','):
    tags_set.update(val)
tags2int = {val:ii+1 for ii, val in enumerate(tags_set)}

tags = df_test['tags'].str.split(',',expand=True)
for ii in range(len(tags.columns),count):
    tags[ii] = None
tags.columns = tags.columns.map(lambda x: 'tags'+str(x))
for key in tags:
    tags[key] = tags[key].map(tags2int)
    tags[key] = tags[key].fillna(0)

# 合并表
dfNew = pd.merge(df_test, genres, how='left', left_index=True, right_index=True)
dfNew = pd.merge(dfNew, categories, how='left', left_index=True, right_index=True)
dfNew = pd.merge(dfNew, tags, how='left', left_index=True, right_index=True)

# 提取年月日到表
dfNew['purchase_year'] = dfNew['purchase_date'].apply(lambda x: x.year)
dfNew['purchase_month'] = dfNew['purchase_date'].apply(lambda x: x.month)
dfNew['purchase_day'] = dfNew['purchase_date'].apply(lambda x: x.day)
dfNew['release_year'] = dfNew['release_date'].apply(lambda x: x.year)
dfNew['release_month'] = dfNew['release_date'].apply(lambda x: x.month)
dfNew['release_day'] = dfNew['release_date'].apply(lambda x: x.day)

# boolean变量转0/1
dfNew['is_free'] = dfNew['is_free'].astype('int')


''' Deleting non-using data '''
dfNew.drop(['genres'],axis=1,inplace=True)
dfNew.drop(['categories'],axis=1,inplace=True)
dfNew.drop(['tags'],axis=1,inplace=True)
dfNew.drop(['purchase_date'],axis=1,inplace=True)
dfNew.drop(['release_date'],axis=1,inplace=True)


''' save new train csv '''
dfNew.to_csv('data/new_test.csv',index=False)



