#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data preparation


dataset = pd.read_csv('OnlineNewsPopularity.csv')

type(dataset)

dataset.head()

#information about dataset (types of indexes and columns, non-zero values and the use of memory)
dataset.info()

# It can be seen that most of the data are of the quantitative type 


#checking if there are missing values for each variable
dataset . isnull () . sum ()  /  dataset . shape [ 1 ]

# We can see there are no missing values


# # Exploratory Data Analysis

#  Descriptive Analysis:
#  A descriptive statistic of the quantitative variables of the dataset is carried out.


dataset.describe()
#Generate descriptive statistics, summarize the dataset


print(dataset.columns)


dataset.corr()


# # Visualization

# Creating simple plot to check out how different variable impact target variable


# Scatter Plot: Number of words in the content of the article vs number of shares


sns.scatterplot(x= dataset.iloc[:, 3], y= dataset.iloc[:, 60])
sns.jointplot(x= dataset.iloc[:, 3], y= dataset.iloc[:, 60])


# The above plots indicates a negative correlation between the number of words in the content and the number of shares.
# Such as people don't like to read longer news articles, so that they won't share.
# It also indicates outliers



Shares = dataset.iloc[:, 60]

# Visualizing the distribution type of target variable shares

plt.figure()
sns.distplot(Shares)
plt.title('Distribution of share')
plt.show()

#more plots
#box plot for the target variable shares
sns.boxplot(Shares)

#boxplot for number of words in title
n_tokens_title = dataset.iloc[:, 2]
sns.boxplot(n_tokens_title)

#boxplot for the number of words in article
n_tokens_content = dataset.iloc[:, 3]
sns.boxplot(n_tokens_content)


#Computing pairwise correlation of numbers of words in title and shares

c = dataset.iloc[:,[2,60]]
c.corr(method='spearman')

# The length of the title is not well correlated with shares.



#Computing pairwise correlation of numbers of words in Articles and shares
c = dataset.iloc[:,[3,60]]
c.corr(method='spearman')

#  There is no direct dependence on the number of words and shares.


# Article published on different days of the week

dataset.iloc[:,31:39].sum().plot(kind='bar')
plt.ylabel('count')

# In this plot we can see that the dataset contains more articles that are published during weekdays as compare to the weekends

#Pie chart for the topics of article
dataset.iloc[:,13:18].sum().plot(kind='pie', autopct='%1.1f%%')
sns.set(rc={'figure.figsize':(12, 8)})

# This pie chart shows that Most of the articles were published on the topic of "Technology" and the least on "lifestyle" topic.



# ### checking co-relation between the data

corr = dataset.corr()
dims = (16.7, 12.27)
fig, ax = plt.subplots(figsize=dims)
sns.heatmap(corr,  vmax=.8, square=True )

c = dataset.columns[60]


# ### Heatmap  for  most Correlated Features of all

# Top 20 features

k = 20 #number of variables for heatmap
cols = corr.nlargest(k, c).index
cm = np.corrcoef(dataset[cols].values.T)
sns.set(font_scale=1.25)
fig, ax = plt.subplots(figsize=dims)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


#most correlated features
most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


#Distribution of share with highly correlated features
plt.figure()
sns.distplot(dataset.iloc[:,27])
plt.title('Distribution of share')
plt.show()


# ### Checking Skewness and Kurtosis Values to check the perfection of the normal distribution


#Plotting features with high correlation
sns.jointplot(x= dataset.iloc[:, 27], y= dataset.iloc[:, 60], kind ='reg')


print("Skewness: %f" % dataset.iloc[:,60].skew())
print("Kurtosis: %f" % dataset.iloc[:,60].kurt())


# We observe slightly high skewness and Kurtosis values so lets try reduce it.


# log1p insures a more homogeneous distribution when a dataset contains numbers close to zero which is more suitable in our case

   
# Applying log1p for shares and visualizing the skewness again

dataset.iloc[:,60] = np.log1p(dataset.iloc[:,60])

sns.distplot(dataset.iloc[:,60], fit=norm);

fig = plt.figure()
res = stats.probplot(dataset.iloc[:,60], plot=plt)
plt.show()

y_train = dataset.iloc[:,60].values

print("Skewness: %f" % dataset.iloc[:,60].skew())
print("Kurtosis: %f" % dataset.iloc[:,60].kurt())



# Applying log1p for avg keywords(average share) and visualizing the skewness again

dataset.iloc[:, 27] = np.log1p(dataset.iloc[:, 27])

sns.distplot(dataset.iloc[:, 27], fit=norm);

fig = plt.figure()
res = stats.probplot(dataset.iloc[:, 27], plot=plt)
plt.show()

y_train = dataset.iloc[:, 27].values

print("Skewness: %f" % dataset.iloc[:, 27].skew())
print("Kurtosis: %f" % dataset.iloc[:, 27].kurt())


# Getting standard deviation of each feature
s = dataset.describe().loc['std']  
s.head(20)


# ### Fixing "skewed" features.
# Here, we fix all of the skewed data to be more normal so that our models will be more accurate when making predictions.


#Drop first two columns because they are irrelevant 
df = dataset.drop(dataset.columns[[0,1]],axis=1)

numeric_feats = df.index

# Check the skew of all numerical features
skewed_feats = df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skewed Features' :skewed_feats})
skewness.head()

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df[feat] = boxcox1p(df[feat], lam)
    df[feat] += 1


# # Splitting data

from sklearn.model_selection import train_test_split

df.head()

print(df.columns)

# highly correlated features
cols

X =df.iloc[:,[25,40,24,28,26,27,5,22,23,7,42,19,18,57,8,54,10,36,34]]
y = df.iloc[:, -1]


#Train and test set split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train.head()


# # Modeling and Prediction


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
import xgboost as xgb


# ### linear Regression


#implementing linear regression model

regressor = LinearRegression()
regressor.fit(X_train, y_train)


train_pred = regressor.predict(X_train) 
test_pred = regressor.predict(X_test)

print('Train data R^2 score:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# ### Decision Tree Regressor 


reg = DecisionTreeRegressor(max_depth=5,random_state=0)
reg.fit(X_train, y_train)


train_pred = reg.predict(X_train) 
test_pred = reg.predict(X_test)


print('Train data R^2 score:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# ### Randmon Forest Regressor

randomForest_model = RandomForestRegressor ( n_estimators  =  150 , random_state  =  0 ) 
randomForest_model.fit ( X_train , y_train ) 


train_pred = randomForest_model.predict(X_train) 
test_pred = randomForest_model.predict(X_test)


print('Train data R^2 score:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# ### Xgboost

xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)
xgb_model.fit(X_train, y_train)

train_pred = xgb_model.predict(X_train) 
test_pred = xgb_model.predict(X_test)

print('Train data R^2 score:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


plt.scatter(y_test,test_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ### Lasso Regression

lasso = Lasso()

parameters = {"alpha": [0.1, 1.0, 2, 5, 10]}

lasso_reg = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5  )

lasso_reg.fit(X_train, y_train)

train_pred = lasso_reg.predict(X_train) 
test_pred =  lasso_reg.predict(X_test)

print('Train data R^2 score:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE:', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


# # Tunning Hyperparameter

# ### Xgboost


# Tuning hyper-parameters by GridSearchCV
param_grid = {'max_depth': [int(x) for x in np.linspace(3, 30, 3)],
              'min_child_weight': [1, 3, 5],
              'subsample': [i/10.0 for i in range(6, 10)],
              }
tuned_xgboost  = GridSearchCV(estimator = xgb.XGBRegressor(learning_rate = 0.1, n_estimators = 200, colsample_bytree = 0.8, objective= 'reg:linear', 
                                            nthread = 4, scale_pos_weight = 1, seed = 27), param_grid = param_grid, scoring = 'neg_mean_squared_error',
                                            n_jobs = -1, iid = False, cv = 4)
tuned_xgboost.fit(X_train, y_train)


tuned_xgboost.best_params_

train_pred = tuned_xgboost.predict(X_train) 
test_pred = tuned_xgboost.predict(X_test)

print('After tuning hyperparameter of Xgboost  ')
print('Train data R^2 score is:',metrics.r2_score(y_train, train_pred))
print('Test data R^2 score is:',metrics.r2_score(y_test, test_pred))
print('Train data RMSE is :', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('Test data RMSE is:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

best_estimator  =  tuned_xgboost.best_estimator_
xgb.plot_importance(best_estimator,max_num_features=20)
plt.show()


# ### Random Forest Regressor tuning

# preparing list of hyperparameter.
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(3, 30, num = 3)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# creating random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Tuning hyper-parameters by RandomizedSearchCV
tune_rf = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid, n_iter = 100, cv = 3, scoring = 'neg_mean_squared_error', verbose=2, random_state = 42, n_jobs = -1)
tune_rf.fit(X_train, y_train)

tune_rf.best_params_

train_pred = tune_rf.predict(X_train) 
test_pred = tune_rf.predict(X_test)

print('tuned random forest regressor Train data R^2 score is:',metrics.r2_score(y_train, train_pred))
print('tuned random forest regressor Test data R^2 score is:',metrics.r2_score(y_test, test_pred))
print('tuned random forest regressor Train data RMSE is :', np.sqrt(metrics.mean_squared_error(y_train, train_pred)))
print('tuned random forest regressor Test data RMSE is:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))


features = df.columns
best_estimator  =  tune_rf.best_estimator_ 
importances = best_estimator.feature_importances_
indices = np.argsort(importances)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# # Conclusion 

# After completing the different stages of the process: Preparation, Exploration, and Modeling, and by applying  the different regressions models. I have come to conclusion that Random Forest regressor after hyper parameter tuning has the best  RMSE 0.1340970 and the best  R squared  0.14462525 on test set.





