import pandas as pd

# read csv
df=pd.read_csv('2020_AFE.csv')

# preprocess
df=df.drop(df.columns[0], axis=1)
df=df[df['V-Type']==31]
df=df.drop('Year', axis=1)
df=df[df['Holiday']!=2]
df=df.reset_index(drop=True)
df.head()

# drop segment 05F0438N-05FR143N
df=df[df['Segment']!='05F0438N-05FR143N']

# read climb segment
dfc = pd.read_csv('etag_climb.csv')

# merge dfc to main df
df = df.merge(dfc,left_on = 'Segment',right_on = 'Segment1',how='left')

# fill climb na
df['Climb'] = df['Climb'].fillna(0).astype(int)

# add high feature
df['High'] = df['Segment'].apply(lambda x : 1 if ((x[2]=='H') or (x[11]=='H')) else 0)

# model training ========================================================================
import pandas as pd

# transform speed to classification
def speed_class(x):
    if x<50:
        return 0
    elif (x>=50 and x<=80):
        return 1
    else:
        return 2
       
df['speed_class'] = df['Speed'].apply(lambda x: speed_class(x))

# X dataframe
X = df[['Direction','Hour', 'Month', 'Wday', 'Holiday', 'Seg_cat', 'Climb', 'High']]

# y dataframe
y = df[['speed_class']]

# import
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV

# set XGBoost model
clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic')

param_dist = {'n_estimators': range(150,1000,10),
              'learning_rate': uniform(0.01, 0.59),
              'subsample': uniform(0.3, 0.6),
              'max_depth': range(3,15,1),
              'colsample_bytree': uniform(0.5, 0.4),
              'min_child_weight': range(1,10,1)
             }

kfold_5 = KFold(shuffle = True)

clf = RandomizedSearchCV(clf_xgb, 
                         param_distributions = param_dist,
                         cv = kfold_5,  
                         n_iter = 100, 
                         scoring = 'roc_auc', 
                         error_score = 0, 
                         verbose = 3, 
                         n_jobs = -1)

# split train & test dataset
X_train_xg, X_test_xg, y_train_xg, y_test_xg = train_test_split(X, y, test_size=0.1,random_state=20)

# fit model
clf.fit(X_train_xg, y_train_xg)

# get the best performance's parameter
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

report_best_scores(clf.cv_results_,1)

# use the best performance's parameter
model = xgb.XGBClassifier(colsample_bytree=0.6593330974815904, learning_rate= 0.2796339926807502, max_depth= 5, min_child_weight= 1, n_estimators= 872, subsample=0.6110079699316302)

# fit model
model.fit(X_train_xg, y_train_xg)

# predict y
y_pred_xg = model.predict(X_test_xg)

# calculate accuracy
xg_acc = metrics.accuracy_score(y_test_xg, y_pred_xg)
print('accuracy: {}'.format(xg_acc))

# export model
import joblib
joblib.dump(model, 'whole_year')