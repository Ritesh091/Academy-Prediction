import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

movies = pd.read_csv('C:\\Users\Ritesh\PycharmProjects\BS\Oscar\pict.csv')
print(movies.head())

movies1 = pd.read_csv('C:\\Users\Ritesh\PycharmProjects\BS\Oscar\pict1.csv')
print(movies1.head())

le = preprocessing.LabelEncoder()
movies['Genre'] = le.fit_transform(movies['Genre'])
movies['Genre1'] = le.fit_transform(movies['Genre1'])
movies['Genre2'] = le.fit_transform(movies['Genre2'])
movies['Winner'] = le.fit_transform(movies['Winner'])
movies['Rating'] = le.fit_transform(movies['Rating'])

movies1['Genre'] = le.fit_transform(movies1['Genre'])
movies1['Genre1'] = le.fit_transform(movies1['Genre1'])
movies1['Genre2'] = le.fit_transform(movies1['Genre2'])
movies1['Winner'] = le.fit_transform(movies1['Winner'])
movies1['Rating'] = le.fit_transform(movies1['Rating'])

#features = zip(genre_encoded,genre1_encoded, genre2_encoded)

train_X = movies.drop(['year', 'category', 'Name', 'Winner'], axis=1)
test_X = movies1.drop(['year', 'category', 'Name', 'Winner'], axis=1)

train_Y = movies['Winner']

clf = svm.SVC(kernel='rbf')
clf.fit(train_X, train_Y)

yhat = clf.predict(test_X)
print(yhat)

rf_model = RandomForestClassifier(n_estimators = 50, oob_score = True, random_state=0)
rf_model.fit(train_X, train_Y)
rf_model.oob_score_

for name, importance in zip(train_X.columns, rf_model.feature_importances_):
    print(name, "=", importance)

#If this is your first night at Fight Club, you have to fight.
pred_forest = rf_model.predict_proba(test_X)
forest_prediction = pd.DataFrame(pred_forest, movies1['Name'])
print(forest_prediction)

clf = DecisionTreeClassifier()
clf.fit(train_X, train_Y)

y_pred = clf.predict_proba(test_X)
dec =  pd.DataFrame(y_pred, movies1['Name'])
print(dec)