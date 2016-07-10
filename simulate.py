import pandas
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold,RFECV,SelectFromModel
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_predict,ShuffleSplit,cross_val_score
from sklearn.linear_model import LinearRegression ,LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC  #classifier
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import samples_generator

poly = np.poly1d([0.003, -0.02, -40, 0])


#target = np.random.randint(2,size = [1000,1])
#data = np.column_stack((data,target))
data = np.array([0,0])
for i in range(0,150,1):
    X = random.randint(-200, 200)
    Y = poly(X)
    if Y > 0:
        Y = 1
    else:
        Y = 0
    array = np.array([X,Y])
    data = np.row_stack((data,array))



X, y = samples_generator.make_classification(n_features=20, n_informative=3, n_redundant=12, n_classes=2,n_clusters_per_class=2,n_samples=1000)

print(X.shape)
print(y.shape)


# .8train /.2test
xtrain, xtest, ytrain, ytest = train_test_split(X, y ,test_size=0.20)
n_samples, n_features = xtrain.shape

# LinearRegression
LR = LinearRegression()
LR.fit(xtrain, ytrain)
prediction = LR.predict(xtest)#Predicting error
print("LR mean_squared_error")
print(mean_squared_error(ytest,prediction))
# RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
RFR.fit(xtrain, ytrain)
prediction = RFR.predict(xtest)#Predicting error
print("RFR mean_squared_error")
print(mean_squared_error(ytest,prediction))

svm  = SVC(C=100,gamma=0.001)
svm.fit(xtrain, ytrain)
prediction = svm.predict(xtest)#Predicting error
print("SVC mean_squared_error")
print(mean_squared_error(ytest,prediction))

cv = ShuffleSplit(n_samples, n_iter=50, test_size=0.2, random_state=0)
test_scores1 = cross_val_score(LR, xtrain, ytrain, cv=cv, n_jobs=2)

cv = ShuffleSplit(n_samples, n_iter=50, test_size=0.2, random_state=0)
test_scores2 = cross_val_score(RFR, xtrain, ytrain, cv=cv, n_jobs=2)

cv = ShuffleSplit(n_samples, n_iter=50, test_size=0.2, random_state=0)
test_scores3 = cross_val_score(svm, xtrain, ytrain, cv=cv, n_jobs=2)

print(test_scores1)
print(test_scores2)
print(test_scores3)

Max = max([np.mean(test_scores1),np.mean(test_scores2),np.mean(test_scores3)])
if Max == np.mean(test_scores1):
    LR.fit(xtrain, ytrain)
    prediction = LR.predict(xtest)#Predicting error
    print("LR mean_squared_error")
    print(mean_squared_error(ytest,prediction))
elif Max == np.mean(test_scores2):
    RFR.fit(xtrain, ytrain)
    prediction = RFR.predict(xtest)#Predicting error
    print("RFR mean_squared_error")
    print(mean_squared_error(ytest,prediction))
elif Max == np.mean(test_scores3):
    svm.fit(xtrain, ytrain)
    prediction = svm.predict(xtest)#Predicting error
    print("SVC mean_squared_error")
    print(mean_squared_error(ytest,prediction))
