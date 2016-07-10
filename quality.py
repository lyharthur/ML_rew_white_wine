import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import VarianceThreshold,RFECV,SelectFromModel
from sklearn.cross_validation import train_test_split,StratifiedKFold,cross_val_predict,ShuffleSplit,cross_val_score
from sklearn.linear_model import LinearRegression ,LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC  #classifier
from sklearn.grid_search import GridSearchCV


filename = ("winequality-red.csv")
white = pandas.read_csv(filename,delimiter=';')
#print(white.columns)
#print(white.shape)

data = white["quality"]
i=0
for val in data:
    if val > 5:
        data.set_value(i,1)
        i= i+1
    else:
        data.set_value(i,0)
        i= i+1
#print(data)

#low variance threshold
#print(white.corr()["quality"])
sel = VarianceThreshold(threshold=(.8*(1-.8)))
white_new = sel.fit_transform(white)
print(white_new)

#print(white_new.shape)

# .8train /.2test
xtrain, xtest, ytrain, ytest = train_test_split(white_new[:,:-1], white_new[:,-1], test_size=0.20)
n_samples, n_features = xtrain.shape


from sklearn.decomposition import RandomizedPCA

pca = RandomizedPCA(n_components=2)
X_pca = pca.fit_transform(xtrain)

print(X_pca.shape)
from itertools import cycle

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['+', 'o', '^', 'v', '<', '>', 'D', 'h', 's']
print(ytrain)
for i, c, m in zip(np.unique(ytrain), cycle(colors), cycle(markers)):
    plt.scatter(X_pca[ytrain == i, 0], X_pca[ytrain == i, 1],
               c=c, marker=m, label=i, alpha=0.5)

plt.legend(loc='best')
plt.show()



print(xtrain.shape)
print(xtest.shape)
print(ytrain.shape)
print(ytest.shape)

# LinearRegression
LR = LinearRegression()

# RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)


##Grid CV find C gamma for svc
svc_params = {
    'C': np.logspace(-1, 2, 4),
    'gamma': np.logspace(-4, 0, 5),
}
n_subsamples = 500
X_small_train, y_small_train = xtrain[:n_subsamples], ytrain[:n_subsamples]
gs_svc = GridSearchCV(SVC(), svc_params, cv=10, n_jobs=-1)
gs_svc.fit(X_small_train, y_small_train)
print(gs_svc.best_params_)
print(gs_svc.best_score_)
##Grid CV find C gamma for svc
svm  = SVC(C=gs_svc.best_params_['C'],gamma=gs_svc.best_params_['gamma'])

cv = ShuffleSplit(n_samples, n_iter=10, test_size=0.1, random_state=0)
test_scores1 = cross_val_score(LR, xtrain, ytrain, cv=cv, n_jobs=2)
cv = ShuffleSplit(n_samples, n_iter=10, test_size=0.1, random_state=0)
test_scores2 = cross_val_score(RFR, xtrain, ytrain, cv=cv, n_jobs=2)
cv = ShuffleSplit(n_samples, n_iter=10, test_size=0.1, random_state=0)
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



