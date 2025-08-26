# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#read Social ads file
data = pd.read_csv('D:\\AI Projects\\data\\Social_ads.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1 , random_state=0)
#label encoding
label_encoder = LabelEncoder()
X_train[:, 1] = label_encoder.fit_transform(X_train[:, 1])
X_test[:, 1] = label_encoder.transform(X_test[:, 1])

#decision tree classifier

testing_depth = [1,2,3,4,5,6,7,8,9,10]

for i in testing_depth:
    classifier = DecisionTreeClassifier(max_depth=i , random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print('Accuracy of the test: ', accuracy_score(y_test, y_pred)*100)
    print('Accuracy of the trained data: ', accuracy_score(y_train, classifier.predict(X_train))*100)

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1 , random_state=0)
#label encoding
label_encoder = LabelEncoder()
X_train[:, 1] = label_encoder.fit_transform(X_train[:, 1])
X_test[:, 1] = label_encoder.transform(X_test[:, 1])

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred1 = logreg.predict(X_test)
y_pred2 = logreg.predict(X_train)

accuracytest = accuracy_score(y_test, y_pred1)
accuracytrain = accuracy_score(y_train, y_pred2)

print('Acuuracy : test ' , accuracytest*100)
print('Acuurancy Train : ' ,accuracytrain*100)

#using standard scaler
from sklearn.preprocessing import StandardScaler
label_encoder = LabelEncoder()
X_train[:, 1] = label_encoder.fit_transform(X_train[:, 1])
X_test[:, 1] = label_encoder.transform(X_test[:, 1])
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_pred3 = sc.predict(X_test)
y_pred4 = sc.predict(X_train)

print('Accuracy of the test: ', accuracy_score(y_test, y_pred3)*100)
print('Accuracy of the trained data: ', accuracy_score(y_train, y_pred4)*100)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
results = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 15)
print("Accuracy: %.5f%% " % (results.mean()*100.0))


# Pick a Classifier you are searching for its best Paramters
from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', random_state = 0)


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10 , 100], 'kernel': ['linear']},
              {'C': [1, 10,100], 'kernel': ['rbf']}]


grid_search = GridSearchCV(estimator = classifier1,        # The Classifer That we need its best Parameters
                           param_grid = parameters,       # It must Be Dictionary or List Of Dictionaries
                           scoring = 'accuracy',          # The type of Evaluation Metric
                           cv = 10,                       # default None : Means K Fold =5 , you can change it to any 'int' Number
                           n_jobs = -1)                  # None :  For No Parallel Jobs , int : For a Certain Number of Parallel jobs , -1 : for Using ALL PROCESSORS!

grid_search = grid_search.fit(X_train, y_train)

print("best accuracy is :" , grid_search.best_score_)

grid_search.best_params_   # best_parms_  is a method in Grid Search to return The Best with resepct to the Metric
