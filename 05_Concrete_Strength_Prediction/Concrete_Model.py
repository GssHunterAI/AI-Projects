#Libraries that will be used with the model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
#importing the dataset from foler 

data = pd.read_excel('D:\\AI Projects\\data\\Concrete_Data.xlsx')

#Split the data into X and Y
X= data.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
Y= data['Concrete compressive strength(MPa, megapascals) ']

#Normalize the data
sc = StandardScaler()
Normalized_X = sc.fit_transform(X)

#Split data to train and test
X_train, X_test, Y_train, Y_test = train_test_split(Normalized_X,Y, test_size=0.2, random_state=0)

#Linear Regression Model
ld = LinearRegression()
ld.fit(X_train, Y_train)
y_pred = ld.predict(X_test)
y_mean = mean_squared_error(Y_test, y_pred)
linearr2= r2_score(Y_test, y_pred)

#Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)
y_pred2 = rf.predict(X_test)
y_mean2 = mean_squared_error(Y_test, y_pred2)
rfr2score = r2_score(Y_test, y_pred2)

#Results
print('Linear Regression Mean Squared Error:', y_mean)
print('Linear Regression R2 Score:', linearr2)
print('Random Forest Mean Squared Error:', y_mean2)
print('Random Forest R2 Score:', rfr2score)

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(Y_test, y_pred, alpha=0.5, color='red', label='Linear Regression Predictions')
plt.scatter(Y_test, y_pred2, alpha=0.5, color='blue', label='Random Forest Predictions')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Comparison of Linear Regression and Random Forest Predictions')
plt.legend()
plt.show()

