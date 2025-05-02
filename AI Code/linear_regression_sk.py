import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# Load the data 
data = pd.read_excel('D:\AI Projects\data\data.xlsx')

#Split the data into X and Y
X = data.iloc[:, 0]
Y = data.iloc[:, -1]

#Normalize the data
sc = StandardScaler()
Normalized_X= sc.fit_transform(X.values.reshape(-1,1))

#Split data to train and test

X_train, X_test, Y_train, Y_test = train_test_split(data['X'], data['Y'], test_size=0.2, random_state=0)

#Training the model

ld = LinearRegression()
ld.fit(X_train.values.reshape(-1,1), Y_train.values.reshape(-1,1))
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, ld.predict(X_train.values.reshape(-1,1)), color='blue')

#Testing the model with the test data

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, ld.predict(X_test.values.reshape(-1,1)), color='blue')

#Test the model accuracy

print(ld.score(X_test.values.reshape(-1,1), Y_test.values.reshape(-1,1)))