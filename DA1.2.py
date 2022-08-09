
# Import Dataset from sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)
    
    
    
    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    
    
    
    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    
    
    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
    
    
    
    return (b_0, b_1)


def plot_regression_line(x1,x2,x3,x4, y_train, b1,b2,b3,b4):
    # plotting the actual points as scatter plot
    

    plt.scatter(x2, y_train, color = "b",
    marker = "o", s = 3)

    plt.scatter(x1, y_train, color = "m",
    marker = "o", s = 3)
    
    plt.scatter(x3, y_train, color = "g",
    marker = "o", s = 3)

    plt.scatter(x4, y_train, color = "r",
    marker = "o", s = 3)
    
    
    # predicted response vector
    y_1 = b1[0] + b1[1]*x1
    y_2 = b2[0] + b2[1]*x2
    y_3 = b3[0] + b3[1]*x3
    y_4 = b4[0] + b4[1]*x4
    
    
    
    # plotting the regression line
    plt.plot(x1, y_1, color = "g")
    plt.plot(x2, y_2, color = "g")
    plt.plot(x3, y_3, color = "g")
    plt.plot(x4, y_4, color = "g")
    
    
    
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    
    # function to show plot
    plt.show()





# Creating pd DataFrames
house_df = pd.read_csv('Housing.csv')

print(house_df)
# Variables
X= house_df.drop(labels= 'price', axis= 1)
#print(X)
y= house_df['price']# Splitting the Dataset
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.35, random_state= 10)
print(X_train)
#print(y_train)
#print(y_test)
# Instantiating LinearRegression() Model
lr = LinearRegression()# Training/Fitting the Model

lr.fit(X_train, y_train)# Making Predictions

#lr.predict(X_test)
pred = lr.predict(X_test)
print(pred)


# Evaluating Model's Performance
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))


x1=X_train['area']
x2=X_train['bedrooms']
x3=X_train['bathrooms']
x4=X_train['stories']


b1 = estimate_coef(x1, y_train)
b2 = estimate_coef(x2, y_train)
b3 = estimate_coef(x3, y_train)
b4 = estimate_coef(x4, y_train)

plot_regression_line(x1,x2,x3,x4, y_train, b1,b2,b3,b4)







