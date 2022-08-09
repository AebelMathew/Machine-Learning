import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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



def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color = "m",
    marker = "o", s = 3)
    
    
    
    # predicted response vector
    y_pred = b[0] + b[1]*x
    
    
    
    # plotting the regression line
    plt.plot(x, y_pred, color = "g")
    
    
    
    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    
    
    # function to show plot
    plt.show()
    


def main():
    # observations / data
    student_df=pd.read_csv('datasets/weight-height.csv')
    x=student_df['Height']
    y=student_df['Weight']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.35, random_state= 10)
    #print(X_train)
    #print(X_test)
    
    
    # estimating coefficients
    b = estimate_coef(X_train, y_train)
    print("Estimated coefficients:\nb_0 = {} \
    \nb_1 = {}".format(b[0], b[1]))
    
    
    h=int(input("Input the height: "))
    w=b[1]*h+b[0]
    print("The predicted weight is: ",w)

    pred=X_test*b[1]+b[0]
    print("The values of test set is:")
    print(X_test)

    print("The predicted values of test set is:")
    print(pred)

    print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
    print('Mean Squared Error:', mean_squared_error(y_test, pred))
    print('Mean Root Squared Error:', np.sqrt(mean_squared_error(y_test, pred)))

    
    # plotting regression line
    plot_regression_line(X_train, y_train, b)
    
    
    
if __name__ == "__main__":
    main()
