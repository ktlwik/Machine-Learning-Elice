import numpy as np
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random
from sklearn.cross_validation import KFold

def main(rs=108):
    data = pd.read_csv("./data/Hitters.csv", header=0)
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[1:-1])
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(x_mat, y_vec, test_size=0.5, random_state=rs)

    # Linear Regression
    rss, r2, mse = multi_var_hitter(x_train, x_test, y_train, y_test, x_label)
    print("Linear Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print()

    # Ridge Regression
    best_lambda_ridge, best_lambda_lasso = get_best_lambda_value_ridge_lasso(data)
    rss, r2, mse = multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda_ridge)
    print("Ridge Regression Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_ridge))
    print()

    # lasso
    rss, r2, mse = multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda_lasso)
    print("lasso Result")
    print("RSS: {}".format(rss))
    print("R^2: {}".format(r2))
    print("MSE: {}".format(mse))
    print("Best lambda value: {}".format(best_lambda_lasso))
    print()


def get_best_lambda_value_ridge_lasso(data):

    best_lambda_ridge = 0.0   # You should find the best lambda value via cross validation
    best_lambda_lasso = 0.0   # You should find the best lambda value via cross validation
    best_ridge_mse = -100000000.0
    best_lasso_mse = -100000000.0 
    L = 0.001
    y = 100000000.0
    jump = 2
    response_var = -1
    y_vec = data.ix[:, response_var].as_matrix().reshape(-1, 1)
    y_label = data.columns[response_var]

    x_label = ", ".join(data.columns[1:-1])
    x_mat = data.ix[:, 1:-1].as_matrix()
    x_mat = x_mat.reshape(-1, x_mat.shape[1])
    
    while L < y:
        
        kf = KFold(len(x_mat), n_folds=10)
        averageR = 0.0
        averageL = 0.0
        for train_index, test_index in kf:
            x_train, x_test = x_mat[train_index], x_mat[test_index]
            y_train, y_test = y_vec[train_index], y_vec[test_index]
            rssR, rR2, mseR =  multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, L)
            averageR += rR2
            rss, r2, mse =  multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, L)
            averageL += r2
        #print(averageR)
        
        if (averageR > best_ridge_mse):
            best_ridge_mse = averageR
            best_lambda_ridge = L
        if (averageL > best_lasso_mse):
            best_lasso_mse = averageR
            best_lambda_lasso = L
        
        
        
        L *= jump
    return best_lambda_ridge, best_lambda_lasso


def multi_var_hitter(x_train, x_test, y_train, y_test, x_label):
    regr = linear_model.LinearRegression()

    regr.fit(x_train, y_train)
    predicted_y_test = regr.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test)
    return rss, r2, mse


def multi_var_hitter_ridge(x_train, x_test, y_train, y_test, x_label, best_lambda):
    clf = linear_model.Ridge(alpha=best_lambda, fit_intercept=True)
    clf.fit(x_train, y_train)
    predicted_y_test = clf.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test) 

    return rss, r2, mse


def multi_var_hitter_lasso(x_train, x_test, y_train, y_test, x_label, best_lambda):
    clf = linear_model.Lasso(alpha=best_lambda, fit_intercept=True)
    clf.fit(x_train, y_train)
    predicted_y_test = clf.predict(x_test)
    rss = np.sum((predicted_y_test - y_test) ** 2)
    r2 = r2_score(y_test, predicted_y_test)
    mse = mean_squared_error(y_test, predicted_y_test) 
    return rss, r2, mse


if __name__ == "__main__":
    main()
