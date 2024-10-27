from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_trainer(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model

def sgd_trainer(x_train, y_train, x_val, y_val, alps=[0.0001, 0.001, 0.01, 0.1]):
    res = []
    for alp in alps:
        sgd_model = SGDRegressor(alpha=alp, max_iter=1000, tol=1e-3, random_state=42)
        sgd_model.fit(x_train, y_train)

        y_vali_prediction = sgd_model.predict(x_val)

        mse = mean_squared_error(y_val, y_vali_prediction)

        r2 = r2_score(y_val, y_vali_prediction)
        res.append({'alp': alp, 'mse': mse, 'r2': r2})
    return res