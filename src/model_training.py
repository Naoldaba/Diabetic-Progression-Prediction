from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def linear_regression_trainer(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model

# def sgd_trainer(x_train, y_train, x_val, y_val, alps=[0.0001, 0.001, 0.01, 0.1]):
#     res = []
#     for alp in alps:
#         sgd_model = SGDRegressor(alpha=alp, max_iter=1000, tol=1e-3, random_state=42)
#         sgd_model.fit(x_train, y_train)

#         y_vali_prediction = sgd_model.predict(x_val)

#         mse = mean_squared_error(y_val, y_vali_prediction)

#         r2 = r2_score(y_val, y_vali_prediction)
#         res.append({'alp': alp, 'mse': mse, 'r2': r2})
#     return res


def sgd_trainer(x_train, y_train, x_val, y_val, alps=[0.0001, 0.001, 0.01, 0.1]):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    
    results = []
    
    for alp in alps:
        sgd_model = SGDRegressor(alpha=alp, max_iter=1000, tol=1e-3, random_state=42)
        sgd_model.fit(x_train_scaled, y_train)

        # Predict on validation set
        y_val_pred = sgd_model.predict(x_val_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'alp': alp,
            'model': sgd_model,
            'mse': mse,
            'r2': r2,
            'predictions': y_val_pred
        })
    
    # Select the best model based on MSE
    best_model_info = min(results, key=lambda x: x['mse'])
    
    return best_model_info, results