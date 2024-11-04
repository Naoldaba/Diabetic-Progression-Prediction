from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def linear_regression_trainer(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    return model

def sgd_trainer(x_train, y_train, x_val, y_val, x_test, y_test, alps=[0.0001, 0.001, 0.01, 0.1]):
    results = []
    
    # Train and evaluate on test set
    for alp in alps:
        sgd_model = SGDRegressor(alpha=alp, max_iter=1000, tol=1e-3, random_state=42)
        sgd_model.fit(x_train, y_train)

        # Predict on test set
        y_test_pred = sgd_model.predict(x_test)
        
        # Calculate metrics on test set
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        results.append({
            'alp': alp,
            'model': sgd_model,
            'test_mse': test_mse,
            'test_r2': test_r2
        })

    # Select the best model based on test MSE
    best_model_info = min(results, key=lambda x: x['test_mse'])
    best_model = best_model_info['model']

    y_val_pred = best_model.predict(x_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    best_model_info.update({
        'val_mse': val_mse,
        'val_r2': val_r2,
        'val_predictions': y_val_pred
    })

    return best_model_info, results