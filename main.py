from src.data_pruning import load_data, split_data
from src.model_training import linear_regression_trainer, sgd_trainer
from src.model_evaluation import evaluate_model, plot_predictions
from sklearn.preprocessing import StandardScaler
import os

# Load and split the data
data = load_data('data/diabetes.tab.txt')
x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

# Train and evaluate Linear Regression model
linear_model = linear_regression_trainer(x_train, y_train)
val_mse, val_r2 = evaluate_model(linear_model, x_val, y_val)
test_mse, test_r2 = evaluate_model(linear_model, x_test, y_test)

# Write Linear Regression results to file in standard format
with open("output/linear_regression_results.txt", "w") as f:
    f.write("Linear Regression Model Evaluation\n")
    f.write("==================================\n")
    f.write(f"Validation MSE: {val_mse:.3f}\n")
    f.write(f"Validation R2: {val_r2:.3f}\n")
    f.write(f"Test MSE: {test_mse:.3f}\n")
    f.write(f"Test R2: {test_r2:.3f}\n")

# Plot and save Linear Regression predictions
plot_predictions(y_val, linear_model.predict(x_val), title="Validation Set: Linear Regression Predictions vs Actual")
plot_predictions(y_test, linear_model.predict(x_test), title="Test Set: Linear Regression Predictions vs Actual")

# Train and evaluate SGD with different alpha values, and save results
sgd_results = sgd_trainer(x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test)[1]
with open("output/sgd_results.txt", "w") as f:
    print(sgd_results)
    f.write("SGD Model Tuning Results\n")
    f.write("========================\n")
    for result in sgd_results:
        f.write(f"Alpha: {result['alp']}\n")
        f.write(f"test MSE: {result['test_mse']:.3f}\n")
        f.write(f"test R2: {result['test_r2']:.3f}\n")
        f.write("\n")

best_sgd_model_info = sgd_results = sgd_trainer(x_train_scaled, y_train, x_val_scaled, y_val, x_test_scaled, y_test)[0]
plot_predictions(
    y_val, best_sgd_model_info['val_predictions'], 
    title=f"Validation Set: SGD (Alpha={best_sgd_model_info['alp']}) Predictions vs Actual"
)

y_test_pred = best_sgd_model_info['model'].predict(x_test_scaled)
plot_predictions(
    y_test, y_test_pred, 
    title=f"Test Set: SGD (Alpha={best_sgd_model_info['alp']}) Predictions vs Actual"
)