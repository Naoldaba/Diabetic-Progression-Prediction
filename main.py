from src.data_pruning import load_data, split_data
from src.model_training import linear_regression_trainer, sgd_trainer
from src.model_evaluation import evaluate_model, plot_predictions
import os

data = load_data('data/diabetes.tab.txt')
x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

model = linear_regression_trainer(x_train, y_train)

val_mse, val_r2 = evaluate_model(model, x_val, y_val)
test_mse, test_r2 = evaluate_model(model, x_test, y_test)

with open("output/evaluation_metrices.txt", "w") as f:
    f.write(f"Validation MSE ---> {val_mse}\nValidation R2 ---> {val_r2}\n")
    f.write(f"Test MSE ---> {test_mse}\nTest R2 ---> {test_r2}\n")

plot_predictions(y_val, model.predict(x_val), title="Validation Set: Predictions vs Actual")
plot_predictions(y_test, model.predict(x_test), title="Test Set: Predictions vs Actual")

sgd_results = sgd_trainer(x_train, y_train, x_val, y_val)
print("SGD Tuning Results:", sgd_results)
