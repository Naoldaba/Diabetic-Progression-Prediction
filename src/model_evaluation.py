from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)

    plt.savefig("output/images/prediction_and_actual.png")


def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    return mse, r2