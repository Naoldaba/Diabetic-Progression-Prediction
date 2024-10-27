import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath, delimiter='\t')
    data.rename(columns={'Y': 'Outcome'}, inplace=True)
    
    return data

def split_data(data):
    X = data.drop(columns='Outcome')
    Y = data['Outcome']
    
    x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)
    
    # Return the training, validation, and test sets
    return x_train, x_val, x_test, y_train, y_val, y_test
