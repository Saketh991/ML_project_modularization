import pandas as pd
from sklearn.impute import KNNImputer

def load_data(filepath):
    """ Load dataset from a file. """
    return pd.read_csv(filepath)
def check_and_impute(data):
    
    # Check if there are any null values in the DataFrame
    if data.isnull().any().any():
        print("Null values found. Starting imputation.")
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        imputed_data = imputer.fit_transform(data)
        # Return the DataFrame with imputed values and the same column names
        return pd.DataFrame(imputed_data, columns=data.columns)
    else:
        print("No null values found. No imputation needed.")
        return data

if __name__ == "__main__":
    df = load_data('data/polish_companies_bankruptcy.csv')
    print("Received the data from main.py, preprocessing the data")
    df = check_and_impute(df)
    df.to_csv('data/cleaned_data.csv', index=False)  # Save the cleaned data for the next step
