import pandas as pd
from ucimlrepo import fetch_ucirepo

def fetch_and_save_data():
    # Fetch dataset using ucimlrepo
    polish_companies_bankruptcy = fetch_ucirepo(id=365)
    
    # Data (as pandas dataframes)
    X = polish_companies_bankruptcy.data.features
    y = polish_companies_bankruptcy.data.targets
    
    # Concatenate features and targets
    data = pd.concat([X, y], axis=1)
    
    # Save the data to CSV for easy access by other modules
    data.to_csv('data/polish_companies_bankruptcy.csv', index=False)
    print("Data saved to 'data/polish_companies_bankruptcy.csv'.")

if __name__ == "__main__":
    fetch_and_save_data()