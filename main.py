from modules.load_data import fetch_and_save_data
from modules.preprocess import check_and_impute
from modules.visualization import visualize_tsne, visualize_umap
from modules.feature_engineering import select_features
import pandas as pd
from modules.model import train_models

def main():
    # Fetch and load data
    fetch_and_save_data()
    
    # Assuming the data file path and the rest of the pipeline remain the same
    df = pd.read_csv('data/polish_companies_bankruptcy.csv')
    df_cleaned = check_and_impute(df)

    # Data Visualization
    visualize_tsne(df_cleaned.drop('class', axis=1))
    visualize_umap(df_cleaned.drop('class', axis=1))

    # Feature Engineering
    X = df_cleaned.drop('class', axis=1)
    y = df_cleaned['class']
    selected_features = select_features(X, y)
    df_selected = df_cleaned[selected_features + ['class']]

    # Model Building and Evaluation
    
    train_models(df_selected.drop('class', axis=1), df_selected['class'])

if __name__ == "__main__":
    main()
