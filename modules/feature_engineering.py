import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score

def load_data(filepath):
    """ Load dataset from a file. """
    return pd.read_csv(filepath)

def select_features(X, y):
    """ Select features using backward elimination and XGBoost. """
    xgb = XGBClassifier(eval_metric='logloss')
    sfs = SequentialFeatureSelector(xgb, n_features_to_select=10, direction='backward')
    pipeline = Pipeline([
        ('feature_selection', sfs),
        ('classification', xgb)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("Cross-validation scores:", scores)
    pipeline.fit(X, y)
    selected_features = X.columns[pipeline.named_steps['feature_selection'].get_support()]
    return selected_features

if __name__ == "__main__":
    df = load_data('data/cleaned_data.csv')
    X = df.drop('class', axis=1)
    y = df['class']
    selected_features = select_features(X, y)
    df_selected = df[selected_features.to_list() + ['class']]
    df_selected.to_csv('data/selected_features_data.csv', index=False)
    print("Saved the dataset with the selected features")
