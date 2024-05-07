import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle  # Import pickle for model saving
import os

def load_data(filepath):
    """ Load dataset from a file. """
    return pd.read_csv(filepath)



def train_models(X, y):
    """ Train multiple models with Grid Search to find the best hyperparameters and save the best models. """
    
    # Setting up the parameter grid for Random Forest
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Setting up the parameter grid for Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.9, 1.0],
        'max_depth': [3, 7, 9]
    }

    models = {
        'Random Forest': (RandomForestClassifier(), rf_params),
        'Gradient Boosting': (GradientBoostingClassifier(), gb_params)
    }

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Directory to save the best models
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists

    # Training and tuning each model
    for name, (model, params) in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Getting the best estimator
        best_model = grid_search.best_estimator_
        
        # Testing the best estimator
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} best model accuracy: {accuracy}")

        # Save the best model to a file
        model_filename = os.path.join(model_dir, f'{name.replace(" ", "_").lower()}_best_model.pkl')
        with open(model_filename, 'wb') as file:
            pickle.dump(best_model, file)
        print(f"Saved {name} best model to {model_filename}")

    return grid_search.best_params_

# Example usage (assuming X and y are your features and target arrays respectively)
# best_params = train_models(X, y)


if __name__ == "__main__":
    df = load_data('data/selected_features_data.csv')
    print("Got the data from feature engineering, now training the models ")
    X = df.drop('class', axis=1)
    y = df['class']
    train_models(X, y)
