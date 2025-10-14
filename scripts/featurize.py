"""
Feature Engineering script for Titanic survival prediction.
Handles feature transformations and engineering steps."""

import argparse
from pathlib import Path
import warnings
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer



# Ignore all warnings
warnings.filterwarnings("ignore")

def feauturize_data(df):
    """Feature engineering for the Titanic dataset.
    
    Args:
        df: Combined dataframe containing both train and test data
        
    Returns:
        train_processed: Processed training data
        test_processed: Processed test data
    """
    # Split into train and test
    train = df.loc[:890].copy()
    test = df.loc[891:].copy()
    
    # Process test set
    test.drop(columns=['Survived'], inplace=True)
    test = test.drop("PassengerId", axis=1)
    
    # Process train set
    train['Survived'] = train['Survived'].astype('int64')
    train = train.drop("PassengerId", axis=1)
    X_train = train.drop("Survived", axis=1)
    y_train = train["Survived"]
    
    # Define feature transformations
    num_cat_transformation = ColumnTransformer([
        ('scaling', MinMaxScaler(), [0, 2]),  # Age, Fare
        ('onehot1', OneHotEncoder(sparse=False), [1, 3]),  # Pclass, Sex
        ('ordinal', OrdinalEncoder(), [4]),  # SibSp
        ('onehot2', OneHotEncoder(sparse=False), [5, 6])  # Parch, Embarked
    ], remainder='passthrough')
    
    # Define binning transformation
    bins = ColumnTransformer([
        ('Kbins', KBinsDiscretizer(n_bins=15, encode='ordinal', strategy='quantile'), [0, 2]),  # Age, Fare
    ], remainder='passthrough')
    
    # Create and fit the feature engineering pipeline
    feature_pipeline = Pipeline([
        ('num_cat_transformation', num_cat_transformation),
        ('bins', bins)
    ])
    
    # Fit and transform training data
    X_train_processed = feature_pipeline.fit_transform(X_train)
    
    # Transform test data using fitted pipeline
    X_test_processed = feature_pipeline.transform(test)
    
    # Return processed datasets
    return X_train_processed, y_train, X_test_processed




def load_data(train_path, test_path):
    """Load training and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test




def main():
    parser = argparse.ArgumentParser(description="Preprocess Titanic dataset")
    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training CSV file"
    )
    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test CSV file"
    )
    parser.add_argument(
        "--output_train",
        type=str,
        required=True,
        help="Output path for preprocessed training data",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        required=True,
        help="Output path for preprocessed test data",
    )

    args = parser.parse_args()

    # Create output directories if they don't exist
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_test).parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train, test = load_data(args.train_path, args.test_path)
    print(f"Loaded train: {train.shape}, test: {test.shape}")

    # Combine train and test for consistent preprocessing
    test['Survived'] = -1  # Placeholder for test set
    combined_df = pd.concat([train, test], ignore_index=False)
    
    print("Feature engineering...")
    X_train_processed, y_train, X_test_processed = feauturize_data(combined_df)

    print("Saving preprocessed data...")
    # Convert to DataFrame for saving
    train_preprocessed = pd.DataFrame(X_train_processed)
    train_preprocessed['Survived'] = y_train
    test_preprocessed = pd.DataFrame(X_test_processed)
    
    train_preprocessed.to_csv(args.output_train, index=False)
    test_preprocessed.to_csv(args.output_test, index=False)

    print(f"Preprocessed train saved to: {args.output_train}")
    print(f"Preprocessed test saved to: {args.output_test}")
    print(f"Final train shape: {X_train_processed.shape}")
    print(f"Final test shape: {X_test_processed.shape}")


if __name__ == "__main__":
    main()
