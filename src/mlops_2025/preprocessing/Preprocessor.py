"""
Data preprocessing script for Titanic survival prediction.
Handles data loading, cleaning, and basic preprocessing steps.
"""
import pandas as pd
from mlops_2025.preprocessing import Base_Preprocessor

class Preprocessor(Base_Preprocessor):
    def clean_data(train, test):
        """Clean the data by handling missing values and dropping unnecessary columns."""
        # Drop Cabin column due to numerous null values
        train.drop(columns=["Cabin"], inplace=True)
        test.drop(columns=["Cabin"], inplace=True)

        # Fill missing values
        train["Embarked"].fillna("S", inplace=True)
        test["Fare"].fillna(test["Fare"].mean(), inplace=True)

        # Create unified dataframe for easier manipulation
        df = pd.concat([train, test], sort=True).reset_index(drop=True)
        df.corr(numeric_only=True)["Age"].abs()
        # Fill missing Age values using group median
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )

        return df


