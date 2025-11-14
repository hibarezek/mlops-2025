class DataSplitter():
    
    def split_data(self, df):
        """Split the unified dataframe back into train and test sets."""
        train = df.loc[:890].copy()
        test = df.loc[891:].copy()

        # Remove Survived column from test set
        if "Survived" in test.columns:
            test.drop(columns=["Survived"], inplace=True)

        # Ensure Survived column is int in train set
        if "Survived" in train.columns:
            train["Survived"] = train["Survived"].astype("int64")

        return train, test
