import pandas as pd
class Data_Loader():

    def load_data(train_path, test_path)-> pd.DataFrame:
        """Load training and test datasets."""
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
    
    def save_data(df, output_path):
        """Save dataframe to CSV."""
        df.to_csv(output_path, index=False)