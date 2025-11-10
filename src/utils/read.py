import pandas as pd 

class Read:
    def read_excel(path):
        return pd.read_excel(path)
    def read_csv(path):
        return pd.read_csv(path)
    def read_parquet(path):
        return pd.read_parquet(path)