def write(df, path):
    df.to_parquet(path, index=False)