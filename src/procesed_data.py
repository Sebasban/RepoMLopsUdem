from utils.constants import Constants
from utils.read import Read
from utils.write import write
from sklearn.pipeline import Pipeline
from utils.pipeline_clean import PipelineCleanNumeric, PipelineCleanCategoric

def main(df):
    pipeline = Pipeline([
    ('numeric_cleaner', PipelineCleanNumeric()),
    ('categoric_cleaner', PipelineCleanCategoric(one_hot=True, top_k=5)),
    ('passthrough', 'passthrough')
    ])
    return pipeline.fit_transform(df)

if __name__ == "__main__":
    root_path = Constants.root_path.value
    raw_path = Constants.path_raw.value
    raw_file = Constants.raw_file.value
    procesed_path = Constants.path_processed.value
    processed_file = Constants.processed_file.value

    #ETL
    df = Read.read_csv(f'{root_path}{raw_path}{raw_file}.csv')
    df = main(df)
    write(df, f'{root_path}{procesed_path}{processed_file}.parquet')
