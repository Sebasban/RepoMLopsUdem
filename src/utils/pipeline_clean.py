import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PipelineCleanNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.clean_numeric(X)

    @staticmethod
    def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1) Normaliza nombres
        df.columns = (df.columns.str.strip().str.lower()
                    .str.replace(" ", "_").str.replace("-", "_"))

        # 2) Strings: trim, lower y '?' -> NaN
        for c in df.select_dtypes(include="object").columns:
            df[c] = (df[c].astype("string").str.strip().str.lower()
                            .replace({"?": pd.NA}))

        # 3) Target binario
        if "income" in df.columns:
            df["income"] = (df["income"]
                            .replace({">50k": 1, "<=50k": 0, ">50k.": 1, "<=50k.": 0})
                            .astype("Int64"))

        # 4) Feature engineering
        if {"capital_gain","capital_loss"}.issubset(df.columns):
            df["capital_net"] = df["capital_gain"].fillna(0) - df["capital_loss"].fillna(0)
            df["has_capital_gain"] = (df["capital_gain"].fillna(0) > 0).astype("Int8")
            df["has_capital_loss"] = (df["capital_loss"].fillna(0) > 0).astype("Int8")

        # 5) Columnas redundantes y no deseadas
        drop_cols = []
        if "education" in df.columns and "education_num" in df.columns:
            drop_cols.append("education")
        
        # Eliminar columnas específicas
        for col in ["fnlwgt", "race", "sex"]:
            if col in df.columns:
                drop_cols.append(col)
        
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        # 6) Tipado numérico
        for c in {"age","education_num","capital_gain","capital_loss","hours_per_week","capital_net"} & set(df.columns):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 7) Duplicados exactos
        df = df.drop_duplicates(ignore_index=True)
        return df
        
class PipelineCleanCategoric(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot: bool = True, top_k: int | None = None, min_freq: int | None = None):
        self.one_hot = one_hot
        self.top_k = top_k
        self.min_freq = min_freq

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.clean_categoric(X, one_hot=self.one_hot, top_k=self.top_k, min_freq=self.min_freq)

    @staticmethod
    def clean_categoric(df: pd.DataFrame, *, one_hot: bool = True,
                    top_k: int | None = None, min_freq: int | None = None) -> pd.DataFrame:
        df = df.copy()

        # Categóricas -> imputación + rare → __other__ + one-hot
        cat_cols = df.select_dtypes(include=["object","string"]).columns.tolist()
        if cat_cols:
            # imputación por moda simple ("missing")
            for c in cat_cols:
                df[c] = df[c].fillna("missing")

                # compactar rarezas (elige UNO de los dos criterios: top_k o min_freq)
                if top_k is not None:
                    keep = set(df[c].value_counts().head(top_k).index)
                    df[c] = df[c].where(df[c].isin(keep), "__other__")
                elif min_freq is not None:
                    vc = df[c].value_counts()
                    keep = set(vc[vc >= min_freq].index)
                    df[c] = df[c].where(df[c].isin(keep), "__other__")

            if one_hot:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=False,
                                    dtype="Int8", prefix_sep="=")

        return df