import pandas as pd
import os
import joblib

def load_data(filepath:str):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".tsv":
        return pd.read_csv(filepath, sep="\t")
    return pd.read_csv(filepath)

def save_data(df:pd.DataFrame, filepath: str, filename: str):
    final_path = os.path.join(filepath, filename)
    dirpath = os.path.dirname(final_path)
    os.makedirs(dirpath, exist_ok=True)
    df.to_csv(final_path, index= False)

def save_objects(obj, filename):
    with open(f"models/{filename}", "wb") as f:
        joblib.dump(obj, f)