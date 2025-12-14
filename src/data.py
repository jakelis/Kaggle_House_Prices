import pandas as pd 
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIG_DATA_DIR = ROOT /'data/raw/'

def load_train_data():
    return pd.read_csv(ORIG_DATA_DIR/'train.csv')


def load_test_data():
    return pd.read_csv(ORIG_DATA_DIR/'test.csv')