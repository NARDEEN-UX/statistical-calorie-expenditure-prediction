import pandas as pd

def load_the_data(main_file_path):
    """
    Loads the dataset from the given path and prints Shape, Columns, and Types.
    """
    print("--- 1. LOADING THE DATA ---")
    df = pd.read_csv(main_file_path)
    
    # Check Shape
    rows, cols = df.shape
    print(f"\n[SHAPE]: {rows} rows × {cols} columns")
    
    # Check Column Names
    print(f"\n[COLUMNS]: {list(df.columns)}")
    
    # Check Data Types
    print("\n[DATA TYPES]:")
    print(df.dtypes)
    
    return df