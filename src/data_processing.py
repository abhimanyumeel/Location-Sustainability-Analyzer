import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    """Load and combine the datasets"""
    # Load training data
    train_df = pd.read_csv('data/raw/train.csv')
    
    # Basic data exploration
    print("\nDataset Overview:")
    print("Shape:", train_df.shape)
    print("\nColumns:", train_df.columns.tolist())
    
    # Print unique locations
    print("\nAvailable Locations:")
    print(train_df['ADDRESS'].unique())
    
    return train_df

def process_data(df):
    """Process the data for model training"""
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Handle missing values if any
    processed_df = processed_df.dropna()
    
    # Convert categorical columns to appropriate type
    categorical_columns = ['POSTED_BY', 'BHK_OR_RK', 'ADDRESS']
    for col in categorical_columns:
        processed_df[col] = processed_df[col].astype('category')
    
    return processed_df

if __name__ == "__main__":
    # Test the data loading and processing
    df = load_data()
    processed_df = process_data(df)
    
    print("\nProcessed Data Overview:")
    print(processed_df.head())
    print("\nProcessed Data Info:")
    print(processed_df.info())
