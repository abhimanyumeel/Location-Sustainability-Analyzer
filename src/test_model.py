from model import HousePriceModel
import pandas as pd
from data_processing import load_data

def test_prediction():
    # Get a sample from our training data
    train_df = load_data()
    sample_data = train_df.iloc[0]  # Get first row as sample
    
    # Load the trained model
    model = HousePriceModel.load_model()
    
    # Create a test input
    test_input = pd.DataFrame({
        'POSTED_BY': [sample_data['POSTED_BY']],
        'UNDER_CONSTRUCTION': [sample_data['UNDER_CONSTRUCTION']],
        'RERA': [sample_data['RERA']],
        'BHK_NO.': [3],  # We can modify this
        'BHK_OR_RK': [sample_data['BHK_OR_RK']],
        'SQUARE_FT': [1500],  # We can modify this
        'READY_TO_MOVE': [1],
        'RESALE': [1],
        'ADDRESS': [sample_data['ADDRESS']],
        'LONGITUDE': [sample_data['LONGITUDE']],
        'LATITUDE': [sample_data['LATITUDE']]
    })
    
    # Make prediction
    prediction = model.predict(test_input)
    
    print("\nTest Input:")
    print(test_input)
    print(f"\nPredicted Price: {prediction[0]:.2f} lakhs")

if __name__ == "__main__":
    test_prediction()
