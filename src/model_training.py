from data_processing import load_data, process_data
from model import HousePriceModel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate():
    # Load and process data
    print("Loading data...")
    df = load_data()
    processed_df = process_data(df)
    
    # Create and train model
    print("\nTraining model...")
    model = HousePriceModel()
    test_score = model.train(processed_df)
    
    # Get feature importance
    feature_columns = [
        'POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.',
        'BHK_OR_RK', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE',
        'ADDRESS', 'LONGITUDE', 'LATITUDE'
    ]
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.model.feature_importances_
    })
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.sort_values('importance', ascending=False),
                x='importance', y='feature')
    plt.title('Feature Importance in Price Prediction')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    
    # Print summary statistics
    print("\nModel Training Summary:")
    print(f"Test Score (R²): {test_score:.4f}")
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
    
    # Print some sample predictions
    sample_data = processed_df.sample(5)
    predictions = model.predict(sample_data)
    actual_values = sample_data['TARGET(PRICE_IN_LACS)']
    
    print("\nSample Predictions vs Actual Values (in Lakhs):")
    for pred, actual in zip(predictions, actual_values):
        print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}, Difference: {(pred-actual):.2f}")
    
    return model, test_score

if __name__ == "__main__":
    model, test_score = train_and_evaluate()
