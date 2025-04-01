import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class HousePriceModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, df, training=True):
        """Prepare features for model training/prediction"""
        X = df.copy()
        
        # Handle categorical variables
        categorical_columns = ['POSTED_BY', 'BHK_OR_RK', 'ADDRESS']
        
        if training:
            # Initialize label encoders for each categorical column
            for col in categorical_columns:
                self.label_encoders[col] = LabelEncoder()
                X[f'{col}_encoded'] = self.label_encoders[col].fit_transform(X[col])
        else:
            # Use existing label encoders for transformation
            for col in categorical_columns:
                X[f'{col}_encoded'] = self.label_encoders[col].transform(X[col])
        
        # Select features for model
        feature_columns = [
            'POSTED_BY_encoded',
            'UNDER_CONSTRUCTION',
            'RERA',
            'BHK_NO.',
            'BHK_OR_RK_encoded',
            'SQUARE_FT',
            'READY_TO_MOVE',
            'RESALE',
            'ADDRESS_encoded',
            'LONGITUDE',
            'LATITUDE'
        ]
        
        X = X[feature_columns]
        
        # Scale features
        if training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return X_scaled
    
    def train(self, df):
        """Train the model"""
        # Prepare features
        X = self.prepare_features(df, training=True)
        y = df['TARGET(PRICE_IN_LACS)']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\nModel Performance:")
        print(f"Training RMSE: {train_rmse:.2f} lakhs")
        print(f"Testing RMSE: {test_rmse:.2f} lakhs")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        # Save model and preprocessors
        self.save_model()
        
        return test_r2
    
    def predict(self, df):
        """Make predictions"""
        X = self.prepare_features(df, training=False)
        return self.model.predict(X)
    
    def save_model(self):
        """Save model and preprocessors"""
        joblib.dump(self.model, 'data/model.joblib')
        joblib.dump(self.scaler, 'data/scaler.joblib')
        joblib.dump(self.label_encoders, 'data/label_encoders.joblib')
    
    @classmethod
    def load_model(cls):
        """Load saved model"""
        instance = cls()
        instance.model = joblib.load('data/model.joblib')
        instance.scaler = joblib.load('data/scaler.joblib')
        instance.label_encoders = joblib.load('data/label_encoders.joblib')
        return instance

if __name__ == "__main__":
    # Load and process data
    from data_processing import load_data, process_data
    
    # Load and process data
    print("Loading and processing data...")
    df = load_data()
    processed_df = process_data(df)
    
    # Train model
    print("\nTraining model...")
    model = HousePriceModel()
    test_score = model.train(processed_df)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': ['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.',
                   'BHK_OR_RK', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE',
                   'ADDRESS', 'LONGITUDE', 'LATITUDE'],
        'importance': model.model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
