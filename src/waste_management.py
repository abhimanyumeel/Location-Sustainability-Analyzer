import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path

class WasteManagementMetrics:
    def __init__(self, data_path: str = "data/raw/waste_management/waste_management_data.csv"):
        """Initialize the WasteManagementMetrics class with the dataset path"""
        self.data_path = Path(data_path)
        self._load_data()
        
    def _load_data(self) -> None:
        """Load the waste management dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            # Get the latest year's data for each city
            self.data = self.data.sort_values('Year', ascending=False).groupby('City/District').first().reset_index()
        except FileNotFoundError:
            print(f"Warning: Waste management data file not found at {self.data_path}")
            self.data = None
            
    def get_city_metrics(self, city: str) -> Dict[str, float]:
        """Get waste management metrics for a specific city"""
        if self.data is None:
            return self._get_default_metrics()
            
        # Clean and standardize the city name
        city = city.lower().strip()
        
        # Try exact match first
        city_data = self.data[self.data['City/District'].str.lower() == city]
        
        # If no exact match, try to find the city name within the address
        if len(city_data) == 0:
            for index, row in self.data.iterrows():
                if row['City/District'].lower() in city:
                    city_data = self.data.iloc[[index]]
                    break
        
        # If still no match, try to find if the address contains any of our cities
        if len(city_data) == 0:
            for index, row in self.data.iterrows():
                if city in row['City/District'].lower():
                    city_data = self.data.iloc[[index]]
                    break
        
        if len(city_data) == 0:
            print(f"No waste management data found for city: {city}")
            return self._get_default_metrics()
            
        metrics = {
            'municipal_efficiency': float(city_data['Municipal Efficiency Score (1-10)'].iloc[0]),
            'recycling_rate': float(city_data['Recycling Rate (%)'].iloc[0]) / 100,
            'population_density': float(city_data['Population Density (People/km²)'].iloc[0]),
            'waste_generated': float(city_data['Waste Generated (Tons/Day)'].iloc[0])
        }
        
        return metrics
        
    def calculate_waste_score(self, city: str) -> Tuple[float, str, Dict[str, float]]:
        """Calculate the waste management score for a location"""
        metrics = self.get_city_metrics(city)
        
        # Base score from municipal efficiency (50% weight)
        base_score = metrics['municipal_efficiency'] * 0.5
        
        # Recycling rate contribution (20% weight)
        recycling_score = min(metrics['recycling_rate'] * 10, 10) * 0.2
        
        # Population density impact (15% weight)
        # Lower score for very high density areas due to waste management challenges
        density_factor = np.clip(1 - (metrics['population_density'] / 50000), 0, 1)
        density_score = density_factor * 10 * 0.15
        
        # Waste generation impact (15% weight)
        # Lower score for high waste generation per capita
        waste_score = (1 - min(metrics['waste_generated'] / 1000, 1)) * 10 * 0.15
        
        final_score = base_score + recycling_score + density_score + waste_score
        
        # Clip final score between 1 and 10
        final_score = np.clip(final_score, 1, 10)
        
        # Generate rating based on score
        if final_score >= 8:
            rating = "Excellent"
        elif final_score >= 6:
            rating = "Good"
        elif final_score >= 4:
            rating = "Fair"
        else:
            rating = "Poor"
            
        # Return the score, rating, and detailed metrics
        return final_score, rating, {
            'municipal_efficiency': metrics['municipal_efficiency'],
            'recycling_rate': metrics['recycling_rate'] * 100,  # Convert to percentage
            'population_density': metrics['population_density'],
            'waste_generated': metrics['waste_generated']
        }
        
    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when city data is not available"""
        return {
            'municipal_efficiency': 5.0,  # Median score
            'recycling_rate': 0.3,       # 30% recycling rate
            'population_density': 10000,  # Average urban density
            'waste_generated': 500       # Average waste generation
        } 