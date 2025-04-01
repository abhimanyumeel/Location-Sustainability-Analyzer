import unittest
from src.sustainability_data import SustainabilityDataProcessor
import json
from pathlib import Path

class TestSustainabilityDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = SustainabilityDataProcessor()
        # Create test data directory
        self.test_dir = Path('test_data')
        self.test_dir.mkdir(exist_ok=True)
        
    def test_fetch_osm_data(self):
        """Test if we can fetch data from OpenStreetMap"""
        lat, lon = 12.9716, 77.5946  # Bangalore coordinates
        data = self.processor.fetch_osm_data(lat, lon)
        
        # Check if we got valid JSON response
        self.assertIsInstance(data, dict)
        self.assertIn('elements', data)
        
        # Save the response for inspection
        test_file = self.test_dir / 'test_osm_data.json'
        with open(test_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def test_calculate_sustainability_metrics(self):
        """Test sustainability metrics calculation"""
        # Load test data
        test_file = self.test_dir / 'test_osm_data.json'
        if not test_file.exists():
            self.test_fetch_osm_data()  # Generate test data if it doesn't exist
            
        with open(test_file, 'r') as f:
            osm_data = json.load(f)
            
        lat, lon = 12.9716, 77.5946
        metrics = self.processor.calculate_sustainability_metrics(osm_data, lat, lon)
        
        # Check if all required metrics are present
        required_metrics = ['transport_score', 'green_space_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertGreaterEqual(metrics[metric], 0)
            
    def test_process_location(self):
        """Test complete location processing"""
        lat, lon = 12.9716, 77.5946
        address = "Ksfc Layout, Bangalore"
        metrics = self.processor.process_location(lat, lon, address)
        
        # Check if all metrics are present
        required_metrics = [
            'air_quality', 'air_quality_details', 'water_availability',
            'waste_management', 'osm_metrics', 'sustainability_score'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            if metric == 'osm_metrics':
                self.assertIn('green_space', metrics[metric])
                self.assertIn('sustainable_transport', metrics[metric])
                self.assertIn('count', metrics[metric]['green_space'])
                self.assertIn('count', metrics[metric]['sustainable_transport'])
            elif isinstance(metrics[metric], (int, float)):
                value = float(metrics[metric])
                self.assertGreaterEqual(value, 0)
            
        # Check sustainability score range
        sustainability = float(metrics['sustainability_score'])
        self.assertLessEqual(sustainability, 100)
        self.assertGreaterEqual(sustainability, 0)
        
    def test_multiple_locations(self):
        """Test processing multiple locations"""
        print("\nTest processing multiple locations ...")
        
        # Test locations with addresses
        locations = [
            (12.9716, 77.5946, "Ksfc Layout, Bangalore"),
            (19.0760, 72.8777, "Bandra West, Mumbai"),
            (28.6139, 77.2090, "Connaught Place, Delhi")
        ]
        
        for lat, lon, address in locations:
            metrics = self.processor.process_location(lat, lon, address)
            print(f"\nMetrics for location ({lat}, {lon}):")
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"{key}: {value}")  # Print dictionary as is
                else:
                    print(f"{key}: {float(value):.2f}")  # Convert to float only if not a dict

    def test_air_quality_data(self):
        """Test air quality data fetching from OpenWeatherMap"""
        print("\nTesting air quality data...")
        
        # Test locations (latitude, longitude)
        locations = [
            (12.9716, 77.5946),  # Bangalore
            (19.0760, 72.8777),  # Mumbai
            (28.6139, 77.2090)   # Delhi
        ]
        
        for lat, lon in locations:
            print(f"\nTesting location ({lat}, {lon}):")
            
            # Test basic air quality score
            score = self.processor.get_air_quality_data(lat, lon)
            print(f"Air Quality Score: {score:.2f}")
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
            # Test detailed air quality information
            details = self.processor.get_air_quality_details(lat, lon)
            print("\nAir Quality Details:")
            for key, value in details.items():
                print(f"{key}: {value}")
            
            self.assertIsInstance(details, dict)
            self.assertIn('aqi', details)
            self.assertIn('pm25', details)
            self.assertIn('unit', details)
            self.assertIn('last_updated', details)
            self.assertIn('location', details)
            
            # Verify AQI range (OpenWeatherMap uses 1-5 scale)
            self.assertGreaterEqual(details['aqi'], 1)
            self.assertLessEqual(details['aqi'], 5)

def main():
    # Run tests
    unittest.main(verbosity=2)
    
    # Run additional location tests
    processor = SustainabilityDataProcessor()
    print("\nTesting multiple locations...")
    processor.test_multiple_locations()

if __name__ == '__main__':
    main()