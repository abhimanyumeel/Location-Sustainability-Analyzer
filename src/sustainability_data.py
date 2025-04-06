import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple
import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import os
from pathlib import Path
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from data_collection.osm_collector import OSMDataCollector
from waste_management import WasteManagementMetrics

# Load environment variables from .env file
load_dotenv()

class SustainabilityDataProcessor:
    def __init__(self):
        """Initialize the processor with API keys"""
        load_dotenv()  # Load environment variables
        self.geolocator = Nominatim(user_agent="housing_sustainability")
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY')  # Change to OpenWeather API key
        if not self.weather_api_key:
            raise ValueError("OpenWeather API key not found in environment variables")
        print(f"Using OpenWeather API key: {self.weather_api_key[:10]}...")
        
        # Initialize OSM collector
        self.osm_collector = OSMDataCollector()
        self.waste_metrics = WasteManagementMetrics()

    def fetch_osm_data(self, lat: float, lon: float, radius: float = 2000) -> Dict:
        """
        Fetch data from OpenStreetMap for a given location with expanded queries
        """
        # Define the query parameters with more comprehensive tags
        query = f"""
        [out:json][timeout:30];
        (
          // Public Transport - expanded
          way["highway"="bus_stop"](around:{radius},{lat},{lon});
          node["highway"="bus_stop"](around:{radius},{lat},{lon});
          node["public_transport"="stop_position"](around:{radius},{lat},{lon});
          node["public_transport"="station"](around:{radius},{lat},{lon});
          way["public_transport"="platform"](around:{radius},{lat},{lon});
          node["railway"="station"](around:{radius},{lat},{lon});
          node["railway"="subway_entrance"](around:{radius},{lat},{lon});
          node["railway"="tram_stop"](around:{radius},{lat},{lon});
          node["amenity"="bus_station"](around:{radius},{lat},{lon});
          way["amenity"="bus_station"](around:{radius},{lat},{lon});
          
          // Green Spaces - expanded
          way["leisure"="park"](around:{radius},{lat},{lon});
          way["leisure"="garden"](around:{radius},{lat},{lon});
          way["landuse"="recreation_ground"](around:{radius},{lat},{lon});
          way["landuse"="grass"](around:{radius},{lat},{lon});
          way["natural"="wood"](around:{radius},{lat},{lon});
          way["natural"="grassland"](around:{radius},{lat},{lon});
          way["leisure"="playground"](around:{radius},{lat},{lon});
          way["landuse"="forest"](around:{radius},{lat},{lon});
          way["natural"="tree_row"](around:{radius},{lat},{lon});
          way["landuse"="greenfield"](around:{radius},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data=query,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Ensure we have a valid response with elements
            if not isinstance(data, dict) or 'elements' not in data:
                print(f"Invalid response format from OSM API: {data}")
                return {'elements': []}
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching OSM data: {e}")
            return {'elements': []}  # Return empty data instead of failing

    def calculate_sustainability_metrics(self, osm_data: Dict, lat: float, lon: float) -> Dict:
        """
        Calculate sustainability metrics from OSM data with more realistic scoring
        """
        metrics = {
            'transport_score': 0,
            'green_space_score': 0,
            'education_score': 0,
            'healthcare_score': 0
        }
        
        # Process transport data (0-1 scale)
        transport_nodes = [node for node in osm_data['elements'] 
                         if 'tags' in node and ('public_transport' in node['tags'] or 
                                              node['tags'].get('railway') == 'station' or
                                              node['tags'].get('amenity') == 'bus_station')]
        metrics['transport_score'] = min(1.0, len(transport_nodes) / 20)  # Normalize to 20 transport points
        
        # Process green spaces (0-1 scale)
        green_spaces = [node for node in osm_data['elements'] 
                       if 'tags' in node and (node['tags'].get('leisure') == 'park' or
                                            node['tags'].get('landuse') == 'recreation_ground')]
        metrics['green_space_score'] = min(1.0, len(green_spaces) / 10)  # Normalize to 10 green spaces
        
        # Process education facilities (0-1 scale)
        schools = [node for node in osm_data['elements'] 
                  if 'tags' in node and (node['tags'].get('amenity') == 'school' or
                                       node['tags'].get('amenity') == 'university')]
        metrics['education_score'] = min(1.0, len(schools) / 8)  # Normalize to 8 educational facilities
        
        # Process healthcare facilities (0-1 scale)
        hospitals = [node for node in osm_data['elements'] 
                    if 'tags' in node and (node['tags'].get('amenity') in ['hospital', 'clinic'])]
        metrics['healthcare_score'] = min(1.0, len(hospitals) / 5)  # Normalize to 5 healthcare facilities
        
        return metrics

    def get_air_quality_data(self, lat: float, lon: float) -> float:
        """
        Get air quality data from OpenWeatherMap API
        Returns normalized AQI score (0-1)
        """
        try:
            url = "https://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key
            }
            response = requests.get(url, params=params)
            
            # Add detailed error logging
            if response.status_code == 401:
                print(f"API Key Error: Please check if your OpenWeather API key is valid and activated")
                print(f"Current API key: {self.weather_api_key[:10]}...")
                return 0.0
            
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('list'):
                print(f"No air quality data found for coordinates ({lat}, {lon})")
                return 0.0
            
            # Get the AQI value (1-5 scale)
            aqi = data['list'][0]['main']['aqi']
            
            # Normalize AQI score (0-1)
            # OpenWeatherMap uses 1-5 scale where 1 is good and 5 is very poor
            normalized_score = max(0.0, min(1.0, 1.0 - ((aqi - 1) / 4)))
            
            return normalized_score
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching air quality data: {e}")
            return 0.0
        except (KeyError, ValueError, IndexError) as e:
            print(f"Error processing air quality data: {e}")
            return 0.0

    def get_air_quality_details(self, lat: float, lon: float) -> Dict:
        """
        Get detailed air quality information from OpenWeatherMap
        """
        try:
            url = "http://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('list'):
                return {
                    'aqi': 0,
                    'pm25': 0,
                    'unit': 'µg/m³',
                    'last_updated': None,
                    'location': None
                }
            
            result = data['list'][0]
            return {
                'aqi': result['main']['aqi'],
                'pm25': result['components'].get('pm2_5', 0),
                'unit': 'µg/m³',
                'last_updated': datetime.fromtimestamp(result['dt']).isoformat(),
                'location': f"{lat}, {lon}"
            }
            
        except Exception as e:
            print(f"Error getting air quality details: {e}")
            return {
                'aqi': 0,
                'pm25': 0,
                'unit': 'µg/m³',
                'last_updated': None,
                'location': None
            }

    def get_water_availability(self, lat: float, lon: float) -> float:
        """
        Calculate water availability score based on multiple factors
        """
        try:
            # Get location details with timeout
            location = self.geolocator.reverse((lat, lon), timeout=10)
            address = location.raw.get('address', {})
            
            # Factors affecting water availability
            factors = {
                'water_bodies': 0.0,
                'groundwater': 0.0,
                'rainfall': 0.0
            }
            
            # Check for water bodies in the area
            water_bodies = [node for node in self.fetch_osm_data(lat, lon)['elements']
                           if 'tags' in node and node['tags'].get('water')]
            factors['water_bodies'] = min(1.0, len(water_bodies) / 5)
            
            # Get rainfall data (placeholder - you can integrate with weather API)
            # For now, using a simple calculation based on latitude
            rainfall_score = 1.0 - abs(lat - 0) / 90  # Higher score for equatorial regions
            factors['rainfall'] = max(0.0, min(1.0, rainfall_score))
            
            # Calculate groundwater score (placeholder)
            # For now, using a simple calculation based on location
            factors['groundwater'] = 0.7  # Placeholder value
            
            # Calculate weighted average
            weights = {'water_bodies': 0.4, 'groundwater': 0.3, 'rainfall': 0.3}
            score = sum(factors[key] * weights[key] for key in weights.keys())
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating water availability: {e}")
            # Return a default score based on latitude
            rainfall_score = 1.0 - abs(lat - 0) / 90
            return max(0.0, min(1.0, rainfall_score))

    def get_waste_management_score(self, lat: float, lon: float) -> float:
        """
        Calculate waste management score based on multiple factors
        """
        try:
            # Get location details
            location = self.geolocator.reverse((lat, lon))
            address = location.raw.get('address', {})
            
            # Factors affecting waste management
            factors = {
                'recycling': 0.0,
                'collection': 0.0,
                'disposal': 0.0
            }
            
            # Check for recycling facilities
            recycling_points = [node for node in self.fetch_osm_data(lat, lon)['elements']
                              if 'tags' in node and node['tags'].get('amenity') == 'recycling']
            factors['recycling'] = min(1.0, len(recycling_points) / 3)
            
            # Check for waste collection points
            waste_collection = [node for node in self.fetch_osm_data(lat, lon)['elements']
                              if 'tags' in node and node['tags'].get('amenity') == 'waste_basket']
            factors['collection'] = min(1.0, len(waste_collection) / 10)
            
            # Placeholder for disposal facilities
            factors['disposal'] = 0.6  # Placeholder value
            
            # Calculate weighted average
            weights = {'recycling': 0.4, 'collection': 0.3, 'disposal': 0.3}
            score = sum(factors[key] * weights[key] for key in weights.keys())
            
            return float(score)
            
        except Exception as e:
            print(f"Error calculating waste management score: {e}")
            return 0.0

    def calculate_economic_sustainability(self, price_in_lakhs: float, yearly_income_in_lakhs: float) -> Dict:
        """
        Calculate economic sustainability based on price-to-income ratio and affordability metrics
        """
        try:
            # Calculate price-to-income ratio (lower is better)
            price_to_income_ratio = price_in_lakhs / yearly_income_in_lakhs
            
            # Calculate affordability score (0-10 scale)
            # A ratio of 5 or less is considered good (score > 7)
            # A ratio of 10 or more is considered poor (score < 4)
            if price_to_income_ratio <= 5:
                affordability_score = 10 - (price_to_income_ratio * 0.6)
            else:
                affordability_score = max(1, 7 - ((price_to_income_ratio - 5) * 0.6))
                
            # Calculate monthly EMI (assuming 20-year loan with 8.5% interest)
            loan_amount = price_in_lakhs * 100000  # Convert to rupees
            interest_rate = 8.5 / 12 / 100  # Monthly interest rate
            loan_term_months = 20 * 12
            emi = (loan_amount * interest_rate * (1 + interest_rate)**loan_term_months) / ((1 + interest_rate)**loan_term_months - 1)
            
            # Calculate EMI to monthly income ratio (should be <= 0.5 for good score)
            monthly_income = (yearly_income_in_lakhs * 100000) / 12
            emi_to_income_ratio = emi / monthly_income
            
            # Calculate EMI affordability score (0-10 scale)
            if emi_to_income_ratio <= 0.3:
                emi_score = 10
            elif emi_to_income_ratio <= 0.5:
                emi_score = 7
            else:
                emi_score = max(1, 7 - ((emi_to_income_ratio - 0.5) * 10))
                
            return {
                'price_to_income_ratio': float(price_to_income_ratio),
                'affordability_score': float(affordability_score),
                'emi_monthly': float(emi),
                'emi_to_income_ratio': float(emi_to_income_ratio),
                'emi_affordability_score': float(emi_score),
                'economic_sustainability_score': float((affordability_score + emi_score) / 2)
            }
            
        except Exception as e:
            print(f"Error calculating economic sustainability: {e}")
            return {
                'price_to_income_ratio': 0.0,
                'affordability_score': 0.0,
                'emi_monthly': 0.0,
                'emi_to_income_ratio': 0.0,
                'emi_affordability_score': 0.0,
                'economic_sustainability_score': 0.0
            }

    def calculate_sustainability_score(self, metrics: Dict, economic_metrics: Dict = None) -> float:
        """
        Calculate overall sustainability score combining environmental and economic factors
        """
        # Environmental scores (out of 10)
        transport_score = min(10, metrics['osm_metrics']['sustainable_transport']['count'] * 1.5)
        green_space_score = min(10, metrics['osm_metrics']['green_space']['count'] * 2)
        air_quality_score = metrics['air_quality'] * 10
        water_score = metrics['water_availability'] * 10
        # Use the actual waste management score from our waste metrics
        waste_score = metrics['waste']['score']
        
        # Calculate environmental score with weights
        env_weights = {
            'transport': 0.2,
            'green_space': 0.2,
            'air_quality': 0.2,
            'water': 0.2,
            'waste': 0.2
        }
        
        environmental_score = (
            transport_score * env_weights['transport'] +
            green_space_score * env_weights['green_space'] +
            air_quality_score * env_weights['air_quality'] +
            water_score * env_weights['water'] +
            waste_score * env_weights['waste']
        )
        
        # Apply environmental penalties
        if metrics.get('air_quality_details', {}).get('aqi', 0) >= 4:
            environmental_score *= 0.9
        
        # If economic metrics are provided, include them in final score
        if economic_metrics:
            economic_score = economic_metrics['economic_sustainability_score']
            
            # Combined score (60% environmental, 40% economic)
            final_score = (environmental_score * 0.6) + (economic_score * 0.4)
        else:
            final_score = environmental_score
        
        return max(1, min(10, final_score))

    def process_location(self, lat: float, lon: float, address: str, price_in_lakhs: float = None, yearly_income_in_lakhs: float = None) -> Dict:
        """
        Process all sustainability data for a given location, including economic factors if provided
        """
        try:
            # Get OSM data
            osm_data = self.fetch_osm_data(lat, lon)
            
            # Process transport facilities
            transport_facilities = [
                node for node in osm_data['elements'] 
                if 'tags' in node and (
                    node['tags'].get('highway') == 'bus_stop' or
                    node['tags'].get('public_transport') in ['stop_position', 'station', 'platform'] or
                    node['tags'].get('railway') in ['station', 'subway_entrance', 'tram_stop'] or
                    node['tags'].get('amenity') == 'bus_station'
                )
            ]
            
            # Count bus stops specifically
            bus_stops = [
                node for node in osm_data['elements']
                if 'tags' in node and node['tags'].get('highway') == 'bus_stop'
            ]
            
            # Process green spaces - look for both nodes and ways
            green_spaces = [
                element for element in osm_data['elements']
                if 'tags' in element and (
                    element['tags'].get('leisure') in ['park', 'garden', 'playground'] or
                    element['tags'].get('landuse') in ['recreation_ground', 'grass', 'forest', 'greenfield'] or
                    element['tags'].get('natural') in ['wood', 'grassland', 'tree_row']
                )
            ]
            
            # Calculate scores with better normalization
            osm_metrics = {
                'green_space': {
                    'count': len(green_spaces),
                    'details': [space.get('tags', {}).get('name', 'Unnamed green space') 
                              for space in green_spaces[:5]]  # Show up to 5 named spaces
                },
                'sustainable_transport': {
                    'count': len(transport_facilities),
                    'bus_stops': len(bus_stops),
                    'details': [facility.get('tags', {}).get('name', 'Unnamed transport facility') 
                              for facility in transport_facilities[:5]]  # Show up to 5 named facilities
                }
            }
            
            # Get air quality data
            air_quality_score = self.get_air_quality_data(lat, lon)
            air_quality_details = self.get_air_quality_details(lat, lon)
            
            # Calculate water availability
            water_availability = self.get_water_availability(lat, lon)
            
            # Calculate waste management metrics first
            city_name = address.split(',')[-1].strip()
            waste_score, waste_rating, waste_details = self.waste_metrics.calculate_waste_score(city_name)
            
            # If we got default metrics, try with the full address
            if waste_details['municipal_efficiency'] == 5.0 and waste_details['recycling_rate'] == 30.0:
                waste_score, waste_rating, waste_details = self.waste_metrics.calculate_waste_score(address)
            
            # Create the metrics dictionary with all components
            metrics = {
                'air_quality': float(air_quality_score),
                'air_quality_details': air_quality_details,
                'water_availability': float(water_availability),
                'waste': {
                    'score': waste_score,
                    'rating': waste_rating,
                    'details': f"Municipal Efficiency: {waste_details['municipal_efficiency']:.1f}/10, Recycling Rate: {waste_details['recycling_rate']:.1f}%, Population Density: {waste_details['population_density']:.0f}/km², Waste Generated: {waste_details['waste_generated']:.1f} tons/day"
                },
                'osm_metrics': osm_metrics
            }
            
            # Add economic metrics if price and income are provided
            if price_in_lakhs is not None and yearly_income_in_lakhs is not None:
                economic_metrics = self.calculate_economic_sustainability(price_in_lakhs, yearly_income_in_lakhs)
                metrics['economic_metrics'] = economic_metrics
                
                # Calculate combined sustainability score
                metrics['sustainability_score'] = float(self.calculate_sustainability_score(metrics, economic_metrics))
                
                # Add affordability rating
                metrics['affordability_rating'] = self.get_affordability_rating(economic_metrics['economic_sustainability_score'])
            else:
                # Calculate sustainability score without economic metrics
                metrics['sustainability_score'] = float(self.calculate_sustainability_score(metrics))
                metrics['economic_metrics'] = None
                metrics['affordability_rating'] = 'Not Available'
            
            return metrics
            
        except Exception as e:
            print(f"Error processing location {address}: {str(e)}")
            # Return a complete dictionary with default values
            return {
                'air_quality': 0.0,
                'air_quality_details': {'aqi': 0, 'pm25': 0, 'unit': 'µg/m³', 'last_updated': None, 'location': None},
                'water_availability': 0.0,
                'waste': {
                    'score': 5.0,
                    'rating': 'Fair',
                    'details': 'Municipal Efficiency: 5.0/10, Recycling Rate: 30.0%, Population Density: 10000/km², Waste Generated: 500.0 tons/day'
                },
                'osm_metrics': {'green_space': {'count': 0}, 'sustainable_transport': {'count': 0}},
                'sustainability_score': 0.0,
                'economic_metrics': None,
                'affordability_rating': 'Not Available'
            }

    def get_affordability_rating(self, economic_score: float) -> str:
        """
        Get affordability rating based on economic sustainability score
        """
        if economic_score >= 9:
            return "Highly Affordable"
        elif economic_score >= 7:
            return "Affordable"
        elif economic_score >= 5:
            return "Moderately Affordable"
        elif economic_score >= 3:
            return "Expensive"
        else:
            return "Very Expensive"

def main():
    # Test the processor
    processor = SustainabilityDataProcessor()
    
    # Test with a sample location (e.g., Bangalore)
    test_location = {
        'address': 'Ksfc Layout, Bangalore',
        'lat': 12.9716,
        'lon': 77.5946
    }
    
    metrics = processor.process_location(
        test_location['lat'],
        test_location['lon'],
        test_location['address']
    )
    
    print("\nSustainability Metrics:")
    for key, value in metrics.items():
        if key == 'osm_metrics':
            print("\nOSM Metrics:")
            for osm_key, osm_value in value.items():
                print(f"  {osm_key}:")
                for sub_key, sub_value in osm_value.items():
                    print(f"    {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 