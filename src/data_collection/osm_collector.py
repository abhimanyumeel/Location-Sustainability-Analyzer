import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
from datetime import datetime
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OSMDataCollector:
    def __init__(self):
        """Initialize the OSM data collector"""
        self.base_url = "https://overpass-api.de/api/interpreter"
        self.data_dir = Path("data/sustainability/raw/osm")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define sustainability-related tags
        self.sustainability_tags = {
            'green_space': [
                'leisure=park',
                'landuse=recreation_ground',
                'natural=wood',
                'landuse=forest'
            ],
            'water_bodies': [
                'water=*',
                'natural=water',
                'waterway=*'
            ],
            'sustainable_transport': [
                'public_transport=station',
                'railway=station',
                'amenity=bus_station',
                'amenity=bicycle_rental'
            ],
            'waste_management': [
                'amenity=recycling',
                'amenity=waste_basket',
                'amenity=waste_disposal'
            ]
        }

    def fetch_location_data(self, address: str, lat: float, lon: float, radius: float = 5000) -> Dict:
        """
        Fetch sustainability data for a specific location
        """
        logger.info(f"Fetching data for location: {address} ({lat}, {lon})")
        
        # Create query for all sustainability features
        query = f"""
        [out:json][timeout:25];
        (
            // Green Spaces
            {''.join(f'node[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['green_space'])}
            {''.join(f'way[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['green_space'])}
            
            // Water Bodies
            {''.join(f'node[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['water_bodies'])}
            {''.join(f'way[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['water_bodies'])}
            
            // Sustainable Transport
            {''.join(f'node[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['sustainable_transport'])}
            {''.join(f'way[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['sustainable_transport'])}
            
            // Waste Management
            {''.join(f'node[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['waste_management'])}
            {''.join(f'way[{tag}](around:{radius},{lat},{lon});' for tag in self.sustainability_tags['waste_management'])}
        );
        out body;
        >;
        out skel qt;
        """
        
        try:
            response = requests.post(
                self.base_url,
                data=query,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Save raw data
            self._save_raw_data(address, data)
            
            # Process the data
            metrics = self.process_osm_data(data)
            
            # Add location information
            metrics['location'] = {
                'address': address,
                'latitude': lat,
                'longitude': lon,
                'radius': radius
            }
            
            return metrics
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {address}: {e}")
            return None

    def _save_raw_data(self, address: str, data: Dict) -> None:
        """Save raw OSM data to file"""
        # Create a safe filename from the address
        safe_filename = "".join(c for c in address if c.isalnum() or c in (' ', '-', '_')).rstrip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_filename}_{timestamp}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved raw data to {filepath}")

    def process_osm_data(self, data: Dict) -> Dict:
        """Process raw OSM data into meaningful metrics"""
        elements = data.get('elements', [])
        
        metrics = {
            'green_space': {
                'count': len([e for e in elements if any(tag in e.get('tags', {}) 
                    for tag in self.sustainability_tags['green_space'])]),
                'types': self._count_types(elements, self.sustainability_tags['green_space'])
            },
            'water_bodies': {
                'count': len([e for e in elements if any(tag in e.get('tags', {}) 
                    for tag in self.sustainability_tags['water_bodies'])]),
                'types': self._count_types(elements, self.sustainability_tags['water_bodies'])
            },
            'sustainable_transport': {
                'count': len([e for e in elements if any(tag in e.get('tags', {}) 
                    for tag in self.sustainability_tags['sustainable_transport'])]),
                'types': self._count_types(elements, self.sustainability_tags['sustainable_transport'])
            },
            'waste_management': {
                'count': len([e for e in elements if any(tag in e.get('tags', {}) 
                    for tag in self.sustainability_tags['waste_management'])]),
                'types': self._count_types(elements, self.sustainability_tags['waste_management'])
            }
        }
        
        return metrics

    def _count_types(self, elements: List[Dict], tags: List[str]) -> Dict[str, int]:
        """Count different types of features"""
        types = {}
        for element in elements:
            element_tags = element.get('tags', {})
            for tag in tags:
                if tag in element_tags:
                    types[tag] = types.get(tag, 0) + 1
        return types

def main():
    """Test the OSM data collector with sample locations"""
    collector = OSMDataCollector()
    
    # Test with a sample location from our dataset
    test_location = {
        'address': 'Ksfc Layout, Bangalore',
        'lat': 12.9716,
        'lon': 77.5946
    }
    
    metrics = collector.fetch_location_data(
        test_location['address'],
        test_location['lat'],
        test_location['lon']
    )
    
    if metrics:
        logger.info(f"\nMetrics for {test_location['address']}:")
        for category, data in metrics.items():
            if category != 'location':
                logger.info(f"\n{category}:")
                for key, value in data.items():
                    logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
