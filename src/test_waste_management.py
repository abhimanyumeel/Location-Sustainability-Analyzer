import pandas as pd
from waste_management import WasteManagementMetrics

def test_waste_management():
    # Initialize the metrics calculator
    waste_metrics = WasteManagementMetrics()
    
    # Test with a few sample cities
    test_cities = [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Chennai",
        "Hyderabad",
        "Jaipur",
        "Kolkata",
        "Lucknow",
        "Ahmedabad",
        "Surat",
        "Pune",
        "Indore"
    ]
    
    print("\nWaste Management Metrics Test Results:")
    print("-" * 50)
    
    for city in test_cities:
        # Get metrics for the city
        metrics = waste_metrics.get_city_metrics(city)
        score, rating, details = waste_metrics.calculate_waste_score(city)
        
        print(f"\nCity: {city}")
        print(f"Municipal Efficiency: {details['municipal_efficiency']:.1f}/10")
        print(f"Recycling Rate: {details['recycling_rate']:.1f}%")
        print(f"Population Density: {details['population_density']:.0f} people/km²")
        print(f"Waste Generated: {details['waste_generated']:.1f} tons/day")
        print(f"Final Score: {score:.1f}/10")
        print(f"Rating: {rating}")
        print("-" * 30)

if __name__ == "__main__":
    print("Testing Waste Management Implementation...")
    test_waste_management() 