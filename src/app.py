import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from model import HousePriceModel
from data_processing import load_data
from sustainability_data import SustainabilityDataProcessor
from typing import Dict

# Load training data to get unique values
train_df = load_data()

# Set page config
st.set_page_config(
    page_title="Indian Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF9933;  /* Indian flag color */
        color: white;
        border: none;
        padding: 15px 24px;
        border-radius: 8px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #138808;  /* Indian flag green */
        transform: translateY(-2px);
    }
    .stSelectbox, .stSlider, .stNumberInput {
        margin-bottom: 1.5rem;
    }
    .success {
        color: #138808;
    }
    .warning {
        color: #FF9933;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_trained_model():
    return HousePriceModel.load_model()

@st.cache_resource
def load_sustainability_processor():
    return SustainabilityDataProcessor()

def create_map(lat, lon, location_name):
    # Swap coordinates if needed
    if lat > 68:
        lat, lon = lon, lat
        
    # Create map with modern style and adjusted dimensions
    m = folium.Map(
        location=[lat, lon],
        zoom_start=13,
        tiles='CartoDB positron',  # Restored original style
        width=700,
        height=450,
        control_scale=True
    )
    
    # Add a modern marker with custom styling
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f'<div style="width: 200px; text-align: center;">'
            f'<h4 style="color: #1E1E1E; margin: 0;">{location_name}</h4>'
            f'<p style="color: #666; margin: 5px 0;">Selected Property Location</p>'
            '</div>',
            max_width=300
        ),
        tooltip=location_name,
        icon=folium.Icon(
            color='red',
            icon='home',
            prefix='fa'
        )
    ).add_to(m)
    
    # Add a circle for area highlight
    folium.Circle(
        location=[lat, lon],
        radius=500,
        color='#FF9933',
        fill=True,
        fill_color='#FF9933',
        fill_opacity=0.2,
        weight=2,
        popup='500m radius'
    ).add_to(m)
    
    # Add alternative map styles
    folium.TileLayer(
        'CartoDB dark_matter',
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="http://cartodb.com/attributions">CartoDB</a>',
        name='Dark Mode'
    ).add_to(m)
    
    folium.TileLayer(
        'OpenStreetMap',
        attr='&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        name='Street View'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_price_distribution(prediction, area_prices):
    fig = go.Figure()
    
    # Add area price distribution
    fig.add_trace(go.Histogram(
        x=area_prices,
        name='Area Prices',
        opacity=0.7,
        nbinsx=30
    ))
    
    # Add predicted price line
    fig.add_vline(
        x=prediction,
        line_dash="dash",
        line_color="red",
        annotation_text="Predicted Price"
    )
    
    fig.update_layout(
        title='Price Distribution in Selected Area',
        xaxis_title='Price (Lakhs)',
        yaxis_title='Number of Properties',
        showlegend=True
    )
    return fig

def create_sustainability_radar(metrics: Dict) -> go.Figure:
    """
    Create a radar chart for sustainability metrics
    """
    # Define the metrics to display
    categories = [
        'Transport', 'Green Space', 'Air Quality', 'Water',
        'Waste Management'
    ]
    
    # Get the values for each category
    values = [
        metrics['osm_metrics']['sustainable_transport']['count'] * 5,  # Normalize to 100
        metrics['osm_metrics']['green_space']['count'] * 10,  # Normalize to 100
        metrics['air_quality'] * 100,
        metrics['water_availability'] * 100,
        metrics['waste_management'] * 100
    ]
    
    # Create the radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Sustainability Metrics',
        line=dict(color='#FF9933'),
        fillcolor='rgba(255, 153, 51, 0.2)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color='#ffffff'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(color='#ffffff'),
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        showlegend=False,
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        title=dict(
            text='Sustainability Metrics Overview',
            font=dict(
                size=20,
                color='#FF9933'
            ),
            x=0.5,
            y=0.95
        ),
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    return fig

def get_sustainability_metrics(lat, lon, address):
    try:
        processor = load_sustainability_processor()
        print(f"Calculating metrics for coordinates: {lat}, {lon}")  # Debug log
        metrics = processor.process_location(lat, lon, address)
        print(f"Metrics calculated successfully: {metrics}")  # Debug log
        return metrics
    except Exception as e:
        st.error(f"Error calculating sustainability metrics: {e}")
        print(f"Error details: {str(e)}")  # Debug log
        return None

def main():
    # Sidebar
    with st.sidebar:
        st.title("🏠 Indian Housing Price Predictor")
        st.write("Predict house prices in Indian cities")
        st.markdown("---")
        st.markdown("""
        ### How to use:
        1. Select location
        2. Enter property details
        3. Choose additional features
        4. Click 'Predict Price'
        """)
        st.markdown("---")
        st.markdown("### About:")
        st.write("This app predicts house prices in Indian cities based on various features including location, size, and amenities.")

    # Main content
    st.title("Indian Housing Price Predictor")
    st.write("Predict house prices based on location and property details")

    # Create three columns for input
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Location")
        addresses = sorted(train_df['ADDRESS'].unique())
        address = st.selectbox("Select Location", addresses)
        
        try:
            selected_row = train_df[train_df['ADDRESS'] == address].iloc[0]
            latitude = float(selected_row['LATITUDE'])
            longitude = float(selected_row['LONGITUDE'])
            
            if latitude > 68:
                latitude, longitude = longitude, latitude
                
        except (ValueError, TypeError) as e:
            latitude, longitude = 20.5937, 78.9629

    with col2:
        st.subheader("🏠 Property Details")
        bhk_no = st.number_input("Number of BHK", min_value=1, max_value=10, value=2)
        bhk_or_rk = st.selectbox("Type", train_df['BHK_OR_RK'].unique())
        square_ft = st.number_input("Square Feet", min_value=100, max_value=10000, value=1000)
        posted_by = st.selectbox("Posted By", train_df['POSTED_BY'].unique())

    with col3:
        st.subheader("✨ Additional Features")
        under_construction = st.selectbox("Under Construction?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        rera = st.selectbox("RERA Approved?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ready_to_move = st.selectbox("Ready to Move?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        resale = st.selectbox("Resale?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Streamlined map header
    st.markdown("""
        <div style='background-color: #1E1E1E; padding: 0.8rem 1.2rem; border-radius: 8px; margin: 1rem 0; 
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); display: flex; align-items: center;'>
            <h4 style='color: #FF9933; margin: 0; font-size: 1.1rem;'>📍 Location Map</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Centered map container with proper dimensions
    map_container = st.container()
    with map_container:
        # Create columns for better centering
        _, col_map, _ = st.columns([0.1, 0.8, 0.1])
        with col_map:
            m = create_map(latitude, longitude, address)
            folium_static(m, width=680, height=450)  # Adjusted dimensions to fit container

    # Create two columns for buttons
    button_col1, button_col2 = st.columns(2)

    with button_col1:
        # Price prediction button
        if st.button("Predict Price 🎯", key="predict_button"):
            # Create input DataFrame
            input_data = pd.DataFrame({
                'POSTED_BY': [posted_by],
                'UNDER_CONSTRUCTION': [under_construction],
                'RERA': [rera],
                'BHK_NO.': [bhk_no],
                'BHK_OR_RK': [bhk_or_rk],
                'SQUARE_FT': [square_ft],
                'READY_TO_MOVE': [ready_to_move],
                'RESALE': [resale],
                'ADDRESS': [address],
                'LONGITUDE': [longitude],
                'LATITUDE': [latitude]
            })

            # Load model and make prediction
            model = load_trained_model()
            prediction = model.predict(input_data)[0]

            # Get area statistics first
            area_prices = train_df[train_df['ADDRESS'] == address]['TARGET(PRICE_IN_LACS)']
            avg_price = area_prices.mean()
            min_price = area_prices.min()
            max_price = area_prices.max()
            price_per_sqft = (prediction*100000/square_ft)

            # Display results with enhanced styling
            st.markdown("""
                <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px; margin: 2rem 0; 
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <h2 style='color: #FF9933; margin-bottom: 1.5rem; font-size: 1.8rem; display: flex; align-items: center;'>
                        📊 Prediction Results
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                # Enhanced Predicted Price Card
                st.markdown(f"""
                <div style='background-color: #1E1E1E; padding: 2rem; border-radius: 15px; 
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 1rem;'>
                    <h3 style='color: #FF9933; margin: 0; font-size: 1.3rem; margin-bottom: 1rem;'>
                        💰 Predicted Price
                    </h3>
                    <div style='background-color: #2C2C2C; padding: 1.5rem; border-radius: 10px; 
                                border-left: 5px solid #FF9933;'>
                        <h2 style='color: #138808; margin: 0; font-size: 2.5rem; font-weight: bold;'>
                            ₹{prediction:.2f} Lakhs
                        </h2>
                        <p style='color: #888; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                            ₹{price_per_sqft:.2f} per Sq.Ft
                        </p>
                    </div>
                </div>

                <!-- Area Statistics Card -->
                <div style='background-color: #1E1E1E; padding: 2rem; border-radius: 15px; 
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                    <h3 style='color: #FF9933; margin: 0; font-size: 1.3rem; margin-bottom: 1rem;'>
                        📈 Area Statistics
                    </h3>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
                        <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                            <p style='color: #888; margin: 0; font-size: 0.9rem;'>Average</p>
                            <h4 style='color: #fff; margin: 0.5rem 0; font-size: 1.2rem;'>
                                ₹{avg_price:.2f}L
                            </h4>
                        </div>
                        <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                            <p style='color: #888; margin: 0; font-size: 0.9rem;'>Minimum</p>
                            <h4 style='color: #fff; margin: 0.5rem 0; font-size: 1.2rem;'>
                                ₹{min_price:.2f}L
                            </h4>
                        </div>
                        <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                            <p style='color: #888; margin: 0; font-size: 0.9rem;'>Maximum</p>
                            <h4 style='color: #fff; margin: 0.5rem 0; font-size: 1.2rem;'>
                                ₹{max_price:.2f}L
                            </h4>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with result_col2:
                # Enhanced price distribution plot
                fig = create_price_distribution(prediction, area_prices)
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    title_font_size=20,
                    title_font_color='#FF9933',
                    showlegend=True,
                    legend_font_color='#ffffff',
                    xaxis_title_font_color='#ffffff',
                    yaxis_title_font_color='#ffffff',
                    xaxis_tickfont_color='#ffffff',
                    yaxis_tickfont_color='#ffffff'
                )
                st.plotly_chart(fig, use_container_width=True)

    with button_col2:
        # Sustainability button
        if st.button("Calculate Sustainability Score 🌱", key="sustainability_button"):
            with st.spinner("Calculating sustainability metrics..."):
                metrics = get_sustainability_metrics(latitude, longitude, address)
                
                if metrics:
                    # Create two columns for the sustainability metrics
                    sustain_col1, sustain_col2 = st.columns(2)
                    
                    with sustain_col1:
                        # Overall Sustainability Score
                        st.markdown("""
                            <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                                <h3 style='color: #FF9933; margin: 0; font-size: 1.3rem;'>
                                    🌱 Overall Sustainability
                                </h3>
                                <div style='background-color: #2C2C2C; padding: 1.5rem; margin-top: 1rem;
                                        border-radius: 10px; text-align: center;'>
                                    <h2 style='color: #138808; margin: 0; font-size: 2.5rem;'>
                                        {:.1f}/10
                                    </h2>
                                </div>
                            </div>
                        """.format(metrics['sustainability_score']), unsafe_allow_html=True)

                    with sustain_col2:
                        # Individual Metrics
                        st.markdown("""
                            <div style='background-color: #1E1E1E; padding: 1rem; border-radius: 10px;
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                                <h3 style='color: #FF9933; margin: 0; font-size: 1.3rem;'>
                                    📊 Sustainability Metrics
                                </h3>
                                <div style='display: grid; grid-template-columns: repeat(2, 1fr); 
                                        gap: 1rem; margin-top: 1rem;'>
                        """, unsafe_allow_html=True)

                        # Air Quality
                        st.markdown("""
                            <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                                <p style='color: #888; margin: 0;'>Air Quality</p>
                                <h4 style='color: #fff; margin: 0.5rem 0;'>{:.1f}/10</h4>
                            </div>
                        """.format(metrics['air_quality'] * 10), unsafe_allow_html=True)

                        # Transport
                        st.markdown("""
                            <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                                <p style='color: #888; margin: 0;'>Transport</p>
                                <h4 style='color: #fff; margin: 0.5rem 0;'>{:.1f}/10</h4>
                            </div>
                        """.format(min(10, metrics['osm_metrics']['sustainable_transport']['count'] * 1.5)), unsafe_allow_html=True)

                        # Green Space
                        st.markdown("""
                            <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                                <p style='color: #888; margin: 0;'>Green Space</p>
                                <h4 style='color: #fff; margin: 0.5rem 0;'>{:.1f}/10</h4>
                            </div>
                        """.format(min(10, metrics['osm_metrics']['green_space']['count'] * 2)), unsafe_allow_html=True)

                        # Water
                        st.markdown("""
                            <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                                <p style='color: #888; margin: 0;'>Water</p>
                                <h4 style='color: #fff; margin: 0.5rem 0;'>{:.1f}/10</h4>
                            </div>
                        """.format(metrics['water_availability'] * 10), unsafe_allow_html=True)

                        # Waste Management
                        st.markdown("""
                            <div style='background-color: #2C2C2C; padding: 1rem; border-radius: 10px; text-align: center;'>
                                <p style='color: #888; margin: 0;'>Waste</p>
                                <h4 style='color: #fff; margin: 0.5rem 0;'>{:.1f}/10</h4>
                            </div>
                        """.format(metrics['waste_management'] * 10), unsafe_allow_html=True)

                        st.markdown("</div></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()