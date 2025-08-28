import sys, os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import datetime
from datetime import time
import pytz
import time as t
from pathlib import Path
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException

# Set page configuration
st.set_page_config(
    page_title="NutriPlan AI",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function to load external CSS ---
def load_css(css_file_name):
    try:
        current_dir = os.path.dirname(__file__)
        css_file_path = os.path.join(current_dir, css_file_name)
        with open(css_file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"Error: CSS file '{css_file_name}' not found. Please ensure it is in the same directory as app.py.")
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")

# Load the external CSS file
load_css("style.css")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath='notebook/NutriPlan_AI_Dataset.csv'):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file not found at '{filepath}'. Please ensure it is in the correct directory.")
        return None

@st.cache_data
def get_unique_ingredients(df):
    if df is None or 'Main_Ingredients' not in df.columns:
        return []
    all_ingredients = set()
    for ingredients_list in df['Main_Ingredients'].dropna().str.split(','):
        for ingredient in ingredients_list:
            all_ingredients.add(ingredient.strip())
    return sorted(list(all_ingredients))

# --- Initialize Session State ---
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'user_input_display' not in st.session_state:
    st.session_state.user_input_display = None
if 'last_inputs' not in st.session_state:
    st.session_state.last_inputs = {}
# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Recommender" # Default to the main recommender page

# Load the main dataset
nutri_df = load_data()

# --- Image URLs for Dynamic Backgrounds (Publicly Hosted) ---
# Using reliable publicly hosted URLs from Unsplash for guaranteed display
# --- Image URLs for Dynamic Backgrounds (Healthy Food Theme) ---
recommender_bg_image_url = "https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=1400&q=80"  # Plated colorful meal
about_bg_image_url = "https://images.unsplash.com/photo-1506806732259-39c2d0268443?auto=format&fit=crop&w=1400&q=80"        # Fresh fruits & vegetables



if nutri_df is not None:
    # --- Dynamic Background Image Injection for the BODY element ---
    # This CSS is injected directly onto the 'body' to ensure it's the outermost background.
    # The '!important' flag helps override any default Streamlit body styling.
    current_bg_url = ""
    if st.session_state.current_page == "Recommender":
        current_bg_url = recommender_bg_image_url
    elif st.session_state.current_page == "About":
        current_bg_url = about_bg_image_url
    
    # Only inject if a URL is determined to avoid empty background-image property
    if current_bg_url:
        st.markdown(f"""
            <style>
                body {{
                    background-image: url('{current_bg_url}') !important;
                    background-size: cover !important;
                    background-position: center !important;
                    background-repeat: no-repeat !important;
                    background-attachment: fixed !important;
                }}
            </style>
            """, unsafe_allow_html=True)

    # --- App Title and Header ---
    st.markdown('<h1 class="main-header">🍲 NutriPlan AI - Smart Dish Recommender</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.3em; color: #f0f0f0;'>Your personal guide to healthy and delicious meals.</p>", unsafe_allow_html=True)

    # --- Navigation Buttons ---
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("🍽️ Dish Recommender", use_container_width=True, key="nav_recommender"):
            st.session_state.current_page = "Recommender"
    with col_nav2:
        if st.button("💡 About NutriPlan AI", use_container_width=True, key="nav_about"):
            st.session_state.current_page = "About"
    
    st.markdown("<br>", unsafe_allow_html=True) # Add some spacing

    # --- Page Content based on Navigation ---
    if st.session_state.current_page == "Recommender":
        # --- Input Form Section ---
        with st.container():
            st.markdown('<div class="card"> <h2 class="sub-header">✨ Customize Your Dish Search</h2>', unsafe_allow_html=True)

            cuisine_types = sorted(nutri_df['Cuisine_Type'].unique())
            meal_types = sorted(nutri_df['Meal_Type'].unique())
            dietary_preferences = sorted(nutri_df['Dietary_Preference'].unique())
            occasion_types = sorted(nutri_df['Occasion_Type'].unique())
            all_ingredients_list = get_unique_ingredients(nutri_df)

            col1, col2 = st.columns(2)

            with col1:
                cuisine = st.selectbox("Cuisine Type 🌍", cuisine_types)
                meal_type = st.selectbox("Meal Type 🍽️", meal_types)
                difficulty = st.selectbox("Difficulty Level 🌶️", ["Easy", "Medium", "Hard"])
                occasion = st.selectbox("Occasion Type 🎉", occasion_types)

            with col2:
                diet = st.selectbox("Dietary Preference 🌱", dietary_preferences)
                ingredients_list = st.multiselect(
                    "Select Your Available Ingredients 🥕",
                    all_ingredients_list,
                    default=["Rice", "Chicken"]
                )
                ingredients_str = ", ".join(ingredients_list)

                st.markdown("<h4 style='text-align: center; color: #4ecdc4;'>Nutritional Goals 📊</h4>", unsafe_allow_html=True)
                calorie_range = st.slider(
                    "Calories per Serving �",
                    min_value=int(nutri_df['Calories_per_Serving'].min()),
                    max_value=int(nutri_df['Calories_per_Serving'].max()),
                    value=(150, 700)
                )

            st.markdown('</div>', unsafe_allow_html=True)
        

        # --- State Management Logic ---
        current_inputs = {
            "cuisine": cuisine, "meal_type": meal_type, "difficulty": difficulty,
            "occasion": occasion, "diet": diet, "ingredients": ingredients_str,
            "calories": calorie_range
        }
        
        if current_inputs != st.session_state.last_inputs:
            st.session_state.recommendations = None
            st.session_state.user_input_display = None

        # --- Prediction Logic ---
        if st.button("🍽️ Suggest Dishes", use_container_width=True, key="suggest_dishes_button"):
            if not ingredients_list:
                st.warning("⚠️ Please select at least one ingredient.")
            else:
                try:
                    st.session_state.user_input_display = {
                        "Cuisine Type": cuisine, "Meal Type": meal_type, "Dietary Preference": diet,
                        "Ingredients": ingredients_str, "Difficulty": difficulty, "Occasion": occasion,
                        "Calorie Range": f"{calorie_range[0]} - {calorie_range[1]} kcal"
                    }
                    st.session_state.last_inputs = current_inputs

                    input_data = {
                        "Meal_ID": "dummy_id", "Cuisine_Type": cuisine, "Meal_Type": meal_type,
                        "Dietary_Preference": diet, "Main_Ingredients": ingredients_str,
                        "Difficulty_Level": difficulty, "Occasion_Type": occasion,
                        "Calories_per_Serving": 0, "Protein_Content(g)": 0,
                    }
                    input_df = pd.DataFrame([input_data])
                    
                    pipeline = PredictPipeline()  # Initialize your pipeline here
                    results = pipeline.predict(input_df, top_k=20)
                    
                    if results:
                        results_df = pd.DataFrame(results)
                        if 'Max_Calories' in results_df.columns:
                            results_df = results_df.rename(columns={'Max_Calories': 'Calories_per_Serving'})
                        
                        filtered_results = results_df[
                            results_df['Calories_per_Serving'].between(calorie_range[0], calorie_range[1])
                        ].head(5)
                        
                        st.session_state.recommendations = filtered_results
                    else:
                        st.session_state.recommendations = pd.DataFrame()

                except CustomException as e:
                    st.error(f"❌ A prediction error occurred: {e}")
                    st.session_state.recommendations = None
                except Exception as e:
                    st.error(f"🐛 An unexpected error occurred: {e}")
                    st.session_state.recommendations = None

        # --- Display Results Section ---
        if st.session_state.user_input_display:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">📋 Your Selected Preferences</h2>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                for key, value in st.session_state.user_input_display.items():
                    st.markdown(f"**{key}**: {value}")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.recommendations is not None:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="sub-header">🍛 Top Dish Suggestions For You!</h2>', unsafe_allow_html=True)
            
            recommendations = st.session_state.recommendations
            if recommendations.empty:
                st.info("ℹ️ No suggestions found for your criteria. Try adjusting the filters.")
            else:
                for index, row in recommendations.iterrows():
                    dish_details = nutri_df[nutri_df['Dish_Name'] == row['Dish_Name']].iloc[0]
                    with st.container():
                        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                        st.markdown(f"<h3>{dish_details['Dish_Name']}</h3>", unsafe_allow_html=True)
                        st.markdown(f"**Calories**: {dish_details['Calories_per_Serving']} kcal")
                        st.markdown(f"**Cuisine**: {dish_details['Cuisine_Type']}")
                        st.markdown(f"**Ingredients**: {dish_details['Main_Ingredients']}")
                        st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")
                st.markdown('<h2 class="sub-header">💪 Want a More Indulgent Option?</h2>', unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #f0f0f0;'>Select a dish to find a similar, higher-calorie alternative.</p>", unsafe_allow_html=True)

                selected_dish_name = st.selectbox(
                    "Choose a dish: 😋",
                    options=recommendations['Dish_Name'],
                    key="alternative_selectbox"
                )

                if selected_dish_name:
                    original_dish_row = nutri_df[nutri_df['Dish_Name'] == selected_dish_name].iloc[0]
                    original_calories = original_dish_row['Calories_per_Serving']
                    original_ingredients = set(ing.strip() for ing in original_dish_row['Main_Ingredients'].split(','))

                    def find_alternatives(row):
                        current_ingredients = set(ing.strip() for ing in row['Main_Ingredients'].split(','))
                        if len(original_ingredients.intersection(current_ingredients)) > 0 and row['Calories_per_Serving'] > original_calories:
                            return True
                        return False

                    alternatives_df = nutri_df[nutri_df.apply(find_alternatives, axis=1)]

                    if not alternatives_df.empty:
                        highest_calorie_alt = alternatives_df.sort_values(by='Calories_per_Serving', ascending=False).iloc[0]
                        with st.container():
                            st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                            st.markdown(f"<h3>Alternative Suggestion: {highest_calorie_alt['Dish_Name']}</h3>", unsafe_allow_html=True)
                            st.markdown(f"**Calories**: {highest_calorie_alt['Calories_per_Serving']} kcal (Original was {original_calories} kcal)")
                            st.markdown(f"**Cuisine**: {highest_calorie_alt['Cuisine_Type']}")
                            st.markdown(f"**Ingredients**: {highest_calorie_alt['Main_Ingredients']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No higher-calorie alternative with similar ingredients was found in the dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.current_page == "About":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">💡 About NutriPlan AI</h2>', unsafe_allow_html=True)
        st.markdown("""
            <p style='font-size: 1.1em; color: #f0f0f0; line-height: 1.6;'>
                Welcome to <b>NutriPlan AI</b> 🤖, your intelligent companion for discovering delicious and healthy meals tailored to your preferences!
                Our application leverages advanced data analysis and machine learning to provide personalized dish recommendations based on your selected criteria.
            </p>
            <p style='font-size: 1.1em; color: #f0f0f0; line-height: 1.6;'>
                <h3>How It Works: ⚙️</h3>
                <ul>
                    <li><b>Input Your Preferences:</b> Select your desired cuisine type 🌍, meal type 🍽️, dietary preferences 🌱, difficulty level 🌶️, and available ingredients 🥕.</li>
                    <li><b>Define Nutritional Goals:</b> Use the calorie slider 🔥 to narrow down suggestions that fit your dietary requirements.</li>
                    <li><b>Get Instant Recommendations:</b> Our AI model 🧠 processes your inputs to suggest a list of dishes from our extensive database.</li>
                    <li><b>Explore Alternatives:</b> Found a dish you like but want a more indulgent version? Our app can suggest higher-calorie alternatives with similar ingredients 🤤.</li>
                </ul>
            </p>
            <p style='font-size: 1.1em; color: #f0f0f0; line-height: 1.6;'>
                <h3>Our Mission: 🎯</h3>
                To simplify meal planning and encourage healthy eating habits by making personalized, data-driven food recommendations accessible to everyone.
                Whether you're looking for a quick weeknight dinner 🌙, a special occasion meal 🎉, or simply trying to manage your caloric intake, NutriPlan AI is here to help!
            </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #f0f0f0; font-size: 0.9em;">
    <p>NutriPlan AI - Smart Dish Recommender | Helping you make informed nutritional choices</p>
    <p>© 2023 Data-Driven Health Solutions</p>
</div>
""", unsafe_allow_html=True)