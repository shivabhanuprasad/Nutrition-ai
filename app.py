import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# --- Path Setup (FIXED for Deployment) ---
# This logic correctly finds the project's root directory and adds it to the
# system path, allowing for absolute imports from the 'src' package.
try:
    # Get the absolute path of the directory containing app.py (the project root)
    project_root = Path(__file__).resolve().parent
    # Add the project root to the system path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Now, perform absolute imports from the 'src' package
    from src.Pipeline.predict_pipeline import PredictPipeline
    from src.exception import CustomException

except ImportError as e:
    st.error(f"Import Error: {e}. Please ensure your project structure is correct. The app expects an 'src' directory in the same folder as app.py, containing your modules.")
    st.stop()


# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* App background */
    .stApp {
        background: #000000; /* Black background */
        background-attachment: fixed;
    }

    /* General text color for the app, including labels and paragraphs */
    body, .st-emotion-cache-10trblm, .st-emotion-cache-1kyxreq, .st-emotion-cache-aabc9x, .st-emotion-cache-z5fcl4, label, .st-emotion-cache-16idsys p {
        color: white !important; /* Black for all text */
    }

    /* Main container styling */
    .main .block-container {
        background-color: #ffcc80; /* Light orange main container */
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        max-width: 950px;
        margin: auto;
    }

    /* Headings and titles */
    h1, h2, h3, h4 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: bold;
        color: white !important; /* Black for main headings */
        text-align: center;
    }

    /* Section card styling */
    .section-box {
        background: #ffe0b2; /* Lighter orange/cream for input box */
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-top: 1.5rem;
        border: 1px solid #ffb74d; /* Darker orange border */
    }
    .section-box h2 {
        font-size: 1.75rem;
        margin-bottom: 1.5rem;
        color: #000000 !important; /* Black for section titles */
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #ff7043, #ff5722); /* Coral/orange gradient */
        color: #000000; /* Black text for contrast on button, as requested */
        border-radius: 8px;
        border: 2px solid #000000; /* Black border for definition */
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #f4511e, #e64a19); /* Darker coral/orange */
        transform: scale(1.02);
    }

    /* Styling for dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Custom style for result cards */
    .result-card {
        background-color: #ffe0b2; /* Lighter orange to match section box */
        border-left: 5px solid #ff9800; /* Amber border */
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem; /* Space between cards */
    }
    .result-card p, .result-card strong {
        color: #000000 !important; /* Black text for dish names */
    }
</style>
""", unsafe_allow_html=True)


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


# Load the main dataset
nutri_df = load_data()

if nutri_df is not None:
    # --- App Title and Header ---
    st.title("🍲 NutriPlan AI - Smart Dish Recommender(Indian and Chinese)")
    st.markdown("### Your personal guide to healthy and delicious meals.")

    # --- Input Form Section ---
    with st.container():
        st.markdown('<div class="section-box"> <h2>🥗 Find Your Perfect Dish</h2>', unsafe_allow_html=True)

        cuisine_types = sorted(nutri_df['Cuisine_Type'].unique())
        meal_types = sorted(nutri_df['Meal_Type'].unique())
        dietary_preferences = sorted(nutri_df['Dietary_Preference'].unique())
        occasion_types = sorted(nutri_df['Occasion_Type'].unique())
        all_ingredients_list = get_unique_ingredients(nutri_df)

        col1, col2 = st.columns(2)

        with col1:
            cuisine = st.selectbox("Cuisine Type", cuisine_types)
            meal_type = st.selectbox("Meal Type", meal_types)
            difficulty = st.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
            occasion = st.selectbox("Occasion Type", occasion_types)

        with col2:
            diet = st.selectbox("Dietary Preference", dietary_preferences)
            ingredients_list = st.multiselect(
                "Select Your Available Ingredients",
                all_ingredients_list,
                default=["Rice", "Chicken"]
            )
            ingredients_str = ", ".join(ingredients_list)

        st.markdown("<h4 style='text-align: center; color: #000000; margin-top: 2rem;'>Nutritional Goals</h4>", unsafe_allow_html=True)
        calorie_range = st.slider(
            "Calories per Serving",
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
    if st.button("🍽️ Suggest Dishes"):
        if not ingredients_list:
            st.warning("Please select at least one ingredient.")
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
                
                pipeline = PredictPipeline()
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
                st.error(f"A prediction error occurred: {e}")
                st.session_state.recommendations = None
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                st.session_state.recommendations = None

    # --- Display Results Section ---
    if st.session_state.user_input_display:
        st.subheader("📋 Your Preferences")
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            for key, value in st.session_state.user_input_display.items():
                st.markdown(f"**{key}:** {value}")
            st.markdown('</div>', unsafe_allow_html=True)


    if st.session_state.recommendations is not None:
        st.subheader("🍛 Top Dish Suggestions")
        
        recommendations = st.session_state.recommendations
        if recommendations.empty:
            st.info("No suggestions found for your criteria. Try adjusting the filters.")
        else:
            for index, row in recommendations.iterrows():
                dish_details = nutri_df[nutri_df['Dish_Name'] == row['Dish_Name']].iloc[0]
                with st.container():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"<h4>{dish_details['Dish_Name']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"**Calories:** {dish_details['Calories_per_Serving']} kcal")
                    st.markdown(f"**Cuisine:** {dish_details['Cuisine_Type']}")
                    st.markdown(f"**Ingredients:** {dish_details['Main_Ingredients']}")
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("💪 Want a More Indulgent Option?")
            st.markdown("Select a dish to find a similar, higher-calorie alternative.")

            selected_dish_name = st.selectbox(
                "Choose a dish:",
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
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown(f"**Alternative Suggestion:** {highest_calorie_alt['Dish_Name']}")
                        st.markdown(f"**Calories:** {highest_calorie_alt['Calories_per_Serving']} kcal (Original was {original_calories} kcal)")
                        st.markdown(f"**Cuisine:** {highest_calorie_alt['Cuisine_Type']}")
                        st.markdown(f"**Ingredients:** {highest_calorie_alt['Main_Ingredients']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No higher-calorie alternative with similar ingredients was found in the dataset.")
