"""
Streamlit App for Flight Delay Analysis & Prediction

This app loads saved models from `models/` and displays visualizations from
`images/`. It supports a dashboard with operational insights and a prediction
page for arrival delay estimates.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List

# Add parent directory to path to import model_utils
sys.path.append(str(Path(__file__).parent))

from model_utils import load_model, predict_with_preprocessing


def load_chart(image_path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout()
    return fig


def get_chart_cards(image_dir: Path) -> List[Dict[str, str]]:
    """Get all available charts from the images directory."""
    if not image_dir.exists():
        return []
    
    charts = []
    for image_file in sorted(image_dir.glob("*.png")):
        title = image_file.stem.replace("_", " ").title()
        charts.append({
            "path": image_file,
            "title": title,
            "caption": f"Chart: {image_file.name}",
        })
    return charts


def get_airline_codes_to_names() -> Dict[str, str]:
    """Map airline codes to full names."""
    return {
        "AA": "American Airlines",
        "DL": "Delta Air Lines",
        "UA": "United Airlines",
        "WN": "Southwest Airlines",
        "AS": "Alaska Airlines",
        "B6": "JetBlue Airways",
        "F9": "Frontier Airlines",
        "NK": "Spirit Airlines",
        "Other": "Other Carriers",
    }


def derive_model_inputs(user_inputs: Dict[str, object]) -> Dict[str, float]:
    departure_hour = int(user_inputs["CRSDepTime"] // 100)
    day_of_week_zero_index = max(0, min(6, user_inputs["DayOfWeek"] - 1))
    is_weekend = 1 if day_of_week_zero_index >= 5 else 0
    is_rush_hour = 1 if (6 <= departure_hour <= 9 or 16 <= departure_hour <= 19) else 0
    crs_elapsed_time = max(30, int(user_inputs["Distance"] / 500 * 60))
    dep_delay = max(0.0, (user_inputs["TaxiOut"] - 12) * 0.75)

    airline_freq_map = {
        "AA": 0.06,
        "DL": 0.06,
        "UA": 0.05,
        "WN": 0.05,
        "AS": 0.04,
        "B6": 0.03,
        "F9": 0.03,
        "NK": 0.03,
        "Other": 0.02,
    }
    airline_code = user_inputs["Airline"].split(" ")[0] if " " in user_inputs["Airline"] else user_inputs["Airline"]
    airport_freq = airline_freq_map.get(airline_code, 0.02)

    return {
        "DepDelay": float(dep_delay),
        "CRSElapsedTime": float(crs_elapsed_time),
        "Distance": float(user_inputs["Distance"]),
        "DepHour": float(departure_hour),
        "IsWeekend": float(is_weekend),
        "IsRushHour": float(is_rush_hour),
        "Origin_Freq": float(airport_freq),
        "Dest_Freq": float(airport_freq),
        "Month": float(user_inputs["Month"]),
        "DayOfWeek": float(day_of_week_zero_index),
    }


def build_dashboard(image_dir: Path) -> None:
    st.markdown("### Dashboard")
    st.markdown(
        "This dashboard surfaces operational insights, schedule relationships, and model performance visuals."
    )

    cards = get_chart_cards(image_dir)
    for i in range(0, len(cards), 2):
        cols = st.columns(2)
        for idx, card in enumerate(cards[i : i + 2]):
            with cols[idx]:
                if card["path"].exists():
                    st.subheader(card["title"])
                    st.pyplot(load_chart(card["path"]))
                    st.caption(card["caption"])
                else:
                    st.warning(f"Missing chart: {card['path'].name}")

    st.markdown("---")
    st.subheader("Operational Insights")
    st.markdown(
        "- Flights are more likely to arrive late during peak hours and in winter months.  \
"
        "- Airline network frequencies, airport congestion, and taxi-out time are key operational factors.  \
"
        "- Machine learning can predict delays with pre-departure information, but performance depends on feature quality."
    )

    st.subheader("Feature Importance")
    st.markdown(
        "Use the feature importance chart to identify which scheduling and operational factors most strongly influence delays."
    )


def build_prediction_page(models_dir: Path) -> None:
    st.markdown("### Prediction")
    st.markdown(
        "Use pre-departure inputs to estimate arrival delay on saved models. The page includes Month, DayOfWeek, CRSDepTime, Distance, TaxiOut, and Airline inputs."
    )

    model_files = sorted(models_dir.glob("*.joblib"))
    if not model_files:
        st.error("No model files found in the models/ folder. Please save a model first.")
        return

    model_name_mapping = {}
    for file in model_files:
        display_name = file.stem.replace("_", " ").replace("arrdelay", "").strip()
        if "linear" in display_name.lower():
            display_name = "Linear Regression"
        elif "random" in display_name.lower() or "forest" in display_name.lower():
            display_name = "Random Forest Regression"
        model_name_mapping[display_name] = file.name
    
    model_label = st.selectbox(
        "Choose a saved model",
        options=list(model_name_mapping.keys()),
        key="model_selection"
    )
    selected_model_file = models_dir / model_name_mapping[model_label]

    try:
        model_package = load_model(selected_model_file)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    st.markdown("---")
    st.subheader("Flight Inputs")
    month = st.select_slider("Month", options=list(range(1, 13)), value=6)
    day_of_week = st.select_slider(
        "Day of Week",
        options=list(range(1, 8)),
        value=3,
        format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x - 1],
    )
    crs_dep_time = st.number_input(
        "Scheduled Departure Time (HHMM)",
        min_value=0,
        max_value=2359,
        value=1200,
        step=5,
        help="Example: 0500 = 5:00 AM, 1830 = 6:30 PM",
    )
    distance = st.slider("Distance (miles)", min_value=50, max_value=3000, value=800, step=25)
    taxi_out = st.slider("Taxi Out (minutes)", min_value=0, max_value=90, value=12, step=1)
    
    airline_names_dict = get_airline_codes_to_names()
    airline_options = list(airline_names_dict.values())
    selected_airline_name = st.selectbox("Airline", options=airline_options)
    airline = [k for k, v in airline_names_dict.items() if v == selected_airline_name][0]

    st.markdown("---")
    st.subheader("Selected Inputs")
    st.write(
        {
            "Month": month,
            "DayOfWeek": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week - 1],
            "CRSDepTime": crs_dep_time,
            "Distance": distance,
            "TaxiOut": taxi_out,
            "Airline": airline,
        }
    )

    derived_inputs = derive_model_inputs(
        {
            "Month": month,
            "DayOfWeek": day_of_week,
            "CRSDepTime": crs_dep_time,
            "Distance": distance,
            "TaxiOut": taxi_out,
            "Airline": airline,
        }
    )

    with st.expander("Derived model inputs", expanded=True):
        st.write(derived_inputs)

    if st.button("Predict Arrival Delay"):
        try:
            feature_names = model_package["feature_names"]
            prediction_input = {
                feature: derived_inputs.get(feature, 0.0) for feature in feature_names
            }
            prediction = predict_with_preprocessing(model_package, prediction_input, feature_names)
            st.metric("Predicted Arrival Delay (minutes)", f"{prediction:.1f}")
            if prediction > 0:
                st.warning(f"Expected delay of {prediction:.1f} minutes")
            else:
                st.success(f"Expected arrival {abs(prediction):.1f} minutes early")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    st.subheader("Model Notes")
    st.markdown(
        "- The selected saved model may use a different exact feature schema than the inputs shown.  \
"
        "- Airport frequency and departure delay are estimated proxies for the chosen airline and taxi-out time.  \
"
        "- For best results, train a model using the same input variables as this app."
    )
    st.markdown("### Run this app")
    st.code("streamlit run app/streamlit_app.py")


def main() -> None:
    st.set_page_config(
        page_title="Flight Delay Analysis & Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("✈️ Flight Delay Analysis & Prediction")
    st.markdown(
        "This app combines a visualization dashboard with a prediction tool for arrival delays. "
        "Select a page from the sidebar and load your saved model from the `models/` directory."
    )

    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    image_dir = base_dir / "images"

    page = st.sidebar.radio("Navigation", ["Dashboard", "Prediction"], key="page_navigation")

    if page == "Dashboard":
        build_dashboard(image_dir)
    else:
        build_prediction_page(models_dir)


if __name__ == "__main__":
    main()