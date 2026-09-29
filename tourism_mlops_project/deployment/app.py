
import platform
import streamlit as st

from predictor import TourismPredictor

st.set_page_config(
    page_title="Wellness Tourism Purchase Predictor",
    page_icon="✈️",
    layout="wide",
)

@st.cache_resource
def load_predictor():
    return TourismPredictor()

predictor = load_predictor()

st.title("Wellness Tourism Package Purchase Predictor")
st.caption(
    "Predict the probability that a customer will purchase the tourism package "
    "before proactive sales contact."
)

with st.sidebar:
    st.subheader("Deployment")
    st.write("Python:", platform.python_version())
    st.write("Recommended deployment runtime: Python 3.11")

with st.form("tourism_prediction"):
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=90, value=35)
        contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        duration = st.number_input("Duration of Pitch", min_value=0.0, max_value=120.0, value=15.0)
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Male", "Female"])

    with c2:
        persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
        product = st.selectbox(
            "Product Pitched",
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
        )
        stars = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        marital = st.selectbox("Marital Status", ["Married", "Divorced", "Unmarried", "Single"])
        trips = st.number_input("Number of Trips", min_value=0.0, max_value=30.0, value=3.0)

    with c3:
        passport = st.selectbox("Passport", [0, 1])
        satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
        own_car = st.selectbox("Own Car", [0, 1])
        children = st.number_input("Children Visiting", min_value=0.0, max_value=10.0, value=0.0)
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            max_value=500000.0,
            value=25000.0,
            step=1000.0
        )

    threshold = st.slider(
        "Campaign decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05
    )

    submitted = st.form_submit_button("Predict Purchase Propensity")

if submitted:
    payload = {
        "Age": age,
        "TypeofContact": contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": persons,
        "NumberOfFollowups": followups,
        "ProductPitched": product,
        "PreferredPropertyStar": stars,
        "MaritalStatus": marital,
        "NumberOfTrips": trips,
        "Passport": passport,
        "PitchSatisfactionScore": satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": children,
        "Designation": designation,
        "MonthlyIncome": income,
    }

    probability = predictor.predict_probability(payload)

    st.metric("Predicted purchase probability", f"{probability:.1%}")

    if probability >= threshold:
        st.success("Priority lead at the selected threshold.")
    else:
        st.info("Lower-priority lead at the selected threshold.")

    st.caption(
        "This is a propensity score for marketing prioritization, not a causal or "
        "eligibility decision."
    )
