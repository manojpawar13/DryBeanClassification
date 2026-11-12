
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🫘",
    layout="wide"
)

# Application title
st.title("🫘 Dry Bean Type Classification")

st.write(
    "Predict the type of dry bean using its "
    "physical and geometric measurements."
)

# Load trained model
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "dry_bean_svm_pipeline.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Trained model file not found.")
    st.stop()

model = load_model()

# Load default measurement values
DEFAULTS_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "feature_defaults.pkl"
)

@st.cache_resource
def load_defaults():
    return joblib.load(DEFAULTS_PATH)

if not DEFAULTS_PATH.exists():
    st.error("Feature defaults file not found.")
    st.stop()

feature_defaults = load_defaults()

st.success("Trained SVM model loaded successfully!")

st.info(
    "Model: Tuned Support Vector Machine (RBF Kernel)"
)

st.write(
    "Model Test Accuracy: 92.36%"
)


# Bean measurement input form
st.header("Enter Bean Measurements")

# Preserve the exact feature names and order used in training
feature_names = list(model.feature_names_in_)

with st.form("bean_input_form"):

    col1, col2 = st.columns(2)

    input_values = {}

    for index, feature in enumerate(feature_names):

        with col1 if index % 2 == 0 else col2:
            input_values[feature] = st.number_input(
    label=feature,
    min_value=0.0,
    value=float(feature_defaults[feature]),
    format="%.6f"
)

    predict_button = st.form_submit_button(
        "Predict Bean Type",
        type="primary"
    )


if predict_button:

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    # Validate input measurements
    if (input_df <= 0).any().any():
        st.error(
            "All bean measurements must be greater than zero."
        )

    
    else:
        # Generate prediction
        prediction = model.predict(input_df)[0]

        # Get class probabilities
        probabilities = model.predict_proba(input_df)[0]

        # Create probability DataFrame
        probability_df = pd.DataFrame({
            "Bean Type": model.classes_,
            "Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            by="Probability",
            ascending=False
        )

        # Probability of the predicted class
        predicted_probability = probability_df.loc[
            probability_df["Bean Type"] == prediction,
            "Probability"
        ].iloc[0]

        # Display prediction
        st.divider()
        st.subheader("Prediction Result")

        st.success(f"Predicted Bean Type: {prediction}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Identified Bean Variety",
                value=prediction
            )

        with col2:
            st.metric(
                label="Estimated Probability",
                value=f"{predicted_probability * 100:.2f}%"
            )

        # Display probability chart
        st.subheader("Class Probability Comparison")

        st.bar_chart(
            probability_df.set_index("Bean Type")[
                "Probability"
            ]
        )

        # Display detailed probabilities
        st.subheader("Detailed Class Probabilities")

        st.dataframe(
            probability_df,
            hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability",
                    format="percent",
                    min_value=0,
                    max_value=1
                )
            },
            use_container_width=True
        )

        st.caption(
            "Probabilities are model estimates and "
            "do not guarantee prediction correctness."
        )


