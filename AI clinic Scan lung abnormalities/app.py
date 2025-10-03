import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import altair as alt
import io

CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
MODEL_PATH = 'model.pth'

@st.cache_resource
def load_model(model_path):
    try:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'. Please ensure 'model.pth' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict(model, image: Image.Image, transform):
    try:
        input_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        
        prediction_class = CLASS_NAMES[predicted_index]
        confidence = probabilities[predicted_index].item()
        
        return prediction_class, confidence, probabilities

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "ERROR", 0.0, torch.tensor([0.0, 0.0])

def main():
    st.set_page_config(page_title="Pneumonia Classifier", layout="wide")
    st.title("Chest X-Ray Pneumonia Classification")
    st.markdown("Upload a chest X-ray image to classify it as **NORMAL** or **PNEUMONIA**.")

    model = load_model(MODEL_PATH)
    data_transform = get_transforms()

    if model is None:
        return

    uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])

        with col1:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
            except Exception as e:
                st.error(f"Could not load image: {e}")
                return

        with col2:
            st.header("Classification Result:")
            
            prediction_class, confidence, probabilities = predict(model, image, data_transform)

            if prediction_class == "PNEUMONIA":
                st.markdown(f"**<span style='color:red; font-size: 30px;'>PNEUMONIA DETECTED</span>**", unsafe_allow_html=True)
                st.markdown("⚠️ **Action Recommended:** Please consult a medical professional immediately with this result.")
            elif prediction_class == "NORMAL":
                st.markdown(f"**<span style='color:green; font-size: 30px;'>NORMAL</span>**", unsafe_allow_html=True)
                st.markdown("✅ The image appears normal based on the model's analysis.")
            else:
                st.markdown(f"**<span style='color:orange; font-size: 30px;'>{prediction_class}</span>**", unsafe_allow_html=True)

            st.markdown(f"Confidence: **{confidence:.2%}**")
            
            # --- Comparison Chart Feature ---
            prob_data = pd.DataFrame({
                'Class': CLASS_NAMES,
                'Probability': [p.item() for p in probabilities]
            })

            highlight_color = '#FF4B4B' if prediction_class == 'PNEUMONIA' else '#008000'

            chart = alt.Chart(prob_data).mark_bar().encode(
                x=alt.X('Probability', axis=alt.Axis(format='%', title='Confidence')),
                y=alt.Y('Class', title='Classification', sort='-x'),
                color=alt.condition(
                    alt.datum.Class == prediction_class,
                    alt.value(highlight_color),
                    alt.value('lightgray')
                ),
                tooltip=['Class', alt.Tooltip('Probability', format='.2%')]
            ).properties(
                title='Model Confidence Comparison'
            )
            st.altair_chart(chart, use_container_width=True)
            # --- End Chart Feature ---

            st.info("Disclaimer: This tool is an AI aid and is NOT a substitute for professional medical diagnosis.")

if __name__ == '__main__':
    main()
