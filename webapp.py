import streamlit as st
from PIL import Image
import pytesseract
from google import genai
from google.genai import types
import cv2
import numpy as np
import json
from pydantic import BaseModel, Field
from typing import List

# --- 1. App Configuration ---
st.set_page_config(
    page_title="Ingredient Safety Scanner",
    page_icon="🛡️",
    layout="centered"
)

# --- 2. API Setup ---
# Tries to get the key from Streamlit Secrets (for Web Deployment)
# If not found, asks the user to input it (for Local Testing)
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    api_key = st.text_input("Enter Google API Key", type="password")

if api_key:
    client = genai.Client(api_key=api_key)

# --- 3. Data Models (Pydantic) ---
class HarmfulIngredient(BaseModel):
    name: str = Field(..., description="The name of the ingredient found in the list.")
    harmful_effect: str = Field(..., description="A brief summary of the potential harmful health effects, considering the estimated quantity based on list order.")
    category: str = Field(..., description="Category of harm, e.g., 'Carcinogenic', 'Diabetic Risk', 'Allergen', 'Obesity', 'Cardiovascular Risk', 'Hormonal Disruption', 'Teratogenic', 'DNA Damage'.")

class IngredientAnalysis(BaseModel):
    items: List[HarmfulIngredient]

# --- 4. Your Custom Image Processing Logic ---
def preprocess_image(image_pil):
    """
    Applies the specific pipeline: Resize -> Grayscale -> GaussianBlur 
    -> Otsu Threshold -> Morph Close -> Invert Check
    """
    # Convert PIL (RGB) to OpenCV (BGR)
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Resize (Upscale 2x)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Gaussian Blur (5x5) - Removes scanning noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. Otsu Thresholding
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphological Closing - Fills gaps in letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 6. Inversion Check (Fix for White text on Dark background)
    white_pixel_count = np.sum(thresh == 255)
    total_pixels = thresh.size
    if white_pixel_count > total_pixels / 2:
        pass # Background is white (Standard)
    else:
        thresh = cv2.bitwise_not(thresh) # Background is dark (Invert)
    
    return thresh

# --- 5. Main UI Layout ---
st.title("🛡️ Ingredient Safety Scanner")
st.markdown("Scan food labels to detect carcinogens, allergens, and other health risks.")

# Input Selection
option = st.radio("Select Input:", ("Camera", "Upload Image"), horizontal=True)

image_input = None
if option == "Camera":
    image_input = st.camera_input("Take a photo of the ingredients list")
else:
    image_input = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_input and api_key:
    # Show Original
    st.image(image_input, caption="Original Image", use_container_width=True)
    
    with st.spinner("Processing image..."):
        # Load and Preprocess
        pil_image = Image.open(image_input)
        processed_cv2 = preprocess_image(pil_image)
        
        # Convert back to PIL for Tesseract
        processed_pil = Image.fromarray(processed_cv2)
        
        # Debug View (Optional)
        with st.expander("View Computer Vision Output"):
            st.image(processed_pil, caption="Processed for OCR", use_container_width=True)

        # Run Tesseract OCR
        # UNCOMMENT THE NEXT LINE IF RUNNING LOCALLY ON WINDOWS:
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\peish\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        
        try:
            extracted_text = pytesseract.image_to_string(processed_pil, config='--psm 6')
        except Exception as e:
            st.error(f"OCR Error: {e}. If running locally, check your Tesseract path.")
            extracted_text = ""

    if extracted_text:
        st.subheader("📝 Extracted Text")
        # Allow user to fix OCR errors manually
        final_text = st.text_area("Verify text before analysis:", extracted_text, height=150)
        
        if st.button("Analyze Risks 🧬"):
            if len(final_text.strip()) < 5:
                st.error("Text is too short or empty. Please retake the photo.")
            else:
                with st.spinner("Analyzing against FDA safety guidelines..."):
                    prompt = f"""
                    Analyze this ingredients list: {final_text}.
                    Identify only the ingredients with potential negative health effects, your analysis should be based on the FDA food safety list and guidance, and judged based on the amount of said ingredients as well (for example, low amount of salt is generally considered safe, but if salt is located higher up in the list, which means higher salt content, it should be list out as harmful and explanation is given in the 'Effect' object).
                    The order in which the ingredients are listed (which represents the amount of said ingredients) should not be ignored and should be take into consideration as well.
                    DO NOT SKIMP OVER ANY INGREDIENTS, EVERY INGREDIENTS NEEDS TO BE ANALYZE BEFORE POTENTIALLY NEGATIVE INGREDIENTS CAN BE IDENTIFIED.
                    """
                    
                    try:
                        response = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=IngredientAnalysis
                            )
                        )
                        
                        data = json.loads(response.text)
                        
                        if not data['items']:
                            st.success("✅ No obviously harmful ingredients detected!")
                        else:
                            st.warning(f"⚠️ Found {len(data['items'])} ingredients of concern:")
                            for item in data['items']:
                                with st.container():
                                    st.markdown(f"### 🔴 {item['name']}")
                                    st.caption(f"Category: {item['category']}")
                                    st.info(item['harmful_effect'])
                                    st.divider()
                                    
                    except Exception as e:
                        st.error(f"AI Analysis Error: {e}")

elif not api_key:

    st.info("👆 Please enter your Google API Key to start.")
