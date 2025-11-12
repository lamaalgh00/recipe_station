import streamlit as st
from openai import OpenAI
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

# ---------- CONFIG ----------
load_dotenv()  # loads OPENAI_API_KEY from .env if present
client = OpenAI()  # uses OPENAI_API_KEY env variable

st.set_page_config(page_title="Smart Recipe Recommender", layout="centered")
st.title("🍽️ Dishcovery")
st.write("Upload a food photo, set your preferences, and get tailored recipes.")

# ---------- IMAGE UPLOAD ----------
uploaded_file = st.file_uploader(
    "📸 Upload a photo of your meal or ingredients",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

# ---------- USER PREFERENCES ----------
st.markdown("### ⚙️ Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    cuisine_type = st.selectbox(
        "Cuisine type",
        ["Any", "Italian", "Arabic", "Asian", "Mexican", "Indian", "French", "Mediterranean"]
    )

with col2:
    allergies = st.text_input(
        "Allergies (comma-separated)",
        value=""
    )

with col3:
    taste_pref = st.selectbox(
        "Taste preference",
        ["Any", "Salty", "Sweet", "Spicy", "Savory", "Umami", "Sour"]
    )


def image_to_data_url(image_file) -> str:
    """Convert uploaded file to base64 data URL for OpenAI vision."""
    image = Image.open(image_file).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def detect_ingredients(data_url: str) -> str:
    """Use OpenAI (GPT-4o-mini) to detect ingredients from an image."""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
    {
        "role": "system",
        "content": (
            "You are a food recognition assistant. "
            "Look at the image and identify the main edible ingredients. "
            "Ignore small decorative or natural parts such as leaves, stems, peels, or shadows "
            "if they are part of the same fruit or vegetable. "
            "List only the distinct food ingredients as bullet points. "
            "If unsure, say 'unsure'."
        ),
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What ingredient or ingredients do you see in this image? List them clearly.",
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ],
    },
]

    )
    return completion.choices[0].message.content.strip()


def recommend_recipes(ingredients_text: str, cuisine: str, allergies: str, taste: str) -> str:
    """Use OpenAI to recommend recipes based on detected ingredients and user prefs."""
    prompt = f"""
You are a professional chef and recipe generator.

Detected ingredients (reference list):
{ingredients_text}

User preferences:
- Preferred cuisine: {cuisine}
- Allergies (must avoid): {allergies}
- Taste preference: {taste}

Task:
Recommend 3 recipes that:
- Use mainly the detected ingredients
- Respect the allergies and avoid them completely
- Match the cuisine and taste preference as much as possible

Rules for formatting:
1. Any ingredient in your recipe that is NOT present in the detected ingredients list must have a * directly after it.
   Example: “1 tsp salt*”
2. At the end of the entire output, add a short note:
   “*Ingredients marked with * are not detected in the image and can be ordered from HungerStation Market.”

Output format:

### 1. Recipe Name
- one word 3 tags (for example #salty, #italian)
- Short description (1–2 sentences)
- Key ingredients (bullet list with quantities)
- Steps (3–6 short steps)

Use clear markdown formatting.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate safe, clear cooking recipes."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content.strip()



# ---------- MAIN ACTION ----------
if st.button("Find recipes", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        try:
            # 1) Convert image to data URL
            data_url = image_to_data_url(uploaded_file)

            # 2) Detect ingredients
            with st.spinner("Detecting ingredients from the image..."):
                ingredients_text = detect_ingredients(data_url)

            st.markdown("### Detected Ingredients")
            st.markdown(ingredients_text)

            # 3) Recommend recipes
            with st.spinner("🍝 Generating recipe recommendations..."):
                recipes_markdown = recommend_recipes(
                    ingredients_text,
                    cuisine_type,
                    allergies,
                    taste_pref
                )

            st.markdown("### 🍽️ Recommended Recipes")
            st.markdown(recipes_markdown)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
