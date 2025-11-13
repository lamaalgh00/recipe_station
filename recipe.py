import os
import io
import base64
from dotenv import load_dotenv
from PIL import Image
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

st.set_page_config(page_title="Smart Recipe Recommender", layout="centered")
st.title("🍽️ Dishcovery")
st.write("Upload a food photo, set your preferences, and get tailored recipes.")

# Upload Image
uploaded_file = st.file_uploader(
    "📸 Upload a photo of your meal or ingredients",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

# USER PREFERENCES 
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
        "Meal type",
        ["Any", "Breakfast", "Lunch", "Dinner", "Dessert", "Snack"]
    )


def detect_ingredients(image_file):
    # Convert the uploaded file into a compressed base64 data URL (string)
    image_file.seek(0)
    image = Image.open(image_file).convert("RGB")

    # Resize + compress to avoid huge payloads
    max_dim = 256
    try:
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    except Exception:
        image = image.resize((max_dim, max_dim))

    buffer = io.BytesIO()
    quality = 30
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    data = buffer.getvalue()

    # reduce quality if still large
    max_bytes = 40_000
    while len(data) > max_bytes and quality > 10:
        quality -= 5
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()

    b64 = base64.b64encode(data).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    system = SystemMessage(content=
        "You are a food recognition assistant. Identify ONLY visible edible ingredients. "
        "If not clearly visible, say 'unsure'. Return a bullet list."
    )

    # Pass the data URL string instead of raw bytes so it can be JSON-serialized
    user = HumanMessage(content=[
        {"type": "text", "text": "Identify the visible ingredients in this photo."},
        {"type": "image_url", "image_url": {"url": data_url}}
    ])

    resp = llm.invoke([system, user])
    return getattr(resp, "content", str(resp)).strip()





def recommend_recipes(ingredients_text: str, cuisine: str, allergies: str, taste: str) -> str:
    """Generate 3 recipe recommendations given detected ingredients and preferences."""
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
   "*Ingredients marked with * are not detected in the image and can be ordered from HungerStation Market."

Output format:

### 1. Recipe Name
- one word 3 tags (for example #salty, #italian)
- Short description (1–2 sentences)
- Key ingredients (bullet list with quantities)
- Steps (3–6 short steps)

Use clear markdown formatting.
"""

    system = SystemMessage(content="You generate safe, clear cooking recipes.")
    user = HumanMessage(content=prompt)
    resp = llm.invoke([system, user])
    return getattr(resp, "content", str(resp)).strip()


# MAIN ACTION
if st.button("Find recipes", use_container_width=True):
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        try:
            uploaded_file.seek(0)
            with st.spinner("Detecting ingredients..."):
                ingredients_text = detect_ingredients(uploaded_file)
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
