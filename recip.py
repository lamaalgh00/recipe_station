import os
import io
import base64
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.agents import initialize_agent, AgentType, tool

# --------------------------------------------------------
# 🔧 Setup
# --------------------------------------------------------
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

st.set_page_config(page_title="🍽️ Dishcovery AI", layout="centered")
st.title("🍽️ Dishcovery AI")
st.write("Upload a food photo → detect ingredients → get 3 recipes → pick your favorite → get recommendations.")

# --------------------------------------------------------
# 🖼 Image upload + user preferences
# --------------------------------------------------------
uploaded_file = st.file_uploader("📸 Upload your food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

st.markdown("### ⚙️ Preferences")
col1, col2, col3 = st.columns(3)
with col1:
    cuisine_type = st.selectbox("Cuisine", ["Any", "Italian", "Arabic", "Asian", "Mexican", "Indian", "French", "Mediterranean"])
with col2:
    allergies = st.text_input("Allergies (comma-separated)", "")
with col3:
    taste_pref = st.selectbox("Taste Preference", ["Any", "Salty", "Sweet", "Spicy", "Savory", "Umami", "Sour"])


# --------------------------------------------------------
# 🧩 Helper functions
# --------------------------------------------------------
def detect_ingredients_chain(image_file):
    system_msg = SystemMessage(content=(
        "You are a food recognition assistant. Identify main edible ingredients. "
        "Ignore leaves, stems, or peels that are part of the same item. "
        "List distinct ingredients as bullet points. If unsure, say 'unsure'."
    ))
    human_msg = HumanMessage(content=[
        {"type": "text", "text": "What ingredients do you see in this image?"},
        {
            "type": "image",
            "image": image_file.read()
        }
    ])
    response = llm.invoke([system_msg, human_msg])
    return response.content.strip()


# --------------------------------------------------------
# 🍝 Recipe Generation Chain
# --------------------------------------------------------
recipe_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional chef and recipe generator."),
    ("user", """
Detected ingredients (reference list):
{ingredients}

User preferences:
- Cuisine: {cuisine}
- Allergies: {allergies}
- Taste: {taste}

Generate 3 recipes:
- Use mostly detected ingredients.
- Avoid allergic ingredients.
- Match cuisine/taste preferences.
Rules:
1. Mark any ingredient NOT found in detected list with a *.
2. End with this note:
   "*Ingredients marked with * are not detected in the image and can be ordered from HungerStation Market."

Output format:
### 1. Recipe Name
- 3 tags (#italian, #salty, etc.)
- Short description (1–2 sentences)
- Key ingredients (with quantities)
- Steps (3–6 short steps)
""")
])

def recipe_chain(ingredients, cuisine, allergies, taste):
    chain = recipe_prompt | llm
    response = chain.invoke({
        "ingredients": ingredients,
        "cuisine": cuisine,
        "allergies": allergies,
        "taste": taste
    })
    return response.content.strip()


# --------------------------------------------------------
# 🤖 Recommendation Agent Tool
# --------------------------------------------------------
@tool
def recommend_similar_recipe(selected_recipe: str) -> str:
    """Return 2 similar recipes or enhancements based on user's favorite dish."""
    suggestions = [
        f"If you liked {selected_recipe}, try making a healthier version with less oil.",
        f"Or try a variation using seasonal vegetables or different spices!"
    ]
    return "\n".join(suggestions)

tools = [recommend_similar_recipe]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# --------------------------------------------------------
# ▶️ Main Flow
# --------------------------------------------------------
if st.button("🍳 Generate Recipes", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload an image first.")
    else:
        try:
            # Step 1: Detect ingredients (no base64, send raw bytes)
            with st.spinner("🧂 Detecting ingredients..."):
                uploaded_file.seek(0)  # reset pointer
                ingredients_text = detect_ingredients_chain(uploaded_file)

            st.markdown("### 🧂 Detected Ingredients")
            st.markdown(ingredients_text)

            # Step 2: Generate recipes
            with st.spinner("🍝 Generating 3 recipes..."):
                recipes_text = recipe_chain(
                    ingredients_text,
                    cuisine_type,
                    allergies,
                    taste_pref
                )

            st.markdown("### 🍽️ Generated Recipes")
            recipe_list = [r.strip() for r in recipes_text.split("### ") if r.strip()]
            recipe_titles = [r.split("\n")[0] for r in recipe_list[:3]]

            # Step 3: Let user choose favorite
            fav = st.radio("❤️ Choose your favorite recipe:", recipe_titles)

            # Step 4: Recommendations
            if st.button("🤖 Get Similar Recommendations"):
                with st.spinner("Finding related recipes..."):
                    rec_response = agent.invoke(
                        f"The user liked {fav}. Recommend similar recipes."
                    )
                st.markdown("### 🧠 Recommended for You")
                st.markdown(rec_response["output"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")

