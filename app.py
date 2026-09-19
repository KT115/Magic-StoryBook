import os
import time
import random
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------------------------------------------------------
# Page Configuration & Fairy Tale Styling
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🏰", layout="centered")

# Custom CSS for fairytale aesthetics, gold accents, parchment cards, and magical animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:ital,wght@0,400;0,700;1,400&family=Cinzel:wght@600;700&display=swap');

/* Main page backdrop */
.stApp {
    background: radial-gradient(circle at 50% 20%, #2b1055 0%, #150930 60%, #090314 100%);
    color: #f7ede2;
    font-family: 'Comic Neue', cursive;
}

/* Titles */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #ffd166 !important;
    text-shadow: 0 0 12px rgba(255, 209, 102, 0.4);
    text-align: center;
}

/* Fairytale card container */
.magic-parchment {
    background: #fffdf7;
    color: #2b2d42;
    padding: 2.2rem;
    border-radius: 24px;
    border: 3px solid #e0a96d;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 209, 102, 0.25);
    margin: 1.5rem 0;
    animation: fadeIn 0.8s ease-in-out;
}

/* Fun interactive riddle box */
.riddle-box {
    background: rgba(255, 255, 255, 0.1);
    border: 2px dashed #f4a261;
    border-radius: 16px;
    padding: 1.2rem;
    margin: 1rem 0;
    text-align: center;
    color: #fceade;
    font-size: 1.15rem;
}

/* Button restyling */
div.stButton > button {
    background: linear-gradient(135deg, #f72585 0%, #7209b7 50%, #4361ee 100%) !important;
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.2rem !important;
    border: 2px solid #ffd166 !important;
    box-shadow: 0 0 15px rgba(255, 209, 102, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    margin: 0 auto;
    display: block;
}

div.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 25px rgba(255, 209, 102, 0.8) !important;
}

/* Smooth fade & pop animation for transitions */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads and caches both Hugging Face models."""
    caption_model = pipeline("image-captioning", model="Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="gpt2")
    return caption_model, story_model


# ---------------------------------------------------------
# 2. Pipeline Helpers
# ---------------------------------------------------------
def get_caption(image, caption_pipe):
    result = caption_pipe(image)
    return result[0]["generated_text"]


def get_story(caption, story_pipe):
    prompt = f"Once upon a time, there was {caption}. One sunny morning, a little hero found"
    output = story_pipe(
        prompt,
        max_new_tokens=85,
        min_length=50,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2
    )
    raw_story = output[0]["generated_text"]
    return raw_story[:raw_story.rfind(".") + 1] if "." in raw_story else raw_story


def text_to_speech(text, filename="fairy_story.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. Interactive Kid Activities During Waiting
# ---------------------------------------------------------
MAGIC_RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up but never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]

def render_magic_spell_loader(step_name):
    """Shows an animated countdown with a riddle and deep breath for kids."""
    riddle_q, riddle_a = random.choice(MAGIC_RIDDLES)
    loader_placeholder = st.empty()
    
    with loader_placeholder.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>✨ Casting Spell: {step_name}... ✨</h3>
            <p><strong>Riddle Time!</strong> {riddle_q}</p>
            <p><i>Take a deep breath and blow gently at your screen to blow the magic dust! 💨</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1.8)  # Gives kids a moment to interact and read
    
    # Reveal answer briefly
    with loader_placeholder.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>🪄 Spell Almost Ready! 🪄</h3>
            <p><strong>Answer:</strong> {riddle_a}</p>
            <p>✨ <i>Abracadabra, Alakazam!</i> ✨</p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1.2)
    loader_placeholder.empty()


# ---------------------------------------------------------
# 4. State Management (Multi-Step Storybook)
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1  # Steps: 1: Portal, 2: Crystal Ball, 3: Scroll, 4: Theatre
if "image" not in st.session_state:
    st.session_state.image = None
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""


# ---------------------------------------------------------
# 5. Main Wizard Application
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🏰 The Whispering Storybook 🏰</h1>", unsafe_allow_html=True)
    
    # Visual chapter progress bar
    chapter_names = ["1. Enchanted Mirror", "2. Crystal Ball", "3. Magic Scroll", "4. Listening Tree"]
    current_chapter = chapter_names[st.session_state.step - 1]
    st.markdown(f"<p style='text-align: center; color: #ffd166; font-size: 1.1rem;'>Chapter {current_chapter}</p>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 4)

    # ---------------------------------------------------------
    # STEP 1: The Enchanted Portal (Image Upload)
    # ---------------------------------------------------------
    if st.session_state.step == 1:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Step 1: Feed the Magic Mirror</h2>
            <p style="font-size: 1.2rem; text-align: center;">
                Choose an illustration, a toy photo, or your own drawing! The mirror will gaze inside.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Drop your image into the cauldron:", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.image = image
            st.image(image, caption="Your Enchanted Picture", use_container_width=True)

            if st.button("🪄 Awaken the Magic Mirror 🪄"):
                st.balloons()
                render_magic_spell_loader("Mirror Vision")
                
                # Run captioning model
                caption_pipe, _ = load_models()
                with st.spinner("Deciphering secrets inside the picture..."):
                    st.session_state.caption = get_caption(image, caption_pipe)
                
                st.session_state.step = 2
                st.rerun()

    # ---------------------------------------------------------
    # STEP 2: The Crystal Ball (Inspect Clues)
    # ---------------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Step 2: The Crystal Ball Speaks!</h2>
            <p style="font-size: 1.2rem; text-align: center;">The spirits in the glass peered closely and whispered:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.image:
                st.image(st.session_state.image, caption="The Source", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #eef1f6; padding: 1.5rem; border-radius: 16px; border: 2px solid #a2d2ff; text-align: center; margin-top: 1rem;">
                <h3 style="color: #1d3557 !important; margin: 0;">✨ I spy with my wizard eye:</h3>
                <p style="font-size: 1.35rem; font-weight: bold; color: #457b9d; margin-top: 0.8rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        if st.button("📜 Spin a Bedtime Tale from This Clue! 📜"):
            st.snow()
            render_magic_spell_loader("Story Weaving")
            
            # Run text generation
            _, story_pipe = load_models()
            with st.spinner("Weaving golden threads into a story..."):
                st.session_state.story = get_story(st.session_state.caption, story_pipe)
            
            st.session_state.step = 3
            st.rerun()

    # ---------------------------------------------------------
    # STEP 3: The Ancient Scroll (Read the Story)
    # ---------------------------------------------------------
    elif st.session_state.step == 3:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">📜 Step 3: The Golden Story Scroll</h2>
            <p style="font-size: 1.25rem; line-height: 1.8; color: #1a1a24;">
        """ + st.session_state.story + """
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🎭 Turn Story into Fairy Audio"):
                render_magic_spell_loader("Voice Enchantment")
                with st.spinner("Calling upon the singing fairies..."):
                    st.session_state.audio_path = text_to_speech(st.session_state.story)
                st.session_state.step = 4
                st.rerun()
        with col2:
            if st.button("🔄 Start New Journey"):
                st.session_state.step = 1
                st.session_state.image = None
                st.rerun()

    # ---------------------------------------------------------
    # STEP 4: The Listening Tree (Audio Theatre)
    # ---------------------------------------------------------
    elif st.session_state.step == 4:
        st.balloons()
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🧚 Step 4: The Fairy Audio Theatre</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #3d405b;">
                Sit back, close your eyes, and listen to the voice of the kingdom!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path, format="audio/mp3")

        st.markdown("""
        <div class="magic-parchment" style="margin-top: 1rem;">
            <h3 style="color: #7209b7 !important;">📖 Story Recap</h3>
            <p style="font-size: 1.1rem; line-height: 1.7; color: #2b2d42;">
        """ + st.session_state.story + """
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🏰 Brew Another Story"):
            if os.path.exists(st.session_state.audio_path):
                os.remove(st.session_state.audio_path)
            st.session_state.step = 1
            st.session_state.image = None
            st.session_state.caption = ""
            st.session_state.story = ""
            st.session_state.audio_path = ""
            st.rerun()


if __name__ == "__main__":
    main()
