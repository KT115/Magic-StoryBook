import os
import time
import random
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# ---------------------------------------------------------
# Page Configuration & Pastel Fairytale Theme
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

/* Dreamy Pastel Rainbow Gradient Backdrop */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
    color: #4a4e69 !important;
}

/* Titles and Headers */
h1 {
    font-family: 'Cinzel Decorative', cursive !important;
    color: #ff477e !important;
    text-shadow: 0 2px 15px rgba(255, 255, 255, 0.9) !important;
    text-align: center;
    font-size: 2.2rem !important;
}

h2, h3 {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 700 !important;
    color: #6a0572 !important;
    text-align: center;
}

/* Storybook Parchment Card */
.magic-parchment {
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(12px);
    border-radius: 28px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35), 0 0 20px rgba(255, 255, 255, 0.8) inset !important;
    padding: 2rem !important;
    margin: 1.2rem 0 !important;
    animation: bounceIn 0.5s ease-out;
}

/* Magical Loading Card */
.magic-loader {
    background: rgba(255, 255, 255, 0.95);
    border: 3px dashed #ff70a6;
    border-radius: 22px;
    padding: 1.5rem;
    margin: 1.2rem 0;
    text-align: center;
    box-shadow: 0 8px 25px rgba(255, 112, 166, 0.25);
}

/* Centered Magic Buttons */
div[data-testid="stColumn"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

div.stButton {
    display: flex;
    justify-content: center;
    margin: 0.8rem auto;
}

div.stButton > button {
    background: linear-gradient(135deg, #ff758f 0%, #ff499e 40%, #70d6ff 100%) !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.4rem !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.35) !important;
    transition: all 0.25s ease-in-out !important;
}

div.stButton > button:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 10px 25px rgba(255, 73, 158, 0.55) !important;
}

/* Cinema Container & Subtitles */
.storybook-cinema {
    position: relative;
    background: #000;
    border: 5px solid #ffb3c6;
    border-radius: 24px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(112, 214, 255, 0.4);
    margin: 1.2rem 0;
}

.subtitles-banner {
    background: rgba(30, 11, 46, 0.88);
    border-top: 2px solid #ffd166;
    color: #fff9a6;
    font-size: 1.2rem;
    font-weight: 700;
    padding: 1rem 1.4rem;
    text-align: center;
    line-height: 1.6;
}

@keyframes bounceIn {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1.0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="🧙‍♂️ Gathering magical fairy spellbooks (downloading models)...")
def load_models():
    """
    Direct model initialization (no pipeline KeyError).
      - Vision: Salesforce/blip-image-captioning-base
      - Story:  distilbert/distilgpt2
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="distilbert/distilgpt2")
    return processor, caption_model, story_model


# ---------------------------------------------------------
# 2. Pipeline Helpers
# ---------------------------------------------------------
def get_caption(image, processor, caption_model):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)


def get_story(caption, story_pipe):
    clean_caption = caption.strip().rstrip(".")
    prompt = (
        f"Once upon a time, there was {clean_caption}. "
        f"Every day was filled with fun, but today a secret golden rainbow door opened! "
        f"With wide curious eyes, our brave little friend stepped forward"
    )
    
    output = story_pipe(
        prompt,
        min_new_tokens=65,
        max_new_tokens=95,
        do_sample=True,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.3
    )
    
    raw_story = output[0]["generated_text"]
    last_period = max(raw_story.rfind("."), raw_story.rfind("!"), raw_story.rfind("?"))
    if last_period != -1:
        story = raw_story[:last_period + 1]
    else:
        story = raw_story + " And everyone smiled happily ever after!"
    return story


def text_to_speech(text, filename="magic_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. Interactive Kid Activities
# ---------------------------------------------------------
RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up and never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]


# ---------------------------------------------------------
# 4. State Management
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""


# ---------------------------------------------------------
# 5. Main Application Flow
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
    
    chapter_titles = [
        "Chapter 1: The Magic Portal",
        "Chapter 2: The Crystal Ball",
        "Chapter 3: The Golden Scroll",
        "Chapter 4: The Voice Harp",
        "Chapter 5: The Living Story Cinema"
    ]
    st.markdown(f"<p style='text-align: center; color: #6a0572; font-size: 1.15rem; font-weight: 700;'>{chapter_titles[st.session_state.step - 1]}</p>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 5)

    # -----------------------------------------------------
    # CHAPTER 1: The Magic Portal
    # -----------------------------------------------------
    if st.session_state.step == 1:
        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 1: The Magic Portal</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Feed a picture of a cute pet, drawing, or toy to the magical portal!
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a picture to transform into a bedtime story:", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            # Store image in persistent session state
            st.session_state.uploaded_img = Image.open(uploaded_file).convert("RGB")
            st.image(st.session_state.uploaded_img, caption="Your Enchanted Picture", use_container_width=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    loader_container = st.empty()
                    q, a = random.choice(RIDDLES)
                    
                    with loader_container.container():
                        st.markdown(f"""
                        <div class="magic-loader">
                            <h3 style="color: #ff477e !important;">✨ Summoning Vision Fairies... ✨</h3>
                            <p style="font-size: 1.15rem; color: #6a0572;">The mirror is sprinkling pixie dust over your picture!</p>
                            <hr style="border-top: 1px dashed #ffb3c6; margin: 0.8rem 0;">
                            <p><strong>🦄 Riddle while waiting:</strong> {q}</p>
                            <p style="color: #ff758f; font-weight: bold;">💨 Blow gentle sparkles on the screen!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Run processing while loader is visible
                    processor, caption_model, _ = load_models()
                    caption_text = get_caption(st.session_state.uploaded_img, processor, caption_model)
                    st.session_state.caption = caption_text

                    # Transition to step 2 immediately
                    loader_container.empty()
                    st.session_state.step = 2
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 2: The Crystal Ball
    # -----------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 2: The Crystal Ball Speaks!</h2>
            <p style="font-size: 1.15rem; text-align: center;">Here is the secret clue the crystal ball spotted inside your picture:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1.4rem; border-radius: 18px; border: 2px solid #70d6ff; text-align: center; margin-top: 1rem;">
                <h3 style="color: #6a0572 !important; margin: 0;">✨ Mirror Revelation:</h3>
                <p style="font-size: 1.3rem; font-weight: bold; color: #ff477e; margin-top: 0.6rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("📜 Weave a Fairytale from This Clue! 📜"):
                loader_container = st.empty()
                q, a = random.choice(RIDDLES)
                
                with loader_container.container():
                    st.markdown(f"""
                    <div class="magic-loader">
                        <h3 style="color: #ff477e !important;">✨ Spinning Golden Story Threads... ✨</h3>
                        <p style="font-size: 1.15rem; color: #6a0572;">The storybook elves are writing an adventure just for you!</p>
                        <hr style="border-top: 1px dashed #ffb3c6; margin: 0.8rem 0;">
                        <p><strong>🦄 Riddle while waiting:</strong> {q}</p>
                    </div>
                    """, unsafe_allow_html=True)

                _, _, story_pipe = load_models()
                st.session_state.story = get_story(st.session_state.caption, story_pipe)
                
                loader_container.empty()
                st.session_state.step = 3
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 3: The Golden Story Scroll
    # -----------------------------------------------------
    elif st.session_state.step == 3:
        st.markdown("""
        <div class="magic-parchment">
            <h2>📜 Chapter 3: The Golden Story Scroll</h2>
        </div>
        """, unsafe_allow_html=True)

        img_col, text_col = st.columns([1, 1.2])
        with img_col:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="The Illustrated Scene", use_container_width=True)
        
        with text_col:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 18px; padding: 1.4rem; min-height: 220px; box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);">
                <p style="font-size: 1.15rem; line-height: 1.8; color: #2b2d42; margin: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("🎶 Proceed to Voice Harp (Chapter 4)"):
                st.session_state.step = 4
                st.rerun()
        with btn_c2:
            if st.button("🔄 Try Another Picture"):
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 4: The Voice Harp
    # -----------------------------------------------------
    elif st.session_state.step == 4:
        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Let the singing fairies transform this written tale into spoken audio narration!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.audio_path or not os.path.exists(st.session_state.audio_path):
            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🧚 Cast Voice Spell"):
                    loader_container = st.empty()
                    with loader_container.container():
                        st.markdown("""
                        <div class="magic-loader">
                            <h3 style="color: #ff477e !important;">✨ Harmonizing the Fairyland Harp... ✨</h3>
                            <p style="font-size: 1.15rem; color: #6a0572;">Recording the royal storyteller's warm voice!</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.session_state.audio_path = text_to_speech(st.session_state.story)
                    loader_container.empty()
                    st.rerun()
        else:
            st.success("✨ Fairy Audio Narrated Successfully!")
            st.audio(st.session_state.audio_path, format="audio/mp3")

            st.write("")
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                if st.button("🎬 Open Living Story Cinema (Chapter 5)"):
                    st.session_state.step = 5
                    st.rerun()
            with btn_c2:
                if st.button("📜 Back to Story Scroll"):
                    st.session_state.step = 3
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 5: The Living Story Cinema (Video & Subtitles)
    # -----------------------------------------------------
    elif st.session_state.step == 5:
        st.balloons()
        st.markdown("""
        <div class="magic-parchment">
            <h2>🎬 Chapter 5: The Living Story Cinema</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Your fairytale illustrated book comes alive with synced voice narration and on-screen subtitles!
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="storybook-cinema">', unsafe_allow_html=True)
        if st.session_state.uploaded_img:
            st.image(st.session_state.uploaded_img, use_container_width=True)
        
        st.markdown(f"""
        <div class="subtitles-banner">
            💛 <strong>Subtitles:</strong> "{st.session_state.story}"
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path, format="audio/mp3")

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("🏰 Create a Brand New Fairytale"):
                if os.path.exists(st.session_state.audio_path):
                    os.remove(st.session_state.audio_path)
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.rerun()


if __name__ == "__main__":
    main()
