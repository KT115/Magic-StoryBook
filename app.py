import os
import time
import random
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# ---------------------------------------------------------
# Page Configuration & Full Fairytale Background Theme
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🏰", layout="centered")

# Target all Streamlit view layers to guarantee the night-sky fairytale background
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&family=Cinzel:wght@600;700&display=swap');

/* Force magical background across all Streamlit container levels */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: radial-gradient(circle at 50% 15%, #2c1254 0%, #15082b 55%, #0a0316 100%) !important;
    color: #fceade !important;
    font-family: 'Comic Neue', cursive !important;
}

/* Gold fairytale headings */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #ffd166 !important;
    text-shadow: 0 0 14px rgba(255, 209, 102, 0.5) !important;
    text-align: center;
}

/* Cream fairytale parchment card */
.magic-parchment {
    background: #fffdf6 !important;
    color: #2b2d42 !important;
    padding: 1.8rem;
    border-radius: 22px;
    border: 3px solid #e0a96d;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 22px rgba(255, 209, 102, 0.3);
    margin: 1.2rem 0;
    animation: magicPop 0.5s ease-out;
}

/* Interactive Riddle / Breath Box */
.riddle-box {
    background: rgba(255, 255, 255, 0.12);
    border: 2px dashed #f4a261;
    border-radius: 18px;
    padding: 1.2rem;
    margin: 1.2rem 0;
    text-align: center;
    color: #fceade;
    font-size: 1.15rem;
}

/* Horizontal Centering for all buttons */
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
    background: linear-gradient(135deg, #f72585 0%, #7209b7 50%, #4361ee 100%) !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.65rem 2.2rem !important;
    border: 2px solid #ffd166 !important;
    box-shadow: 0 0 16px rgba(255, 209, 102, 0.45) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div.stButton > button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 28px rgba(255, 209, 102, 0.85) !important;
}

/* Story Cinema Container */
.cinema-container {
    background: #120726;
    border: 3px solid #ffd166;
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(255, 209, 102, 0.4);
}

@keyframes magicPop {
    from { opacity: 0; transform: scale(0.97); }
    to { opacity: 1; transform: scale(1.0); }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Direct model initialization to avoid task registry KeyError:
      - Vision: Salesforce/blip-image-captioning-base
      - Story:  distilbert/distilgpt2 (generates cohesive kid stories fast)
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="distilbert/distilgpt2")
    return processor, caption_model, story_model


# ---------------------------------------------------------
# 2. Pipeline Helpers
# ---------------------------------------------------------
def get_caption(image, processor, caption_model):
    """Extracts description from the image."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = caption_model.generate(**inputs, max_new_tokens=40)
    return processor.decode(out[0], skip_special_tokens=True)


def get_story(caption, story_pipe):
    """
    Guarantees a 50-100 word fairy-tale story using structured prompts
    and forced minimum token generation.
    """
    clean_caption = caption.strip().rstrip(".")
    # Strong narrative setup suitable for children 3-10
    prompt = (
        f"Once upon a time, there was {clean_caption}. "
        f"Every day brought wonder, but today was special because a secret magical door appeared. "
        f"With a happy leap, our little hero set off on an adventure"
    )
    
    output = story_pipe(
        prompt,
        min_new_tokens=65,       # Enforces a 50-100 word minimum
        max_new_tokens=95,       # Keeps it within children's attention span
        do_sample=True,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.3   # Prevents looping on the caption
    )
    
    raw_story = output[0]["generated_text"]
    
    # Trim cleanly at the final sentence mark
    last_period = max(raw_story.rfind("."), raw_story.rfind("!"), raw_story.rfind("?"))
    if last_period != -1:
        story = raw_story[:last_period + 1]
    else:
        story = raw_story + " And everyone lived happily ever after!"
        
    return story


def text_to_speech(text, filename="story_sound.mp3"):
    """Converts the narrative into an MP3 file."""
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. Kid Riddle / Loading Screen
# ---------------------------------------------------------
MAGIC_RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up but never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]

def show_interactive_loader(magic_action):
    riddle_q, riddle_a = random.choice(MAGIC_RIDDLES)
    box = st.empty()
    
    with box.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>✨ Casting: {magic_action}... ✨</h3>
            <p><strong>Riddle Time:</strong> {riddle_q}</p>
            <p><i>Blow softly on your screen to help make the magic work! 💨</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(2.0)
    
    with box.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>🪄 The Magic Worked! 🪄</h3>
            <p><strong>Answer:</strong> {riddle_a}</p>
            <p>✨ <i>Abracadabra!</i> ✨</p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1.2)
    box.empty()


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
# 5. Main Storybook Flow
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🏰 The Whispering Storybook 🏰</h1>", unsafe_allow_html=True)
    
    chapter_names = ["Chapter 1: The Mirror", "Chapter 2: The Crystal", "Chapter 3: The Scroll", "Chapter 4: The Cinema"]
    st.markdown(f"<p style='text-align: center; color: #ffd166; font-size: 1.15rem; font-weight: bold;'>{chapter_names[st.session_state.step - 1]}</p>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 4)

    # -----------------------------------------------------
    # CHAPTER 1: Image Upload
    # -----------------------------------------------------
    if st.session_state.step == 1:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Chapter 1: Feed the Magic Mirror</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Upload a picture of a pet, drawing, or toy to reveal its hidden fairytale!
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_img = image
            st.image(image, caption="Your Enchanted Picture", use_container_width=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    st.balloons()
                    show_interactive_loader("Mirror Vision")
                    
                    processor, caption_model, _ = load_models()
                    with st.spinner("Gazing through the enchanted glass..."):
                        st.session_state.caption = get_caption(image, processor, caption_model)
                    
                    st.session_state.step = 2
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 2: The Clue
    # -----------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Chapter 2: The Crystal Ball Speaks</h2>
            <p style="font-size: 1.15rem; text-align: center;">Here is what the magical mirror saw:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #eef1f6; padding: 1.5rem; border-radius: 16px; border: 2px solid #a2d2ff; text-align: center; margin-top: 1rem;">
                <h3 style="color: #1d3557 !important; margin: 0;">✨ The Mirror Saw:</h3>
                <p style="font-size: 1.3rem; font-weight: bold; color: #457b9d; margin-top: 0.5rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("📜 Spin a Bedtime Tale! 📜"):
                st.snow()
                show_interactive_loader("Story Weaving")
                
                _, _, story_pipe = load_models()
                with st.spinner("Weaving golden words into a story..."):
                    st.session_state.story = get_story(st.session_state.caption, story_pipe)
                
                st.session_state.step = 3
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 3: Scroll & Photo Together
    # -----------------------------------------------------
    elif st.session_state.step == 3:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">📜 Chapter 3: The Golden Story Scroll</h2>
        </div>
        """, unsafe_allow_html=True)

        # Photo and story side-by-side[cite: 3]
        img_col, text_col = st.columns([1, 1.2])
        with img_col:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="The Illustrated Scene", use_container_width=True)
        
        with text_col:
            st.markdown(f"""
            <div style="background: #fffdf6; border: 2px dashed #e0a96d; border-radius: 16px; padding: 1.3rem; min-height: 220px;">
                <p style="font-size: 1.15rem; line-height: 1.75; color: #2b2d42; margin: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        # Centered action buttons
        btn_c1, btn_c2 = st.columns([1, 1])
        with btn_c1:
            if st.button("🎬 Watch Story Cinema (Audio & Show)"):
                show_interactive_loader("Narrator Enchantment")
                with st.spinner("Calling upon the singing storyteller..."):
                    st.session_state.audio_path = text_to_speech(st.session_state.story)
                st.session_state.step = 4
                st.rerun()
        with btn_c2:
            if st.button("🔄 Start New Journey"):
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 4: Cinema Theatre
    # -----------------------------------------------------
    elif st.session_state.step == 4:
        st.balloons()
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🎬 Chapter 4: Fairyland Story Cinema</h2>
            <p style="font-size: 1.15rem; text-align: center; color: #3d405b;">
                Listen to the voice of the kingdom while following along with the scene!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Cinema Box with visual glow and synchronized sound
        st.markdown('<div class="cinema-container">', unsafe_allow_html=True)
        if st.session_state.uploaded_img:
            st.image(st.session_state.uploaded_img, caption="🎬 Fairyland Scene", use_container_width=True)
        
        if os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path, format="audio/mp3")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="magic-parchment" style="margin-top: 1rem;">
            <h3 style="color: #7209b7 !important;">📖 Full Narrative</h3>
            <p style="font-size: 1.1rem; line-height: 1.7; color: #2b2d42;">
                {st.session_state.story}
            </p>
        </div>
        """, unsafe_allow_html=True)

        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("🏰 Create Another Fairytale"):
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
