import os
import re
import random
import time
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForCausalLM
)
from gtts import gTTS

# 開啟多執行緒加速純 CPU 運算
torch.set_num_threads(4)

# ---------------------------------------------------------
# 1. Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

def inject_global_features():
    """Autoplay fairy BGM and smooth scroll to top on reruns"""
    components.html(
        """
        <audio id="bgm" loop autoplay>
            <source src="https://cdn.pixabay.com/download/audio/2022/01/26/audio_d0c6ff1cb8.mp3" type="audio/mpeg">
        </audio>
        <script>
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
            const mainSection = window.parent.document.querySelector('section.main');
            if (mainSection) { mainSection.scrollTo({top: 0, behavior: 'smooth'}); }

            var bgm = document.getElementById("bgm");
            if (bgm) {
                bgm.volume = 0.15;
                bgm.play().catch(e => console.log("Waiting for user interaction to unlock BGM..."));
            }
            window.parent.document.body.addEventListener('click', function() {
                if (bgm && bgm.paused) {
                    bgm.play().catch(e => console.log("BGM playing..."));
                }
            }, { once: true });
        </script>
        """,
        height=0
    )

def play_fairy_magic_sfx():
    """即時播放仙子魔法棒叮鈴聲 (Live Fairy Magic SFX)"""
    components.html(
        """
        <audio autoplay>
            <source src="https://cdn.pixabay.com/download/audio/2022/03/15/audio_24e07ea724.mp3" type="audio/mpeg">
        </audio>
        """,
        height=0
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Cinzel+Decorative:wght@700&display=swap');

/* Global Background */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
}

/* 標題強制一行出晒所有文字 (White-space nowrap 絕對不折行) */
h1 {
    font-family: 'Cinzel Decorative', cursive !important;
    color: #d81159 !important;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.95) !important;
    text-align: center;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
}

h2, h3 {
    font-family: 'Quicksand', sans-serif !important;
    font-weight: 800 !important;
    color: #4a0e4e !important;
    text-align: center;
    white-space: nowrap !important;
}

p, span, label, div {
    color: #1d2129 !important;
    font-weight: 600;
}

/* Parchment Card Container */
.magic-parchment {
    background: rgba(255, 255, 255, 0.96) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ff758f !important;
    box-shadow: 0 12px 35px rgba(255, 117, 143, 0.3) !important;
    padding: 1.8rem !important;
    margin: 1rem 0 !important;
    box-sizing: border-box !important;
}

/* Bright File Uploader Styling */
[data-testid="stFileUploader"], section[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploadDropzone"], div[data-testid="stFileUploadDropzone"] {
    background: radial-gradient(circle at 50% 50%, #ffffff 0%, #fff0f5 100%) !important;
    border: 3px dashed #ff758f !important;
    border-radius: 24px !important;
    padding: 2.2rem 1.5rem !important;
    box-shadow: 0 10px 25px rgba(255, 117, 143, 0.25), 0 0 20px rgba(112, 214, 255, 0.2) inset !important;
}

[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] p {
    color: #4a0e4e !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploadDropzone"] button {
    background: linear-gradient(135deg, #ff758f 0%, #ff499e 100%) !important;
    color: #ffffff !important;
    border: 2px solid #ffffff !important;
    border-radius: 30px !important;
    font-weight: 800 !important;
    padding: 0.5rem 1.6rem !important;
    box-shadow: 0 4px 15px rgba(255, 73, 158, 0.4) !important;
}

/* Independent Loading Spell Chamber */
.spell-chamber {
    background: #ffffff !important;
    border: 4px solid #ff477e !important;
    border-radius: 32px !important;
    padding: 4rem 2rem !important;
    text-align: center !important;
    box-shadow: 0 0 50px rgba(255, 71, 126, 0.45) !important;
    margin: 3rem auto !important;
    max-width: 650px !important;
}

/* Cute Bouncing Loader Animation */
.cute-loader {
    display: flex;
    justify-content: center;
    gap: 25px;
    margin-bottom: 2rem;
}
.cute-icon {
    font-size: 4.8rem;
    animation: cuteBounce 0.8s infinite alternate ease-in-out;
    filter: drop-shadow(0px 10px 8px rgba(255, 112, 166, 0.45));
}
.cute-icon.delay-1 { animation-delay: 0.25s; }
.cute-icon.delay-2 { animation-delay: 0.5s; }

@keyframes cuteBounce {
    0% { transform: translateY(0) scale(1); }
    100% { transform: translateY(-30px) scale(1.15); }
}

/* Button Styling */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #ff477e 0%, #ff70a6 50%, #70d6ff 100%) !important;
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    border-radius: 50px !important;
    padding: 0.8rem 3.2rem !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 8px 25px rgba(255, 71, 126, 0.45) !important;
    transition: all 0.25s ease-in-out !important;
    margin: 1.5rem auto !important;
    display: block !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 12px 30px rgba(255, 71, 126, 0.6) !important;
}

/* Success Box Styling */
.success-box {
    background: #e6ffed !important;
    border: 2px solid #28a745 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    text-align: center !important;
    color: #155724 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. Independent Loading Page Renderer
# ---------------------------------------------------------
def render_loading_page(title, desc):
    play_fairy_magic_sfx()
    icons = random.choice([
        ("🦄", "✨", "🧚‍♀️"),
        ("🦉", "🌙", "⭐"),
        ("👑", "🪄", "🏰")
    ])
    st.markdown(f"""
    <div class="spell-chamber">
        <div class="cute-loader">
            <span class="cute-icon">{icons[0]}</span>
            <span class="cute-icon delay-1">{icons[1]}</span>
            <span class="cute-icon delay-2">{icons[2]}</span>
        </div>
        <h2 style="color: #d81159 !important; font-size: 2rem; white-space: nowrap;">{title}</h2>
        <p style="font-size: 1.3rem; color: #4a0e4e !important; font-weight: 800; margin: 1.2rem 0;">
            {desc}
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. Fast Transformer Models & Inference
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return proc, model

@st.cache_resource(show_spinner=False)
def load_story_model():
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

def get_caption_fast(image, proc, model):
    img_resized = image.copy()
    img_resized.thumbnail((224, 224))
    inputs = proc(images=img_resized, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=20)
    return proc.decode(out[0], skip_special_tokens=True)

def get_story_fast(caption, tok, model):
    clean_caption = caption.strip().rstrip(".")
    prompt = (
        f"Once upon a time, there was {clean_caption}. "
        f"Every day brought sweet laughter! One morning, a magical rainbow door opened. "
        f"With curious eyes, our brave little friend stepped inside to explore. "
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            min_new_tokens=40,
            max_new_tokens=65,
            do_sample=True,
            temperature=0.75,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    
    # Strict English text filtering
    clean_text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', raw).strip()
    
    last_period = max(clean_text.rfind("."), clean_text.rfind("!"), clean_text.rfind("?"))
    if last_period != -1:
        story = clean_text[:last_period + 1]
    else:
        story = clean_text + " And they lived happily ever after!"
    return story

def text_to_speech(text, filename="story_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 4. State Management
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "ch1"
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""


# ---------------------------------------------------------
# 5. Main Application State Machine
# ---------------------------------------------------------
def main():
    inject_global_features()

    # =========================================================
    # Chapter 1: The Magic Corner & Awakening the Magic Mirror
    # =========================================================
    if st.session_state.page == "ch1":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 1: The Magic Corner</p>", unsafe_allow_html=True)
        st.progress(0.25)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 1: The Magic Corner</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">
                Welcome to the Magic Corner! Upload your magical picture here to awaken the Magic Mirror.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "🌟 Drag & Drop or Browse your magical picture here:",
            type=["png", "jpg", "jpeg"],
            help="Choose a magical photo for your fairytale!",
            key="magic_uploader"
        )

        if uploaded_file is not None:
            play_fairy_magic_sfx()
            st.session_state.uploaded_img = Image.open(uploaded_file).convert("RGB")
            st.session_state.page = "load1"
            st.rerun()

    # =========================================================
    # Loading 1: Awakening the Magic Mirror
    # =========================================================
    elif st.session_state.page == "load1":
        render_loading_page(
            "Awakening the Magic Mirror...", 
            "The Magic Mirror is waking up to gaze deep into your picture!"
        )
        proc, model = load_caption_model()
        st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
        st.session_state.page = "ch2"
        st.rerun()

    # =========================================================
    # Chapter 2: The Magic Mirror Sees...
    # =========================================================
    elif st.session_state.page == "ch2":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 2: The Magic Mirror</p>", unsafe_allow_html=True)
        st.progress(0.50)
        st.balloons()
        time.sleep(1.2)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 2: The Magic Mirror Speaks!</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">The Magic Mirror sees its sacred vision:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state.uploaded_img, caption="Your Enchanted Picture", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; padding: 1.6rem; border-radius: 20px; border: 3px solid #70d6ff; text-align: center; margin-top: 1rem; box-shadow: 0 4px 15px rgba(112, 214, 255, 0.3);">
                <h3 style="color: #4a0e4e !important; margin: 0; white-space: nowrap;">✨ The Magic Mirror Sees:</h3>
                <p style="font-size: 1.35rem; font-weight: 800; color: #d81159 !important; margin-top: 0.8rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("📜 Weave a Fairytale 📜"):
                play_fairy_magic_sfx()
                st.session_state.page = "load2"
                st.rerun()

    # =========================================================
    # Loading 2: Weaving the Golden Scroll
    # =========================================================
    elif st.session_state.page == "load2":
        render_loading_page(
            "Weaving Golden Threads...", 
            "The royal elves are spinning the mirror's vision into a tale!"
        )
        tok, model = load_story_model()
        st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
        st.session_state.page = "ch3"
        st.rerun()

    # =========================================================
    # Chapter 3: The Golden Scroll
    # =========================================================
    elif st.session_state.page == "ch3":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 3: The Golden Scroll</p>", unsafe_allow_html=True)
        st.progress(0.75)
        st.snow()
        time.sleep(1.2)

        st.markdown("""
        <div class="magic-parchment">
            <h2>📜 Chapter 3: The Golden Story Scroll</h2>
        </div>
        """, unsafe_allow_html=True)

        img_col, text_col = st.columns([1, 1.2])
        with img_col:
            st.image(st.session_state.uploaded_img, caption="The Illustrated Scene", use_container_width=True)
        with text_col:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px dashed #ff758f; border-radius: 20px; padding: 1.6rem; min-height: 220px; box-shadow: 0 4px 15px rgba(255, 117, 143, 0.25);">
                <p style="font-size: 1.22rem; line-height: 1.85; color: #1d2129 !important; font-weight: 700; margin: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("🎶 Enter Voice Studio 🎶"):
                play_fairy_magic_sfx()
                st.session_state.page = "load3"
                st.rerun()

    # =========================================================
    # Loading 3: Tuning the Voice Harp
    # =========================================================
    elif st.session_state.page == "load3":
        render_loading_page(
            "Tuning the Voice Harp...", 
            "The singing fairies are preparing to whisper the tale aloud!"
        )
        st.session_state.audio_path = text_to_speech(st.session_state.story)
        st.session_state.page = "ch4"
        st.rerun()

    # =========================================================
    # Chapter 4: The Voice Harp Whispering
    # =========================================================
    elif st.session_state.page == "ch4":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()
        time.sleep(1.2)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">
                Listen closely as the fairy narrator whispers your custom bedtime story!
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state.uploaded_img, caption="The Storybook Scene", use_container_width=True)
        
        with col2:
            st.markdown('<div class="success-box">✨ Fairy Audio Narrated Successfully! ✨</div>', unsafe_allow_html=True)
            st.audio(st.session_state.audio_path, format="audio/mp3")

        st.markdown(f"""
        <div style="background: #ffffff; border: 2px dashed #ff758f; border-radius: 20px; padding: 1.6rem; margin-top: 1.2rem; box-shadow: 0 6px 20px rgba(255, 117, 143, 0.25);">
            <h3 style="color: #d81159 !important; margin-top: 0;">📖 Read Along with the Story:</h3>
            <p style="font-size: 1.22rem; line-height: 1.85; color: #1d2129 !important; font-weight: 700; margin-bottom: 0;">
                {st.session_state.story}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("🏰 Create New Fairytale"):
                play_fairy_magic_sfx()
                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    try: os.remove(st.session_state.audio_path)
                    except: pass
                st.session_state.page = "ch1"
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.rerun()

if __name__ == "__main__":
    main()
