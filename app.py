import os
import random
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

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

def scroll_to_top():
    components.html(
        """
        <script>
            function doScroll() {
                const doc = window.parent.document;
                const targets = [
                    doc.documentElement,
                    doc.body,
                    doc.querySelector('section.main'),
                    doc.querySelector('[data-testid="stAppViewContainer"]')
                ];
                targets.forEach(el => {
                    if (el) el.scrollTop = 0;
                });
                window.parent.scrollTo(0, 0);
            }
            doScroll();
            setTimeout(doScroll, 80);
        </script>
        """,
        height=0
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
    color: #4a4e69 !important;
}

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

.magic-parchment {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2.2rem !important;
    margin: 1rem 0 !important;
}

.spell-chamber {
    background: #ffffff;
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 3.5rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(255, 112, 166, 0.55), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    margin: 2rem auto;
    max-width: 650px;
}

.magic-orb {
    display: inline-block;
    width: 95px;
    height: 95px;
    border-radius: 50%;
    background: linear-gradient(45deg, #ff70a6, #ffd166, #70d6ff);
    box-shadow: 0 0 35px rgba(255, 112, 166, 0.85);
    animation: spinBreathe 3s infinite ease-in-out;
    margin-bottom: 1.2rem;
    line-height: 95px;
    font-size: 3rem;
}

div[data-testid="stColumn"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

div.stButton {
    display: flex;
    justify-content: center;
    margin: 1rem auto;
}

div.stButton > button {
    background: linear-gradient(135deg, #ff758f 0%, #ff499e 40%, #70d6ff 100%) !important;
    color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.6rem !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 6px 20px rgba(255, 107, 107, 0.35) !important;
    transition: all 0.25s ease-in-out !important;
}

div.stButton > button:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 10px 25px rgba(255, 73, 158, 0.55) !important;
}

div.stButton > button:disabled {
    background: #cccccc !important;
    color: #888888 !important;
    border-color: #bbbbbb !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}

@keyframes spinBreathe {
    0% { transform: rotate(0deg) scale(0.9); box-shadow: 0 0 20px #ff70a6; }
    50% { transform: rotate(180deg) scale(1.15); box-shadow: 0 0 45px #70d6ff; }
    100% { transform: rotate(360deg) scale(0.9); box-shadow: 0 0 20px #ffd166; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Models Initialization (Cached)
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


# ---------------------------------------------------------
# 2. Fast Inference Functions
# ---------------------------------------------------------
def get_caption_fast(image, proc, model):
    img_resized = image.copy()
    img_resized.thumbnail((384, 384))
    inputs = proc(images=img_resized, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=25)
    return proc.decode(out[0], skip_special_tokens=True)


def get_story_fast(caption, tok, model):
    clean_caption = caption.strip().rstrip(".")
    prompt = (
        f"Once upon a time, there was {clean_caption}. "
        f"Every day brought sweet laughter! "
        f"One morning, a magical rainbow door opened with sparkles. "
        f"With curious eyes, our brave little friend stepped inside to explore. "
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            min_new_tokens=45,
            max_new_tokens=70,
            do_sample=True,
            temperature=0.75,
            top_k=30,
            top_p=0.85,
            repetition_penalty=1.25,
            pad_token_id=tok.eos_token_id
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    last_period = max(raw.rfind("."), raw.rfind("!"), raw.rfind("?"))
    if last_period != -1:
        story = raw[:last_period + 1]
    else:
        story = raw + " And they lived happily ever after!"
    return story


def text_to_speech(text, filename="story_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up and never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]


# ---------------------------------------------------------
# 3. State Management & Callbacks (Solves Issue 2: Double Click)
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "ch1"
if "start_task" not in st.session_state:
    st.session_state.start_task = False
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""

# Callback Function to lock the button instantly
def start_magic():
    st.session_state.start_task = True


# ---------------------------------------------------------
# 4. Main Application Router
# ---------------------------------------------------------
def main():
    scroll_to_top()
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    
    main_view = st.empty()

    # =========================================================
    # CHAPTER 1: Image Upload
    # =========================================================
    if st.session_state.page == "ch1":
        with main_view.container():
            st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 1: The Magic Portal</p>", unsafe_allow_html=True)
            st.progress(0.25)

            st.markdown("""
            <div class="magic-parchment">
                <h2>🔮 Chapter 1: The Magic Portal</h2>
                <p style="font-size: 1.15rem; text-align: center;">
                    Feed a picture of a cute pet, drawing, or toy to the magical portal!
                </p>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Choose a picture for your bedtime story:", type=["png", "jpg", "jpeg"])

            if uploaded_file is not None:
                st.session_state.uploaded_img = Image.open(uploaded_file).convert("RGB")
                st.image(st.session_state.uploaded_img, caption="Your Enchanted Picture", use_container_width=True)

                _, btn_c, _ = st.columns([1, 2, 1])
                with btn_c:
                    if st.button("🪄 Enter the Mirror Chamber (Next) 🪄"):
                        st.session_state.page = "chamber1"
                        st.session_state.start_task = False
                        st.rerun()

    # =========================================================
    # CHAMBER 1 (LOADING PAGE 1): Decode Picture
    # =========================================================
    elif st.session_state.page == "chamber1":
        with main_view.container():
            q, _ = random.choice(RIDDLES)
            st.markdown(f"""
            <div class="spell-chamber">
                <div class="magic-orb">✨</div>
                <h2 style="color: #ff477e !important; margin: 0 0 0.5rem 0;">The Mirror Chamber</h2>
                <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold; margin: 0.5rem 0 1.2rem 0;">
                    Welcome! The fairies are ready to cast an enchantment over your picture.
                </p>
                <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.4rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                    <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                    <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold; margin: 0.5rem 0;">{q}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                # Issue 2 Fix: Disable button after click using callback
                st.button("✨ Cast Magic: Decode Picture ✨", on_click=start_magic, disabled=st.session_state.start_task)
                
            if st.session_state.start_task:
                with st.spinner("🔮 Analyzing visual clues with BLIP vision transformer... 🔮"):
                    proc, model = load_caption_model()
                    st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
                
                # Reset task state and move to next chapter
                st.session_state.start_task = False
                st.session_state.page = "ch2"
                st.rerun()

    # =========================================================
    # CHAPTER 2: The Crystal Ball
    # =========================================================
    elif st.session_state.page == "ch2":
        with main_view.container():
            st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 2: The Crystal Ball</p>", unsafe_allow_html=True)
            st.progress(0.50)
            st.balloons()

            st.markdown("""
            <div class="magic-parchment">
                <h2>🔮 Chapter 2: The Crystal Ball Speaks!</h2>
                <p style="font-size: 1.15rem; text-align: center;">Here is the secret clue discovered from your picture:</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.session_state.uploaded_img:
                    st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
            with col2:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.92); padding: 1.5rem; border-radius: 18px; border: 2px solid #70d6ff; text-align: center; margin-top: 1rem;">
                    <h3 style="color: #6a0572 !important; margin: 0;">✨ Mirror Revelation:</h3>
                    <p style="font-size: 1.3rem; font-weight: bold; color: #ff477e; margin-top: 0.6rem;">
                        "{st.session_state.caption.capitalize()}"
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("📜 Enter the Story Forge (Next) 📜"):
                    st.session_state.page = "chamber2"
                    st.session_state.start_task = False
                    st.rerun()

    # =========================================================
    # CHAMBER 2 (LOADING PAGE 2): Weave Fairytale
    # =========================================================
    elif st.session_state.page == "chamber2":
        with main_view.container():
            q, _ = random.choice(RIDDLES)
            st.markdown(f"""
            <div class="spell-chamber">
                <div class="magic-orb">📜</div>
                <h2 style="color: #ff477e !important; margin: 0 0 0.5rem 0;">The Story Forge</h2>
                <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold; margin: 0.5rem 0 1.2rem 0;">
                    The royal elves are ready with their starlight ink!
                </p>
                <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.4rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                    <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                    <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold; margin: 0.5rem 0;">{q}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                # Issue 2 Fix: Disable button after click using callback
                st.button("✨ Cast Magic: Weave Fairytale ✨", on_click=start_magic, disabled=st.session_state.start_task)
            
            if st.session_state.start_task:
                with st.spinner("📖 Generating fairytale narrative with text-transformer... 📖"):
                    tok, model = load_story_model()
                    st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
                
                # Reset task state and move to next chapter
                st.session_state.start_task = False
                st.session_state.page = "ch3"
                st.rerun()

    # =========================================================
    # CHAPTER 3: The Golden Scroll
    # =========================================================
    elif st.session_state.page == "ch3":
        with main_view.container():
            st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 3: The Golden Scroll</p>", unsafe_allow_html=True)
            st.progress(0.75)
            st.snow()

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
                if st.button("🎶 Enter the Voice Studio (Next) 🎶"):
                    st.session_state.page = "chamber3"
                    st.session_state.start_task = False
                    st.rerun()
            with btn_c2:
                if st.button("🔄 Try Another Picture"):
                    st.session_state.page = "ch1"
                    st.session_state.uploaded_img = None
                    st.session_state.caption = ""
                    st.session_state.story = ""
                    st.session_state.audio_path = ""
                    st.rerun()

    # =========================================================
    # CHAMBER 3 (LOADING PAGE 3): Record Audio
    # =========================================================
    elif st.session_state.page == "chamber3":
        with main_view.container():
            q, _ = random.choice(RIDDLES)
            st.markdown(f"""
            <div class="spell-chamber">
                <div class="magic-orb">🎶</div>
                <h2 style="color: #ff477e !important; margin: 0 0 0.5rem 0;">The Voice Studio</h2>
                <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold; margin: 0.5rem 0 1.2rem 0;">
                    The singing fairies are warming up their vocal cords!
                </p>
                <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.4rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                    <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                    <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold; margin: 0.5rem 0;">{q}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                # Issue 2 Fix: Disable button after click using callback
                st.button("✨ Cast Magic: Record Audio ✨", on_click=start_magic, disabled=st.session_state.start_task)

            if st.session_state.start_task:
                with st.spinner("🎙️ Sprinkling vocal dust and recording audio... 🎙️"):
                    st.session_state.audio_path = text_to_speech(st.session_state.story)
                
                # Reset task state and move to next chapter
                st.session_state.start_task = False
                st.session_state.page = "ch4"
                st.rerun()

    # =========================================================
    # CHAPTER 4: The Voice Harp
    # =========================================================
    elif st.session_state.page == "ch4":
        with main_view.container():
            st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
            st.progress(1.0)
            st.balloons()

            st.markdown("""
            <div class="magic-parchment">
                <h2>🎶 Chapter 4: The Voice Harp</h2>
                <p style="font-size: 1.15rem; text-align: center;">
                    Listen to the fairy narrator recite your custom bedtime story!
                </p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.session_state.uploaded_img:
                    st.image(st.session_state.uploaded_img, caption="The Storybook Scene", use_container_width=True)
            
            with col2:
                if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                    st.success("✨ Fairy Audio Narrated Successfully!")
                    st.audio(st.session_state.audio_path, format="audio/mp3")

            st.markdown(f"""
            <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 20px; padding: 1.4rem; margin-top: 1.2rem; box-shadow: 0 6px 20px rgba(255, 182, 193, 0.2);">
                <h3 style="color: #ff477e !important; margin-top: 0;">📖 Read Along with the Story:</h3>
                <p style="font-size: 1.18rem; line-height: 1.85; color: #2b2d42; margin-bottom: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                if st.button("🏰 Create a Brand New Fairytale"):
                    if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                        try:
                            os.remove(st.session_state.audio_path)
                        except Exception:
                            pass
                    st.session_state.page = "ch1"
                    st.session_state.uploaded_img = None
                    st.session_state.caption = ""
                    st.session_state.story = ""
                    st.session_state.audio_path = ""
                    st.session_state.start_task = False
                    st.rerun()
            with btn_c2:
                if st.button("📜 Back to Story Scroll"):
                    st.session_state.page = "ch3"
                    st.rerun()

if __name__ == "__main__":
    main()
