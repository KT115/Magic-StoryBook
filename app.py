import os
import re
import random
import time
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS

torch.set_num_threads(4)

st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@600;700;800&family=Cinzel+Decorative:wght@700&display=swap');

[data-testid="stAppViewContainer"], .stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
}

h1, h2, h3 {
    color: #d81159 !important;
    text-align: center;
    word-break: break-word !important;
}

p, span, label, div {
    color: #1d2129 !important;
    font-weight: 600;
    word-break: break-word;
}

.magic-parchment {
    background: rgba(255, 255, 255, 0.96) !important;
    border-radius: 26px !important;
    border: 3px solid #ff758f !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    box-shadow: 0 12px 35px rgba(255, 117, 143, 0.3) !important;
}

.spell-chamber {
    background: #ffffff !important;
    border: 4px solid #ff477e !important;
    border-radius: 32px !important;
    padding: 4rem 2rem !important;
    text-align: center !important;
    margin: 3rem auto !important;
    max-width: 650px !important;
    box-shadow: 0 0 50px rgba(255, 71, 126, 0.45) !important;
}

div.stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 1.5rem 0 !important;
}

div.stButton > button {
    background: linear-gradient(135deg, #ff477e 0%, #ff70a6 50%, #70d6ff 100%) !important;
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    border-radius: 50px !important;
    padding: 0.8rem 3rem !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 8px 25px rgba(255, 71, 126, 0.45) !important;
    margin: 0 auto !important;
    display: block !important;
}

.success-box {
    background: #e6ffed !important;
    border: 2px solid #28a745 !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    text-align: center !important;
    color: #155724 !important;
    font-weight: 800 !important;
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    c_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    c_model.eval()
    
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    s_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    s_model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return proc, c_model, tok, s_model

def generate_caption(image, proc, model):
    img = image.copy()
    img.thumbnail((224, 224))
    inputs = proc(images=img, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=20)
    return proc.decode(out[0], skip_special_tokens=True)

def generate_story(caption, tok, model):
    prompt = f"Once upon a time, there was {caption.strip().rstrip('.')}. Every day brought sweet laughter! One morning, a magical rainbow door opened with brave friends ready to explore."
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, min_new_tokens=40, max_new_tokens=65, do_sample=True, temperature=0.75, top_k=40, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0], skip_special_tokens=True)
    clean = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', raw).strip()
    idx = max(clean.rfind("."), clean.rfind("!"), clean.rfind("?"))
    return clean[:idx+1] if idx != -1 else clean + " And they lived happily ever after!"

for key, val in [("page", "ch1"), ("img", None), ("caption", ""), ("story", ""), ("audio", "")]:
    if key not in st.session_state:
        st.session_state[key] = val

def main():
    proc, c_model, tok, s_model = load_models()
    
    if st.session_state.page == "ch1":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Chapter 1: The Magic Corner</h3>", unsafe_allow_html=True)
        st.progress(0.25)
        
        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 1: The Magic Corner</h2>
            <p>Welcome to the Magic Corner! Upload your magical picture here to awaken the Magic Mirror.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload magical picture:", type=["png", "jpg", "jpeg"], key="uploader")
        if uploaded:
            st.session_state.img = Image.open(uploaded).convert("RGB")
            st.session_state.page = "load1"
            st.rerun()

    elif st.session_state.page == "load1":
        st.markdown("""
        <div class="spell-chamber">
            <h2>Awakening the Magic Mirror...</h2>
            <p>The Magic Mirror is waking up to gaze deep into your picture!</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.caption = generate_caption(st.session_state.img, proc, c_model)
        st.session_state.page = "ch2"
        st.rerun()

    elif st.session_state.page == "ch2":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Chapter 2: The Magic Mirror</h3>", unsafe_allow_html=True)
        st.progress(0.50)
        st.balloons()
        time.sleep(1)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 2: The Magic Mirror Speaks!</h2>
            <p>The Magic Mirror sees its sacred vision:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.img, caption="Your Enchanted Picture", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; padding: 1.6rem; border-radius: 20px; border: 3px solid #70d6ff; text-align: center;">
                <h3>✨ The Magic Mirror Sees:</h3>
                <p style="font-size: 1.25rem; color: #d81159 !important;">"{st.session_state.caption.capitalize()}"</p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("📜 Weave a Fairytale 📜"):
            st.session_state.page = "load2"
            st.rerun()

    elif st.session_state.page == "load2":
        st.markdown("""
        <div class="spell-chamber">
            <h2>Weaving Golden Threads...</h2>
            <p>The royal elves are spinning the mirror's vision into a tale!</p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.story = generate_story(st.session_state.caption, tok, s_model)
        st.session_state.page = "ch3"
        st.rerun()

    elif st.session_state.page == "ch3":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Chapter 3: The Golden Scroll</h3>", unsafe_allow_html=True)
        st.progress(0.75)
        st.snow()
        time.sleep(1)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 3: The Golden Story Scroll</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.img, caption="Illustrated Scene", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 2px dashed #ff758f; border-radius: 20px; padding: 1.6rem;">
                <p style="font-size: 1.15rem; line-height: 1.7;">{st.session_state.story}</p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🎶 Enter Voice Studio 🎶"):
            st.session_state.page = "load3"
            st.rerun()

    elif st.session_state.page == "load3":
        st.markdown("""
        <div class="spell-chamber">
            <h2>Tuning the Voice Harp...</h2>
            <p>The singing fairies are preparing to whisper the tale aloud!</p>
        </div>
        """, unsafe_allow_html=True)
        tts = gTTS(text=st.session_state.story, lang="en")
        st.session_state.audio = "story_audio.mp3"
        tts.save(st.session_state.audio)
        st.session_state.page = "ch4"
        st.rerun()

    elif st.session_state.page == "ch4":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Chapter 4: The Voice Harp</h3>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()
        time.sleep(1)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p>Listen closely as the fairy narrator whispers your custom bedtime story!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.img, caption="Storybook Scene", use_container_width=True)
        with col2:
            st.markdown('<div class="success-box">✨ Fairy Audio Narrated Successfully! ✨</div>', unsafe_allow_html=True)
            st.audio(st.session_state.audio, format="audio/mp3")

        st.markdown(f"""
        <div class="magic-parchment">
            <h3>📖 Read Along:</h3>
            <p style="font-size: 1.15rem; line-height: 1.7;">{st.session_state.story}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🏰 Create New Fairytale"):
            if st.session_state.audio and os.path.exists(st.session_state.audio):
                try: os.remove(st.session_state.audio)
                except: pass
            st.session_state.page = "ch1"
            st.session_state.img = None
            st.session_state.caption = ""
            st.session_state.story = ""
            st.session_state.audio = ""
            st.rerun()

if __name__ == "__main__":
    main()
