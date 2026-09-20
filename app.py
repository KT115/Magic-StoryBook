import os
import random
import streamlit as st
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
# 頁面配置與樣式
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

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
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2rem !important;
    margin: 1.2rem 0 !important;
}

.spell-chamber {
    background: radial-gradient(circle, rgba(255,255,255,0.98) 0%, rgba(255,235,245,0.95) 100%);
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 3rem 1.8rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(255, 112, 166, 0.65), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    animation: pulseChamber 2.5s infinite alternate;
    margin: 2rem auto;
    max-width: 680px;
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

@keyframes spinBreathe {
    0% { transform: rotate(0deg) scale(0.9); box-shadow: 0 0 20px #ff70a6; }
    50% { transform: rotate(180deg) scale(1.15); box-shadow: 0 0 45px #70d6ff; }
    100% { transform: rotate(360deg) scale(0.9); box-shadow: 0 0 20px #ffd166; }
}

@keyframes pulseChamber {
    0% { box-shadow: 0 0 25px rgba(255, 112, 166, 0.4); }
    100% { box-shadow: 0 0 55px rgba(112, 214, 255, 0.85); }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. 模型載入與快取
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
# 2. 推論函式
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
# 3. 狀態管理
# ---------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "ch1"
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False
if "uploaded_img" not in st.session_state:
    st.session_state.uploaded_img = None
if "caption" not in st.session_state:
    st.session_state.caption = ""
if "story" not in st.session_state:
    st.session_state.story = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""


# ---------------------------------------------------------
# 4. 主流程路由
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)

    # =========================================================
    # CHAPTER 1: 上傳圖片
    # =========================================================
    if st.session_state.page == "ch1":
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
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    st.session_state.page = "load1"
                    st.session_state.is_loading = False  # 重設運算標記
                    st.rerun()

    # =========================================================
    # LOADING 1: 獨立全螢幕施法頁面
    # =========================================================
    elif st.session_state.page == "load1":
        q, _ = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">✨</div>
            <h2 style="color: #ff477e !important;">Awakening the Mirror Vision...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The fairies are casting an enchantment over your picture!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.4rem; margin: 1.4rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>💨 Breathe with the glowing orb: inhale, exhale, and blow soft magic dust!</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.05rem;">🔮 Analyzing visual clues with BLIP vision transformer... 🔮</p>
        </div>
        """, unsafe_allow_html=True)

        # 雙階段渲染保證：第一拍只畫 Loading 畫面，第二拍才執行運算
        if not st.session_state.is_loading:
            st.session_state.is_loading = True
            st.rerun()
        else:
            proc, model = load_caption_model()
            st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
            st.session_state.is_loading = False
            st.session_state.page = "ch2"
            st.rerun()

    # =========================================================
    # CHAPTER 2: 水晶球線索
    # =========================================================
    elif st.session_state.page == "ch2":
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
            if st.button("📜 Weave a Fairytale from This Clue! 📜"):
                st.session_state.page = "load2"
                st.session_state.is_loading = False
                st.rerun()

    # =========================================================
    # LOADING 2: 故事生成獨立施法頁面
    # =========================================================
    elif st.session_state.page == "load2":
        q, _ = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">📜</div>
            <h2 style="color: #ff477e !important;">Weaving Golden Story Threads...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The royal elves are dipping enchanted quills into starlight ink!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.4rem; margin: 1.4rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>✨ Chant along: "Abracadabra, alakazam, weave a story as fast as you can!" ✨</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.05rem;">📖 Generating fairytale narrative with text-transformer... 📖</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.is_loading:
            st.session_state.is_loading = True
            st.rerun()
        else:
            tok, model = load_story_model()
            st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
            st.session_state.is_loading = False
            st.session_state.page = "ch3"
            st.rerun()

    # =========================================================
    # CHAPTER 3: 黃金故事卷軸
    # =========================================================
    elif st.session_state.page == "ch3":
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
            if st.button("🎶 Proceed to Voice Harp (Chapter 4)"):
                st.session_state.page = "load3"
                st.session_state.is_loading = False
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
    # LOADING 3: 調音獨立施法頁面
    # =========================================================
    elif st.session_state.page == "load3":
        q, _ = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">🎶</div>
            <h2 style="color: #ff477e !important;">Tuning the Fairyland Harp...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The singing fairies are warming up their vocal cords to narrate your tale!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.4rem; margin: 1.4rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>🎵 Listen closely to the magic chimes in the air! 🎵</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.05rem;">✨ Preparing sound chamber... ✨</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.is_loading:
            st.session_state.is_loading = True
            st.rerun()
        else:
            st.session_state.is_loading = False
            st.session_state.page = "ch4"
            st.rerun()

    # =========================================================
    # CHAPTER 4: 聲音之琴與故事播送
    # =========================================================
    elif st.session_state.page == "ch4":
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
            if not st.session_state.audio_path or not os.path.exists(st.session_state.audio_path):
                st.markdown("<p style='text-align:center;'>Click below to summon the fairy narrator!</p>", unsafe_allow_html=True)
                if st.button("🧚 Cast Voice Spell"):
                    st.session_state.page = "load4"
                    st.session_state.is_loading = False
                    st.rerun()
            else:
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
                st.rerun()
        with btn_c2:
            if st.button("📜 Back to Story Scroll"):
                st.session_state.page = "ch3"
                st.rerun()

    # =========================================================
    # LOADING 4: 語音合成獨立施法頁面
    # =========================================================
    elif st.session_state.page == "load4":
        q, _ = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">🎙️</div>
            <h2 style="color: #ff477e !important;">Recording the Fairy Narration...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                Sprinkling vocal dust and recording the story audio...
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.4rem; margin: 1.4rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.25rem; color: #2b2d42; font-weight: bold;">{q}</p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.05rem;">✨ Magic microphone is listening! ✨</p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.is_loading:
            st.session_state.is_loading = True
            st.rerun()
        else:
            st.session_state.audio_path = text_to_speech(st.session_state.story)
            st.session_state.is_loading = False
            st.session_state.page = "ch4"
            st.rerun()


if __name__ == "__main__":
    main()
