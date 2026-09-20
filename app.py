import os
import re
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

# 開啟多執行緒加速純 CPU 運算
torch.set_num_threads(4)

# ---------------------------------------------------------
# 1. 頁面配置與全局樣式 (修復上傳區黑色底色、單行標題、按鈕置中)
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

def inject_global_features():
    """無感啟動 BGM 與每次切換平滑置頂"""
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
                bgm.play().catch(e => console.log("等待使用者互動解鎖 BGM..."));
            }
            window.parent.document.body.addEventListener('click', function() {
                if (bgm && bgm.paused) {
                    bgm.play().catch(e => console.log("BGM 播放中..."));
                }
            }, { once: true });
        </script>
        """,
        height=0
    )

def play_fairy_magic_sfx():
    """仙子魔法棒晶亮叮鈴聲"""
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

/* 全域背景 */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
}

/* 標題強制單行顯示 (Title in one line) 避免折行 */
h1 {
    font-family: 'Cinzel Decorative', cursive !important;
    color: #d81159 !important;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.95) !important;
    text-align: center;
    font-size: 2rem !important;
    font-weight: 800 !important;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
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

/* 內容卡片 */
.magic-parchment {
    background: rgba(255, 255, 255, 0.96) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ff758f !important;
    box-shadow: 0 12px 35px rgba(255, 117, 143, 0.3) !important;
    padding: 2.2rem !important;
    margin: 1.2rem 0 !important;
}

/* ---------------------------------------------------------
   獨角獸風格 Drag & Drop 檔案上傳區美化 (徹底覆蓋黑色預設底色)
   --------------------------------------------------------- */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploadDropzone"] {
    background: radial-gradient(circle at 50% 50%, #ffffff 0%, #fff0f5 100%) !important;
    border: 3px dashed #ff758f !important;
    border-radius: 24px !important;
    padding: 2.2rem 1.5rem !important;
    box-shadow: 0 10px 25px rgba(255, 117, 143, 0.25), 0 0 20px rgba(112, 214, 255, 0.2) inset !important;
    transition: all 0.3s ease-in-out !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #ff477e !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px rgba(255, 71, 126, 0.35), 0 0 25px rgba(112, 214, 255, 0.35) inset !important;
}

[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
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

/* 獨立施法 Loading 頁面卡片 */
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

/* 可愛彈跳 Loading 動畫 */
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

/* 所有按鈕強制左右居中 */
.stButton, div[data-testid="stButton"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
    margin: 1.5rem auto !important;
}

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
    margin: 0 auto !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) scale(1.04) !important;
    box-shadow: 0 12px 30px rgba(255, 71, 126, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. 獨立 Loading 畫面渲染器
# ---------------------------------------------------------
def render_loading_page(title, desc, status_text):
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
        <h2 style="color: #d81159 !important; font-size: 2rem;">{title}</h2>
        <p style="font-size: 1.3rem; color: #4a0e4e !important; font-weight: 800; margin: 1.2rem 0;">
            {desc}
        </p>
        <div style="background: rgba(255, 245, 248, 0.95); border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ff758f;">
            <p style="font-size: 1.15rem; color: #2b2d42 !important; font-weight: 700; margin: 0;">
                {status_text}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. 極速 Transformer 模型與推論
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
    clean_text = re.sub(r'[*_#~\[\]`<>=]', '', raw).strip()
    
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
# 4. 狀態管理
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
# 5. 狀態機主程式 (實現自動跳轉、左右置中、獨立 Loading)
# ---------------------------------------------------------
def main():
    inject_global_features()

    # =========================================================
    # 獨立頁面 1：Chapter 1（拖放/點擊上傳圖片，上傳即自動跳轉）
    # =========================================================
    if st.session_state.page == "ch1":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 1: The Magic Portal</p>", unsafe_allow_html=True)
        st.progress(0.25)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 1: The Magic Portal</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">
                Drag & Drop or browse a picture of a cute pet, toy, or drawing!
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "🌟 Drag and drop your image here, or click to browse:",
            type=["png", "jpg", "jpeg"],
            help="Choose a magical photo for your fairytale!"
        )

        if uploaded_file is not None:
            st.session_state.uploaded_img = Image.open(uploaded_file).convert("RGB")
            # 上傳後自動切換至獨立 Loading 頁面，不需手動按鈕
            st.session_state.page = "load1"
            st.rerun()

    # =========================================================
    # 獨立頁面 2：Loading 1（完全獨立的魔鏡施法頁面）
    # =========================================================
    elif st.session_state.page == "load1":
        render_loading_page(
            "Awakening the Mirror...", 
            "The fairies are casting an enchantment over your picture!",
            "🔮 Analyzing visual clues with fast vision transformer... 🔮"
        )
        proc, model = load_caption_model()
        st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
        st.session_state.page = "ch2"
        st.rerun()

    # =========================================================
    # 獨立頁面 3：Chapter 2（水晶球線索）
    # =========================================================
    elif st.session_state.page == "ch2":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 2: The Crystal Ball</p>", unsafe_allow_html=True)
        st.progress(0.50)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 2: The Crystal Ball Speaks!</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">Here is the secret clue discovered from your picture:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #ffffff; padding: 1.6rem; border-radius: 20px; border: 3px solid #70d6ff; text-align: center; margin-top: 1rem; box-shadow: 0 4px 15px rgba(112, 214, 255, 0.3);">
                <h3 style="color: #4a0e4e !important; margin: 0;">✨ Mirror Revelation:</h3>
                <p style="font-size: 1.35rem; font-weight: 800; color: #d81159 !important; margin-top: 0.8rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        if st.button("📜 Weave a Fairytale 📜"):
            st.session_state.page = "load2"
            st.rerun()

    # =========================================================
    # 獨立頁面 4：Loading 2（完全獨立、分離的獨立施法頁面）
    # =========================================================
    elif st.session_state.page == "load2":
        render_loading_page(
            "Weaving Golden Threads...", 
            "The royal elves are dipping quills into starlight ink!",
            "📖 Writing your bedtime adventure story... 📖"
        )
        tok, model = load_story_model()
        st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
        st.session_state.page = "ch3"
        st.rerun()

    # =========================================================
    # 獨立頁面 5：Chapter 3（故事卷軸，按鈕左右置中）
    # =========================================================
    elif st.session_state.page == "ch3":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 3: The Golden Scroll</p>", unsafe_allow_html=True)
        st.progress(0.75)
        st.snow()

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

        # 左右分中對齊的按鈕
        if st.button("🎶 Enter Voice Studio 🎶"):
            st.session_state.page = "load3"
            st.rerun()

    # =========================================================
    # 獨立頁面 6：Loading 3（完全獨立的豎琴調音頁面）
    # =========================================================
    elif st.session_state.page == "load3":
        render_loading_page(
            "Tuning the Fairyland Harp...", 
            "The singing fairies are warming up their vocal cords!",
            "🎙️ Synthesizing sweet bedtime voice... 🎙️"
        )
        st.session_state.audio_path = text_to_speech(st.session_state.story)
        st.session_state.page = "ch4"
        st.rerun()

    # =========================================================
    # 獨立頁面 7：Chapter 4（聲音播送與最終成果）
    # =========================================================
    elif st.session_state.page == "ch4":
        st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #4a0e4e; font-size: 1.15rem; font-weight: 800; white-space: nowrap;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p style="font-size: 1.2rem; text-align: center; color: #1d2129 !important;">
                Listen to the fairy narrator recite your custom bedtime story!
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state.uploaded_img, caption="The Storybook Scene", use_container_width=True)
        
        with col2:
            st.success("✨ Fairy Audio Narrated Successfully!")
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
        if st.button("🏰 Create New Fairytale"):
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
