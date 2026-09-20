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

# 開啟多執行緒加速純 CPU 運算 (大幅縮減至 10 秒內)
torch.set_num_threads(4)

# ---------------------------------------------------------
# 1. 頁面配置與全局音樂、樣式
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

def inject_global_audio_and_scroll():
    """注入背景音樂(BGM)與每次換頁置頂的腳本"""
    components.html(
        """
        <!-- BGM 音頻元素 -->
        <audio id="bgm" loop>
            <source src="https://cdn.pixabay.com/download/audio/2022/01/26/audio_d0c6ff1cb8.mp3" type="audio/mpeg">
        </audio>
        <script>
            // 確保每次渲染都在最頂端
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
            const mainSection = window.parent.document.querySelector('section.main');
            if (mainSection) { mainSection.scrollTo({top: 0, behavior: 'smooth'}); }

            // 破解瀏覽器阻擋：只要使用者在畫面點擊任何地方，就開始播放 BGM
            var bgm = document.getElementById("bgm");
            bgm.volume = 0.15; // 15% 小聲背景音
            window.parent.document.body.addEventListener('click', function() {
                if (bgm.paused) {
                    bgm.play().catch(e => console.log("BGM 播放等待中..."));
                }
            }, { once: true });
        </script>
        """,
        height=0
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

/* 全域夢幻背景 */
[data-testid="stAppViewContainer"], [data-testid="stHeader"], .stApp {
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

h2, h3, p { text-align: center; }

/* 童話內容卡片 */
.magic-parchment {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2.2rem !important;
    margin: 1rem 0 !important;
}

/* 獨立施法 Loading 頁面卡片 */
.spell-chamber {
    background: #ffffff;
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 4rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(255, 112, 166, 0.55), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    margin: 3rem auto;
    max-width: 650px;
}

/* 升級版：超級可愛的彈跳 Loading 動畫 */
.cute-loader {
    display: flex;
    justify-content: center;
    gap: 25px;
    margin-bottom: 2rem;
}
.cute-icon {
    font-size: 4.5rem;
    animation: cuteBounce 0.8s infinite alternate ease-in-out;
    filter: drop-shadow(0px 10px 8px rgba(255, 112, 166, 0.4));
}
.cute-icon.delay-1 { animation-delay: 0.25s; }
.cute-icon.delay-2 { animation-delay: 0.5s; }

@keyframes cuteBounce {
    0% { transform: translateY(0) scale(1); }
    100% { transform: translateY(-30px) scale(1.15); }
}

/* 按鈕樣式 */
div.stButton { display: flex; justify-content: center; margin: 1.5rem auto; }
div.stButton > button {
    background: linear-gradient(135deg, #ff758f 0%, #ff499e 40%, #70d6ff 100%) !important;
    color: #ffffff !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.8rem 3rem !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.45) !important;
    transition: all 0.2s ease-in-out !important;
}
div.stButton > button:hover {
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 0 12px 30px rgba(255, 73, 158, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. 獨立 Loading 畫面渲染器 (帶有魔法 Bling 音效)
# ---------------------------------------------------------
def render_loading_page(title, desc, status_text):
    """
    渲染獨立的 Loading 頁面，並自動播放魔法 Bling 音效！
    """
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
        <h2 style="color: #ff477e !important;">{title}</h2>
        <p style="font-size: 1.3rem; color: #6a0572; font-weight: bold; margin: 1rem 0;">
            {desc}
        </p>
        <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
            <p style="font-size: 1.15rem; color: #4361ee; font-weight: bold; margin: 0;">
                {status_text}
            </p>
        </div>
        <!-- 每次進入 Loading 頁面，自動播放一次清脆魔法音效 -->
        <audio autoplay>
            <source src="https://cdn.pixabay.com/download/audio/2021/08/04/audio_0625c1539c.mp3" type="audio/mpeg">
        </audio>
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
    img_resized.thumbnail((224, 224)) # 壓縮至 224x224，運算只需 3 秒！
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
            no_repeat_ngram_size=3, # 絕對防止模型重複產生迴圈或無意義亂碼
            pad_token_id=tok.eos_token_id
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    
    # 強力清理任何星號、底線或奇怪符號
    clean_text = re.sub(r'[*_#~\[\]`=]', '', raw).strip()
    
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
# 5. 主程式流程：真正的完全獨立替換 Loading 頁面
# ---------------------------------------------------------
def main():
    inject_global_audio_and_scroll()
    
    # 建立唯一的主畫面插槽
    # 每次點擊按鈕，都會透過 overwrite 這個容器，達到「瞬間換頁」且「不會疊加」的效果！
    main_view = st.empty()

    # =========================================================
    # Chapter 1: 上傳圖片
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

                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    # 1. 瞬間清空舊畫面
                    main_view.empty()
                    # 2. 立刻畫出獨立的 Loading 畫面並播放魔法音效
                    with main_view.container():
                        render_loading_page(
                            "Awakening the Mirror...", 
                            "The fairies are casting an enchantment over your picture!",
                            "🔮 Analyzing visual clues with fast vision transformer... 🔮"
                        )
                    # 3. 背景執行模型運算 (< 10秒)
                    proc, model = load_caption_model()
                    st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
                    
                    # 4. 運算結束，進入下一頁並刷新
                    st.session_state.page = "ch2"
                    st.rerun()

    # =========================================================
    # Chapter 2: 水晶球線索
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

            if st.button("📜 Weave a Fairytale 📜"):
                # 瞬間清空並進入 Loading 2
                main_view.empty()
                with main_view.container():
                    render_loading_page(
                        "Weaving Golden Threads...", 
                        "The royal elves are dipping quills into starlight ink!",
                        "📖 Writing your bedtime adventure story... 📖"
                    )
                # 執行運算
                tok, model = load_story_model()
                st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
                
                st.session_state.page = "ch3"
                st.rerun()

    # =========================================================
    # Chapter 3: 故事卷軸
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
                st.image(st.session_state.uploaded_img, caption="The Illustrated Scene", use_container_width=True)
            with text_col:
                st.markdown(f"""
                <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 18px; padding: 1.5rem; min-height: 220px; box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);">
                    <p style="font-size: 1.2rem; line-height: 1.85; color: #2b2d42; margin: 0;">
                        {st.session_state.story}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if st.button("🎶 Enter Voice Studio 🎶"):
                # 瞬間清空並進入 Loading 3
                main_view.empty()
                with main_view.container():
                    render_loading_page(
                        "Tuning the Fairyland Harp...", 
                        "The singing fairies are warming up their vocal cords!",
                        "🎙️ Synthesizing sweet bedtime voice... 🎙️"
                    )
                # 執行運算
                st.session_state.audio_path = text_to_speech(st.session_state.story)
                
                st.session_state.page = "ch4"
                st.rerun()

    # =========================================================
    # Chapter 4: 聲音播送與最終結果 (極簡化按鈕)
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
                st.image(st.session_state.uploaded_img, caption="The Storybook Scene", use_container_width=True)
            
            with col2:
                st.success("✨ Fairy Audio Narrated Successfully!")
                st.audio(st.session_state.audio_path, format="audio/mp3")

            st.markdown(f"""
            <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 20px; padding: 1.5rem; margin-top: 1.2rem; box-shadow: 0 6px 20px rgba(255, 182, 193, 0.2);">
                <h3 style="color: #ff477e !important; margin-top: 0;">📖 Read Along with the Story:</h3>
                <p style="font-size: 1.2rem; line-height: 1.85; color: #2b2d42; margin-bottom: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            # 只保留一個乾淨的重置按鈕
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
