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
    AutoModelForSeq2SeqLM
)
from gtts import gTTS

# ---------------------------------------------------------
# 1. 頁面配置與 BGM、捲動控制
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

def play_bgm():
    """注入童話風背景音樂，微小音量 (15%) 循環播放"""
    components.html(
        """
        <audio id="bgm" loop autoplay>
            <source src="https://cdn.pixabay.com/download/audio/2022/01/26/audio_d0c6ff1cb8.mp3" type="audio/mpeg">
        </audio>
        <script>
            var audio = document.getElementById("bgm");
            audio.volume = 0.15;
            let playPromise = audio.play();
            if (playPromise !== undefined) {
                playPromise.catch(function(error) {
                    console.log("Waiting for user click to play BGM...");
                });
            }
        </script>
        """,
        height=0
    )

def scroll_to_top():
    """每次換頁強制畫面回到最頂端"""
    components.html(
        """
        <script>
            function doScroll() {
                const doc = window.parent.document;
                const targets = [doc.documentElement, doc.body, doc.querySelector('section.main')];
                targets.forEach(el => { if (el) el.scrollTop = 0; });
                window.parent.scrollTo(0, 0);
            }
            doScroll();
            setTimeout(doScroll, 100);
        </script>
        """,
        height=0
    )

def auto_trigger_task(trigger_key):
    """注入 JavaScript，在 0.5 秒後自動點擊隱藏的運算按鈕"""
    components.html(
        f"""
        <script>
        setTimeout(function() {{
            const btns = window.parent.document.querySelectorAll('button');
            btns.forEach(b => {{
                if (b.innerText.includes('{trigger_key}')) {{
                    b.click();
                }}
            }});
        }}, 500);
        </script>
        """,
        height=0
    )

# ---------------------------------------------------------
# 2. 童趣風 CSS 樣式 (可愛彈跳動畫 & 隱藏機制)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

/* 全域背景 */
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

/* 童話卡片 */
.magic-parchment {
    background: rgba(255, 255, 255, 0.92) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2.2rem !important;
    margin: 1rem 0 !important;
}

/* 獨立施法室頁面卡片 */
.spell-chamber {
    background: #ffffff;
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 3rem 2rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(255, 112, 166, 0.55), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    margin: 2rem auto;
    max-width: 650px;
}

/* 童趣彈跳動畫 */
.cute-loader {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 1.5rem;
}
.cute-icon {
    font-size: 4rem;
    animation: cuteBounce 1s infinite alternate ease-in-out;
}
.cute-icon.delay-1 { animation-delay: 0.3s; }
.cute-icon.delay-2 { animation-delay: 0.6s; }

@keyframes cuteBounce {
    0% { transform: translateY(0) scale(1); }
    100% { transform: translateY(-25px) scale(1.15); }
}

/* 按鈕樣式 */
div.stButton { display: flex; justify-content: center; margin: 1rem auto; }
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

/* 徹底隱藏包含 🚀 符號的自動觸發按鈕 */
div:has(button:contains('🚀')) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. 極速 Transformer 模型 (大幅降低 Loading 時間)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    """視覺模型：將圖片壓縮至 224x224 加速"""
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    return proc, model

@st.cache_resource(show_spinner=False)
def load_story_model():
    """故事模型：更換為 Flan-T5，速度極快，且不會產生亂碼 *"""
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    model.eval()
    return tok, model


def get_caption_fast(image, proc, model):
    img_resized = image.copy()
    img_resized.thumbnail((224, 224)) # 更小的尺寸換取更高的速度
    inputs = proc(images=img_resized, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=15)
    return proc.decode(out[0], skip_special_tokens=True)


def get_story_fast(caption, tok, model):
    clean_caption = caption.strip().rstrip(".")
    prompt = f"Write a happy bedtime story for kids about {clean_caption}. Once upon a time, "
    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.2
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
    # 清理任何可能的奇怪符號 (過濾 *, _, # 等)
    clean_text = re.sub(r'[*_#~]', '', raw).strip()
    
    last_period = max(clean_text.rfind("."), clean_text.rfind("!"), clean_text.rfind("?"))
    if last_period != -1:
        story = clean_text[:last_period + 1]
    else:
        story = clean_text + " And they lived happily ever after."
        
    return "Once upon a time, " + story


def text_to_speech(text, filename="story_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# 童趣謎題
RIDDLES = [
    "🦄 What has hands but cannot clap? ... A clock! ⏰",
    "🌟 What gets wetter the more it dries? ... A towel! 🛁",
    "🧚 What goes up and never comes down? ... Your age! 🎂"
]

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
# 5. 主程式流程：純淨換頁與全自動 Loading
# ---------------------------------------------------------
def main():
    play_bgm()
    scroll_to_top()
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
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

                _, btn_c, _ = st.columns([1, 2, 1])
                with btn_c:
                    if st.button("🪄 Awaken the Magic Mirror 🪄"):
                        st.session_state.page = "load1"
                        st.rerun()

    # =========================================================
    # Loading 1: 解讀圖片 (自動過渡，無按鈕)
    # =========================================================
    elif st.session_state.page == "load1":
        # 隱藏按鈕被自動點擊後觸發運算
        if st.session_state.get("trigger_load1"):
            st.session_state.trigger_load1 = False
            proc, model = load_caption_model()
            st.session_state.caption = get_caption_fast(st.session_state.uploaded_img, proc, model)
            st.session_state.page = "ch2"
            st.rerun()
        
        # 否則顯示可愛 Loading 動畫並準備自動觸發
        else:
            with main_view.container():
                riddle = random.choice(RIDDLES)
                st.markdown(f"""
                <div class="spell-chamber">
                    <div class="cute-loader">
                        <span class="cute-icon">🦄</span>
                        <span class="cute-icon delay-1">✨</span>
                        <span class="cute-icon delay-2">🧚‍♀️</span>
                    </div>
                    <h2 style="color: #ff477e !important;">Awakening the Mirror Vision...</h2>
                    <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                        The fairies are casting an enchantment over your picture!
                    </p>
                    <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                        <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                        <p style="font-size: 1.15rem; color: #2b2d42; font-weight: bold; margin: 0;">{riddle}</p>
                    </div>
                    <p style="color: #4361ee; font-weight: bold;">🔮 Looking closely at your picture... 🔮</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.button("🚀_1", key="trigger_load1")
                auto_trigger_task("🚀_1")

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

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("📜 Weave a Fairytale 📜"):
                    st.session_state.page = "load2"
                    st.rerun()

    # =========================================================
    # Loading 2: 生成童話故事 (自動過渡，無按鈕)
    # =========================================================
    elif st.session_state.page == "load2":
        if st.session_state.get("trigger_load2"):
            st.session_state.trigger_load2 = False
            tok, model = load_story_model()
            st.session_state.story = get_story_fast(st.session_state.caption, tok, model)
            st.session_state.page = "ch3"
            st.rerun()
        else:
            with main_view.container():
                riddle = random.choice(RIDDLES)
                st.markdown(f"""
                <div class="spell-chamber">
                    <div class="cute-loader">
                        <span class="cute-icon">📜</span>
                        <span class="cute-icon delay-1">✨</span>
                        <span class="cute-icon delay-2">🦉</span>
                    </div>
                    <h2 style="color: #ff477e !important;">Weaving Golden Story Threads...</h2>
                    <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                        The royal elves are dipping enchanted quills into starlight ink!
                    </p>
                    <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                        <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                        <p style="font-size: 1.15rem; color: #2b2d42; font-weight: bold; margin: 0;">{riddle}</p>
                    </div>
                    <p style="color: #4361ee; font-weight: bold;">📖 Writing the bedtime story... 📖</p>
                </div>
                """, unsafe_allow_html=True)

                st.button("🚀_2", key="trigger_load2")
                auto_trigger_task("🚀_2")

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
                <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 18px; padding: 1.4rem; min-height: 220px; box-shadow: 0 4px 15px rgba(255, 182, 193, 0.2);">
                    <p style="font-size: 1.2rem; line-height: 1.8; color: #2b2d42; margin: 0;">
                        {st.session_state.story}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🎶 Enter Voice Studio 🎶"):
                    st.session_state.page = "load3"
                    st.rerun()

    # =========================================================
    # Loading 3: 錄製聲音 (自動過渡，無按鈕)
    # =========================================================
    elif st.session_state.page == "load3":
        if st.session_state.get("trigger_load3"):
            st.session_state.trigger_load3 = False
            st.session_state.audio_path = text_to_speech(st.session_state.story)
            st.session_state.page = "ch4"
            st.rerun()
        else:
            with main_view.container():
                riddle = random.choice(RIDDLES)
                st.markdown(f"""
                <div class="spell-chamber">
                    <div class="cute-loader">
                        <span class="cute-icon">🎙️</span>
                        <span class="cute-icon delay-1">🎵</span>
                        <span class="cute-icon delay-2">🧚‍♂️</span>
                    </div>
                    <h2 style="color: #ff477e !important;">Recording Fairy Narration...</h2>
                    <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                        The singing fairies are warming up their vocal cords!
                    </p>
                    <div style="background: rgba(255,245,248,0.9); border-radius: 20px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                        <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                        <p style="font-size: 1.15rem; color: #2b2d42; font-weight: bold; margin: 0;">{riddle}</p>
                    </div>
                    <p style="color: #4361ee; font-weight: bold;">🎶 Synthesizing sweet bedtime voice... 🎶</p>
                </div>
                """, unsafe_allow_html=True)

                st.button("🚀_3", key="trigger_load3")
                auto_trigger_task("🚀_3")

    # =========================================================
    # Chapter 4: 聲音播送與最終結果 (只有重新開始按鈕)
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
            <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 20px; padding: 1.4rem; margin-top: 1.2rem; box-shadow: 0 6px 20px rgba(255, 182, 193, 0.2);">
                <h3 style="color: #ff477e !important; margin-top: 0;">📖 Read Along with the Story:</h3>
                <p style="font-size: 1.18rem; line-height: 1.85; color: #2b2d42; margin-bottom: 0;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                # 乾淨俐落，只保留一個重新創造的按鈕
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
