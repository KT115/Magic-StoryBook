import os
import re
import time
import random
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, concatenate_audioclips, AudioClip

# ---------------------------------------------------------
# Page Configuration & Pastel Fairytale Theme
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

# 穩健的強制置頂函式 (解決 Issue 1)
def force_scroll_to_top():
    components.html(
        """
        <script>
            function scrollToTop() {
                const mainSection = window.parent.document.querySelector('section.main');
                if (mainSection) {
                    mainSection.scrollTo({ top: 0, behavior: 'smooth' });
                }
                window.parent.scrollTo({ top: 0, behavior: 'smooth' });
            }
            scrollToTop();
            setTimeout(scrollToTop, 150);
            setTimeout(scrollToTop, 400);
        </script>
        """,
        height=0
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

/* 全域夢幻粉彩背景 */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
    color: #4a4e69 !important;
}

/* 標題設定 */
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

/* 羊皮紙卡片 */
.magic-parchment {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2rem !important;
    margin: 1.2rem 0 !important;
    animation: bounceIn 0.5s ease-out;
}

/* 魔法 Loading 等待室 */
.spell-chamber {
    background: radial-gradient(circle, rgba(255,255,255,0.98) 0%, rgba(255,230,240,0.95) 100%);
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 45px rgba(255, 112, 166, 0.6), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    animation: pulseChamber 2.5s infinite alternate;
    margin-top: 1rem;
}

.magic-crystal {
    font-size: 4.8rem;
    animation: floatOrb 2s infinite ease-in-out alternate;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* 按鈕置中與樣式 */
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

@keyframes floatOrb {
    0% { transform: translateY(0px) rotate(0deg) scale(1.0); }
    100% { transform: translateY(-16px) rotate(8deg) scale(1.1); }
}

@keyframes pulseChamber {
    0% { box-shadow: 0 0 25px rgba(255, 112, 166, 0.4); }
    100% { box-shadow: 0 0 55px rgba(112, 214, 255, 0.85); }
}

@keyframes bounceIn {
    0% { transform: scale(0.95); opacity: 0; }
    100% { transform: scale(1.0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. 模型初始化 (Direct Load 避免 KeyError)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
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
        f"Every day was filled with joy and games! "
        f"Suddenly, a tiny magical golden door appeared. "
        f"With curious eyes, our little friend opened it and stepped into wonderland. "
        f"Everyone celebrated happily together!"
    )
    output = story_pipe(
        prompt,
        min_new_tokens=50,
        max_new_tokens=85,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.3
    )
    raw = output[0]["generated_text"]
    last_period = max(raw.rfind("."), raw.rfind("!"), raw.rfind("?"))
    if last_period != -1:
        story = raw[:last_period + 1]
    else:
        story = raw + " And everyone lived happily ever after!"
    return story


def text_to_speech(text, filename="story_full.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. 繪本單頁繪製與翻頁童話影片生成 (解決 Issue 2)
# ---------------------------------------------------------
def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        bbox = draw.textbbox((0, 0), " ".join(current_line), font=font)
        if (bbox[2] - bbox[0]) > max_width:
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines


def render_fairytale_book_page(image_path, sentence, page_num, total_pages, out_path):
    """
    100% 依據 attachment 繪製經典兒童繪本頁面：
    雙粉金細邊框、白淨象牙底色、置中插圖、排版舒適的童書字體與頁碼
    """
    width, height = 1280, 960
    page = Image.new("RGB", (width, height), color="#FFFDF7")
    draw = ImageDraw.Draw(page)

    # 繪製精美繪本書頁外框
    draw.rectangle([25, 25, width - 25, height - 25], outline="#FF758F", width=6)
    draw.rectangle([38, 38, width - 38, height - 38], outline="#FFD166", width=2)
    draw.rectangle([48, 48, width - 48, height - 48], outline="#FFB3C6", width=1)

    # 放置置中的插圖
    try:
        scene = Image.open(image_path).convert("RGB")
        scene.thumbnail((620, 420))
        img_x = (width - scene.width) // 2
        img_y = 90
        page.paste(scene, (img_x, img_y))
        # 插圖精緻細框
        draw.rectangle([img_x - 4, img_y - 4, img_x + scene.width + 4, img_y + scene.height + 4], outline="#FF70A6", width=4)
        draw.rectangle([img_x - 8, img_y - 8, img_x + scene.width + 8, img_y + scene.height + 8], outline="#FFD166", width=1)
    except Exception:
        pass

    # 字體設定
    try:
        font_text = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        font_page = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font_text = ImageFont.load_default()
        font_page = ImageFont.load_default()

    # 繪製本頁單一句子
    text_y = 570
    lines = wrap_text(sentence, font_text, width - 200, draw)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_text)
        line_w = bbox[2] - bbox[0]
        draw.text(((width - line_w) // 2, text_y), line, fill="#2B2D42", font=font_text)
        text_y += 48

    # 底部頁碼標記：📖 Page X of Y
    page_label = f"📖 Page {page_num} of {total_pages}"
    p_bbox = draw.textbbox((0, 0), page_label, font=font_page)
    draw.text(((width - (p_bbox[2] - p_bbox[0])) // 2, height - 70), page_label, fill="#FF477E", font=font_page)

    page.save(out_path)
    return out_path


def make_fairytale_flip_video(image_path, story_text, output_path="storybook_movie.mp4"):
    """
    將故事切分成獨立句子，每頁單獨錄音，並合成自動翻頁的童話影片 (嚴格在 30 秒內)
    """
    # 依標點切分句子
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', story_text) if len(s.strip()) > 3]
    if not sentences:
        sentences = [story_text]
    # 最多 4 頁以確保在 30 秒內讀完
    sentences = sentences[:4]
    total_pages = len(sentences)

    page_clips = []
    temp_files = []

    for idx, sentence in enumerate(sentences, start=1):
        # 1. 繪製單頁繪本
        page_img_path = f"temp_flip_page_{idx}.png"
        render_fairytale_book_page(image_path, sentence, idx, total_pages, page_img_path)
        temp_files.append(page_img_path)

        # 2. 為該頁獨立生成清晰的句子語音
        page_audio_path = f"temp_flip_audio_{idx}.mp3"
        tts = gTTS(text=sentence, lang="en", slow=False)
        tts.save(page_audio_path)
        temp_files.append(page_audio_path)

        audio_clip = AudioFileClip(page_audio_path)
        
        # 加上 0.4 秒翻頁呼吸停頓
        page_duration = max(3.5, audio_clip.duration + 0.5)
        
        # 生成該頁的 video clip (帶有平滑淡入轉場效果)
        clip = ImageClip(page_img_path).set_duration(page_duration).set_audio(audio_clip)
        if idx > 1:
            clip = clip.crossfadein(0.35)
        page_clips.append(clip)

    final_video = concatenate_videoclips(page_clips, method="compose")

    final_video.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )

    # 釋放資源與清理暫存
    for c in page_clips:
        c.close()
    final_video.close()
    for f in temp_files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    return output_path


# ---------------------------------------------------------
# 4. 謎題資料
# ---------------------------------------------------------
RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up and never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]


# ---------------------------------------------------------
# 5. Session State
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
if "video_path" not in st.session_state:
    st.session_state.video_path = ""


# ---------------------------------------------------------
# 6. 主程式流程
# ---------------------------------------------------------
def main():
    # 頂部定位錨點與全域置頂跳轉
    st.markdown('<div id="magic-top"></div>', unsafe_allow_html=True)
    force_scroll_to_top()

    st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # CHAPTER 1: 上傳圖片
    # -----------------------------------------------------
    if st.session_state.step == 1:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 1: The Magic Portal</p>", unsafe_allow_html=True)
        st.progress(0.2)

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
            st.session_state.uploaded_img.save("temp_input_scene.png")
            st.image(st.session_state.uploaded_img, caption="Your Enchanted Picture", use_container_width=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    st.session_state.step = "loading_mirror"
                    st.rerun()

    # -----------------------------------------------------
    # LOADING SCREEN 1: 鏡像魔法陣 (自動跳頂)
    # -----------------------------------------------------
    elif st.session_state.step == "loading_mirror":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-crystal">🔮✨🦄</div>
            <h2 style="color: #ff477e !important;">Awakening the Mirror Vision...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The fairies are casting an enchantment over your picture!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 18px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.2rem; color: #3d405b; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>💨 Blow soft magical breaths toward the screen while the portal opens!</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.1rem;">✨ Gathering starlight & deciphering clues... ✨</p>
        </div>
        """, unsafe_allow_html=True)

        processor, caption_model, _ = load_models()
        st.session_state.caption = get_caption(st.session_state.uploaded_img, processor, caption_model)
        
        time.sleep(1.8)
        st.session_state.step = 2
        st.rerun()

    # -----------------------------------------------------
    # CHAPTER 2: 水晶球解讀
    # -----------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 2: The Crystal Ball</p>", unsafe_allow_html=True)
        st.progress(0.4)
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
            <div style="background: rgba(255, 255, 255, 0.92); padding: 1.4rem; border-radius: 18px; border: 2px solid #70d6ff; text-align: center; margin-top: 1rem;">
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
                st.session_state.step = "loading_story"
                st.rerun()

    # -----------------------------------------------------
    # LOADING SCREEN 2: 故事編織等待室 (自動跳頂)
    # -----------------------------------------------------
    elif st.session_state.step == "loading_story":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-crystal">📜✨🧚‍♀️</div>
            <h2 style="color: #ff477e !important;">Weaving Golden Story Threads...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The royal elves are dipping enchanted quills into starlight ink!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 18px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.2rem; color: #3d405b; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>✨ Chant along: "Abracadabra, alakazam, weave a story as fast as you can!" ✨</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.1rem;">📖 Crafting a magical fairy tale... 📖</p>
        </div>
        """, unsafe_allow_html=True)

        _, _, story_pipe = load_models()
        st.session_state.story = get_story(st.session_state.caption, story_pipe)
        
        time.sleep(2.0)
        st.session_state.step = 3
        st.rerun()

    # -----------------------------------------------------
    # CHAPTER 3: 黃金故事卷軸
    # -----------------------------------------------------
    elif st.session_state.step == 3:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 3: The Golden Scroll</p>", unsafe_allow_html=True)
        st.progress(0.6)
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
                st.session_state.step = 4
                st.rerun()
        with btn_c2:
            if st.button("🔄 Try Another Picture"):
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 4: 聲音之豎琴 (含插圖、語音與故事內文)
    # -----------------------------------------------------
    elif st.session_state.step == 4:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
        st.progress(0.8)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Listen to the fairy narrator tell the story before we bind it into the movie book!
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
                    with st.spinner("Recording fairy voice..."):
                        st.session_state.audio_path = text_to_speech(st.session_state.story)
                    st.rerun()
            else:
                st.success("✨ Fairy Audio Narrated Successfully!")
                st.audio(st.session_state.audio_path, format="audio/mp3")

        # 清楚顯示完整故事內容
        st.markdown(f"""
        <div style="background: #ffffff; border: 2px dashed #ffb3c6; border-radius: 20px; padding: 1.4rem; margin-top: 1.2rem; box-shadow: 0 6px 20px rgba(255, 182, 193, 0.2);">
            <h3 style="color: #ff477e !important; margin-top: 0;">📖 Read Along with the Story:</h3>
            <p style="font-size: 1.18rem; line-height: 1.85; color: #2b2d42; margin-bottom: 0;">
                {st.session_state.story}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                if st.button("🎬 Generate Flip Storybook Movie (Chapter 5)"):
                    st.session_state.step = 5
                    st.rerun()
            with btn_c2:
                if st.button("📜 Back to Story Scroll"):
                    st.session_state.step = 3
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 5: 真正的童話翻頁繪本影片 (<30秒)
    # -----------------------------------------------------
    elif st.session_state.step == 5:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 5: The Fairytale Flip Storybook</p>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎬 Chapter 5: The Fairytale Flip Storybook</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Here is your animated picture book! Each page displays one sentence, automatically turning pages and reading the tale under 30 seconds!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # 生成翻頁故事書影片
        if not st.session_state.video_path or not os.path.exists(st.session_state.video_path):
            with st.spinner("Binding storybook pages and animating video (<30s)..."):
                video_file = make_fairytale_flip_video("temp_input_scene.png", st.session_state.story)
                st.session_state.video_path = video_file
                st.rerun()

        # 播放最終翻頁影片
        st.video(st.session_state.video_path)

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("🏰 Create a Brand New Fairytale"):
                for f in ["temp_input_scene.png", st.session_state.audio_path, st.session_state.video_path]:
                    if os.path.exists(f):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.session_state.video_path = ""
                st.rerun()


if __name__ == "__main__":
    main()
