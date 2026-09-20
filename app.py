import os
import re
import io
import time
import random
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ---------------------------------------------------------
# Page Setup & Theme Styling
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
            setTimeout(doScroll, 250);
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
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(12px);
    border-radius: 26px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35) !important;
    padding: 2rem !important;
    margin: 1.2rem 0 !important;
    animation: bounceIn 0.5s ease-out;
}

.spell-chamber {
    background: radial-gradient(circle, rgba(255,255,255,0.98) 0%, rgba(255,235,245,0.95) 100%);
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 2.5rem 1.8rem;
    text-align: center;
    box-shadow: 0 0 50px rgba(255, 112, 166, 0.65), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    animation: pulseChamber 2.5s infinite alternate;
    margin-top: 0.5rem;
}

.magic-orb {
    display: inline-block;
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: linear-gradient(45deg, #ff70a6, #ffd166, #70d6ff);
    box-shadow: 0 0 30px rgba(255, 112, 166, 0.8);
    animation: spinBreathe 3s infinite ease-in-out;
    margin-bottom: 1rem;
    line-height: 90px;
    font-size: 2.8rem;
}

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

@keyframes spinBreathe {
    0% { transform: rotate(0deg) scale(0.9); box-shadow: 0 0 20px #ff70a6; }
    50% { transform: rotate(180deg) scale(1.15); box-shadow: 0 0 45px #70d6ff; }
    100% { transform: rotate(360deg) scale(0.9); box-shadow: 0 0 20px #ffd166; }
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
# 1. Direct Model Loaders (Zero Pipeline String Task Calls)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_caption_model():
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return proc, model

@st.cache_resource(show_spinner=False)
def load_story_model():
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model

@st.cache_resource(show_spinner=False)
def load_reader_model():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tok, model


# ---------------------------------------------------------
# 2. Inference Functions
# ---------------------------------------------------------
def get_caption(image, proc, model):
    inputs = proc(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=40)
    return proc.decode(out[0], skip_special_tokens=True)


def get_story(caption, tok, model):
    clean_caption = caption.strip().rstrip(".")
    prompt = (
        f"Once upon a time, there was {clean_caption}. "
        f"Every day was full of sweet laughter and adventures! "
        f"Suddenly, a tiny magical rainbow door opened with sparkly stars. "
        f"With wide curious eyes, our brave little friend stepped inside to explore. "
        f"Everyone celebrated and smiled happily ever after!"
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            min_new_tokens=50,
            max_new_tokens=85,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tok.eos_token_id
        )
    raw = tok.decode(out[0], skip_special_tokens=True)
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


def analyze_with_flan(flan_tok, flan_model, caption, story_sentence):
    input_text = (
        f"Context: Image shows {caption}. Story part: '{story_sentence}'. "
        f"Generate a short 4-word fairytale setting:"
    )
    inputs = flan_tok(input_text, return_tensors="pt")
    with torch.no_grad():
        outputs = flan_model.generate(**inputs, max_new_tokens=20)
    visual_idea = flan_tok.decode(outputs[0], skip_special_tokens=True).strip()
    return f"children fairytale book illustration of {visual_idea}, storybook watercolor art, cute, vibrant, warm lighting"


def generate_fairytale_illustration_hf(prompt, fallback_img_path):
    hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    api_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

    if hf_token:
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": prompt, "parameters": {"negative_prompt": "ugly, blurry, distorted, dark"}},
                timeout=20
            )
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            pass

    return Image.open(fallback_img_path).convert("RGB")


# ---------------------------------------------------------
# 3. Fairytale Book Page & Flip Video Assembly
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


def render_fairytale_book_page(illustration, sentence, page_num, total_pages, out_path):
    width, height = 1280, 960
    page = Image.new("RGB", (width, height), color="#FFFDF7")
    draw = ImageDraw.Draw(page)

    draw.rectangle([25, 25, width - 25, height - 25], outline="#FF758F", width=6)
    draw.rectangle([38, 38, width - 38, height - 38], outline="#FFD166", width=2)
    draw.rectangle([48, 48, width - 48, height - 48], outline="#FFB3C6", width=1)

    try:
        ill_copy = illustration.copy()
        ill_copy.thumbnail((620, 420))
        img_x = (width - ill_copy.width) // 2
        img_y = 90
        page.paste(ill_copy, (img_x, img_y))
        draw.rectangle([img_x - 4, img_y - 4, img_x + ill_copy.width + 4, img_y + ill_copy.height + 4], outline="#FF70A6", width=4)
        draw.rectangle([img_x - 8, img_y - 8, img_x + ill_copy.width + 8, img_y + ill_copy.height + 8], outline="#FFD166", width=1)
    except Exception:
        pass

    try:
        font_text = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        font_page = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font_text = ImageFont.load_default()
        font_page = ImageFont.load_default()

    text_y = 570
    lines = wrap_text(sentence, font_text, width - 220, draw)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font_text)
        line_w = bbox[2] - bbox[0]
        draw.text(((width - line_w) // 2, text_y), line, fill="#2B2D42", font=font_text)
        text_y += 48

    page_label = f"📖 Page {page_num} of {total_pages}"
    p_bbox = draw.textbbox((0, 0), page_label, font=font_page)
    draw.text(((width - (p_bbox[2] - p_bbox[0])) // 2, height - 70), page_label, fill="#FF477E", font=font_page)

    page.save(out_path)
    return out_path


def make_fairytale_flip_video(fallback_img_path, caption, story_text, flan_tok, flan_model, output_path="storybook_movie.mp4"):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', story_text) if len(s.strip()) > 3]
    if not sentences:
        sentences = [story_text]
    sentences = sentences[:4]
    total_pages = len(sentences)

    page_clips = []
    temp_files = []

    for idx, sentence in enumerate(sentences, start=1):
        scene_prompt = analyze_with_flan(flan_tok, flan_model, caption, sentence)
        page_illustration = generate_fairytale_illustration_hf(scene_prompt, fallback_img_path)

        page_img_path = f"temp_flip_page_{idx}.png"
        render_fairytale_book_page(page_illustration, sentence, idx, total_pages, page_img_path)
        temp_files.append(page_img_path)

        page_audio_path = f"temp_flip_audio_{idx}.mp3"
        tts = gTTS(text=sentence, lang="en", slow=False)
        tts.save(page_audio_path)
        temp_files.append(page_audio_path)

        audio_clip = AudioFileClip(page_audio_path)
        page_duration = max(3.5, audio_clip.duration + 0.5)
        
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
# 4. Interactive Riddles
# ---------------------------------------------------------
RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up and never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]


# ---------------------------------------------------------
# 5. State Management
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = "ch1"
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
# 6. Main Application Workflow (One Page per Step & Loading)
# ---------------------------------------------------------
def main():
    scroll_to_top()
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)

    # =====================================================
    # CHAPTER 1: Image Upload Page
    # =====================================================
    if st.session_state.step == "ch1":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 1: The Magic Portal</p>", unsafe_allow_html=True)
        st.progress(0.1)

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
                    st.session_state.step = "load1"
                    st.rerun()

    # =====================================================
    # LOADING PAGE 1 (Mirror Vision)
    # =====================================================
    elif st.session_state.step == "load1":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">✨</div>
            <h2 style="color: #ff477e !important;">Awakening the Mirror Vision...</h2>
            <p style="font-size: 1.2rem; color: #6a0572; font-weight: bold;">
                The fairies are casting an enchantment over your picture!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.2rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.2rem; color: #2b2d42; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.05rem;"><i>💨 Breathe with the glowing orb: inhale, exhale, and blow soft magic dust!</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold;">🔮 Analyzing visual clues with BLIP vision transformer... 🔮</p>
        </div>
        """, unsafe_allow_html=True)

        blip_proc, blip_model = load_caption_model()
        st.session_state.caption = get_caption(st.session_state.uploaded_img, blip_proc, blip_model)
        
        time.sleep(2.0)
        st.session_state.step = "ch2"
        st.rerun()

    # =====================================================
    # CHAPTER 2: The Crystal Ball Clue Page
    # =====================================================
    elif st.session_state.step == "ch2":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 2: The Crystal Ball</p>", unsafe_allow_html=True)
        st.progress(0.28)
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
                st.session_state.step = "load2"
                st.rerun()

    # =====================================================
    # LOADING PAGE 2 (Story Weaving)
    # =====================================================
    elif st.session_state.step == "load2":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">📜</div>
            <h2 style="color: #ff477e !important;">Weaving Golden Story Threads...</h2>
            <p style="font-size: 1.2rem; color: #6a0572; font-weight: bold;">
                The royal elves are dipping enchanted quills into starlight ink!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.2rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.2rem; color: #2b2d42; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.05rem;"><i>✨ Chant along: "Abracadabra, alakazam, weave a story as fast as you can!" ✨</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold;">📖 Generating fairytale narrative with text-transformer... 📖</p>
        </div>
        """, unsafe_allow_html=True)

        gpt_tok, gpt_model = load_story_model()
        st.session_state.story = get_story(st.session_state.caption, gpt_tok, gpt_model)
        
        time.sleep(2.0)
        st.session_state.step = "ch3"
        st.rerun()

    # =====================================================
    # CHAPTER 3: Story Scroll Page
    # =====================================================
    elif st.session_state.step == "ch3":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 3: The Golden Scroll</p>", unsafe_allow_html=True)
        st.progress(0.48)
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
                st.session_state.step = "load3"
                st.rerun()
        with btn_c2:
            if st.button("🔄 Try Another Picture"):
                st.session_state.step = "ch1"
                st.session_state.uploaded_img = None
                st.rerun()

    # =====================================================
    # LOADING PAGE 3 (Voice Tuning Chamber)
    # =====================================================
    elif st.session_state.step == "load3":
        st.markdown("""
        <div class="spell-chamber">
            <div class="magic-orb">🎶</div>
            <h2 style="color: #ff477e !important;">Tuning the Fairyland Harp...</h2>
            <p style="font-size: 1.2rem; color: #6a0572; font-weight: bold;">
                The singing fairies are warming up their vocal cords to narrate your tale!
            </p>
            <p style="color: #4361ee; font-weight: bold;">✨ Preparing magical sound studio... ✨</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.5)
        st.session_state.step = "ch4"
        st.rerun()

    # =====================================================
    # CHAPTER 4: Voice Harp Page
    # =====================================================
    elif st.session_state.step == "ch4":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
        st.progress(0.68)

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
                    st.session_state.step = "load4"
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
        if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
            btn_c1, btn_c2 = st.columns([1, 1])
            with btn_c1:
                if st.button("🎬 Proceed to Living Story Cinema (Chapter 5)"):
                    st.session_state.step = "ch5"
                    st.rerun()
            with btn_c2:
                if st.button("📜 Back to Story Scroll"):
                    st.session_state.step = "ch3"
                    st.rerun()

    # =====================================================
    # LOADING PAGE 4 (Voice Synthesis)
    # =====================================================
    elif st.session_state.step == "load4":
        st.markdown("""
        <div class="spell-chamber">
            <div class="magic-orb">🎙️</div>
            <h2 style="color: #ff477e !important;">Recording the Fairy Narration...</h2>
            <p style="font-size: 1.2rem; color: #6a0572; font-weight: bold;">
                Sprinkling vocal dust and recording the story audio...
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.audio_path = text_to_speech(st.session_state.story)
        time.sleep(1.5)
        st.session_state.step = "ch4"
        st.rerun()

    # =====================================================
    # CHAPTER 5: Multimodal Concept & Binding Prompt Page
    # =====================================================
    elif st.session_state.step == "ch5":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 5: The Storybook Studio</p>", unsafe_allow_html=True)
        st.progress(0.85)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎬 Chapter 5: The Storybook Studio</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Our Hugging Face Transformers will now read the accumulated content from Chapters 1–4, generate fairytale scene art for each sentence, and bind everything into an animated flip-book movie!
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="Original Input Scene", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1.2rem; border-radius: 18px; border: 2px dashed #ffb3c6;">
                <p><strong>Clue:</strong> {st.session_state.caption.capitalize()}</p>
                <p><strong>Narration:</strong> Ready in audio format</p>
                <p><strong>Ready to synthesize:</strong> Multi-page illustrated flip storybook (&lt;30s)</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("✨ Bind & Animate Fairytale Storybook ✨"):
                st.session_state.step = "load5"
                st.rerun()

    # =====================================================
    # LOADING PAGE 5 (Transformer Reading & Video Rendering)
    # =====================================================
    elif st.session_state.step == "load5":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-orb">📚</div>
            <h2 style="color: #ff477e !important;">Binding Your Animated Storybook...</h2>
            <p style="font-size: 1.2rem; color: #6a0572; font-weight: bold;">
                Transformers are reading Chapters 1–4, synthesizing illustrated pages, and stitching the audio!
            </p>
            <div style="background: rgba(255,255,255,0.85); border-radius: 20px; padding: 1.2rem; margin: 1.2rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Final Fairy Riddle!</h3>
                <p style="font-size: 1.2rem; color: #2b2d42; font-weight: bold;">{q}</p>
            </div>
            <p style="color: #4361ee; font-weight: bold;">🎬 Rendering 30-second flip-book video reel... 🎬</p>
        </div>
        """, unsafe_allow_html=True)

        flan_tok, flan_model = load_reader_model()
        video_file = make_fairytale_flip_video(
            "temp_input_scene.png",
            st.session_state.caption,
            st.session_state.story,
            flan_tok,
            flan_model
        )
        st.session_state.video_path = video_file
        
        st.session_state.step = "final"
        st.rerun()

    # =====================================================
    # FINAL RESULT PAGE: Animated Storybook Video (<30s)
    # =====================================================
    elif st.session_state.step == "final":
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Final Result: Your Fairytale Storybook</p>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎬 Your Living Fairytale Flip Storybook</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Here is your animated picture book! Each page has been illustrated and narrated, auto-flipping sentence-by-sentence under 30 seconds!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
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
                st.session_state.step = "ch1"
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.session_state.video_path = ""
                st.rerun()


if __name__ == "__main__":
    main()
