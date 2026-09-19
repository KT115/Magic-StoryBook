import os
import time
import random
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip

# ---------------------------------------------------------
# Page Configuration & Pastel Fairytale Theme
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🦄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;700&family=Cinzel+Decorative:wght@700&display=swap');

/* Dreamy Pastel Rainbow Gradient Backdrop */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
.stApp {
    background: radial-gradient(circle at 20% 20%, #ffe5ec 0%, #ffcbf2 30%, #e8dff5 55%, #caffbf 85%, #9bf6ff 100%) !important;
    background-attachment: fixed !important;
    font-family: 'Quicksand', sans-serif !important;
    color: #4a4e69 !important;
}

/* Titles and Headers */
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

/* Storybook Parchment Card */
.magic-parchment {
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(12px);
    border-radius: 28px !important;
    border: 3px solid #ffb3c6 !important;
    box-shadow: 0 12px 35px rgba(255, 154, 162, 0.35), 0 0 20px rgba(255, 255, 255, 0.8) inset !important;
    padding: 2rem !important;
    margin: 1.2rem 0 !important;
    animation: bounceIn 0.5s ease-out;
}

/* Story Text Reader Card */
.story-reader-box {
    background: #ffffff;
    border: 2px dashed #ffb3c6;
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 6px 20px rgba(255, 182, 193, 0.25);
    margin-top: 1.2rem;
}

/* Dedicated Spellcasting Chamber / Wait Page */
.spell-chamber {
    background: radial-gradient(circle, rgba(255,255,255,0.96) 0%, rgba(255,229,236,0.92) 100%);
    border: 4px solid #ff70a6;
    border-radius: 30px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 112, 166, 0.6), 0 0 25px rgba(112, 214, 255, 0.5) inset;
    animation: pulseChamber 2.5s infinite alternate;
}

.magic-crystal {
    font-size: 4.8rem;
    animation: floatOrb 2s infinite ease-in-out alternate;
    display: inline-block;
    margin-bottom: 0.5rem;
}

/* Centered Magic Buttons */
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

/* Subtitle Banner in Chapter 5 */
.subtitles-banner {
    background: rgba(30, 11, 46, 0.9);
    border: 2px solid #ffd166;
    border-radius: 16px;
    color: #fff9a6;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 1rem 1.4rem;
    text-align: center;
    line-height: 1.6;
    margin-top: 0.8rem;
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
# 1. Model Initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Direct model initialization avoiding task registry KeyError."""
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
        f"Every day was filled with fun, but today a secret golden rainbow door opened! "
        f"With wide curious eyes, our brave little friend stepped forward"
    )
    
    output = story_pipe(
        prompt,
        min_new_tokens=65,
        max_new_tokens=95,
        do_sample=True,
        temperature=0.85,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.3
    )
    
    raw_story = output[0]["generated_text"]
    last_period = max(raw_story.rfind("."), raw_story.rfind("!"), raw_story.rfind("?"))
    if last_period != -1:
        story = raw_story[:last_period + 1]
    else:
        story = raw_story + " And everyone smiled happily ever after!"
    return story


def text_to_speech(text, filename="magic_audio.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


def make_storybook_video(image_path, audio_path, output_path="storybook_video.mp4"):
    audio_clip = AudioFileClip(audio_path)
    video_clip = ImageClip(image_path).set_duration(audio_clip.duration)
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(
        output_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )
    audio_clip.close()
    video_clip.close()
    return output_path


# ---------------------------------------------------------
# 3. Interactive Fairytale Riddles
# ---------------------------------------------------------
RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up and never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]


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
if "video_path" not in st.session_state:
    st.session_state.video_path = ""


# ---------------------------------------------------------
# 5. Main Application Flow
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🦄 The Whispering Storybook 🦄</h1>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # CHAPTER 1: Image Upload
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
            st.session_state.uploaded_img.save("temp_scene.png")
            st.image(st.session_state.uploaded_img, caption="Your Enchanted Picture", use_container_width=True)

            _, btn_c, _ = st.columns([1, 2, 1])
            with btn_c:
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    st.session_state.step = "loading_mirror"
                    st.rerun()

    # -----------------------------------------------------
    # LOADING SCREEN 1: Mirror Spellcasting Chamber
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
            <div style="background: rgba(255,255,255,0.75); border-radius: 18px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
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
    # CHAPTER 2: The Crystal Ball Speaks
    # -----------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 2: The Crystal Ball</p>", unsafe_allow_html=True)
        st.progress(0.4)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🔮 Chapter 2: The Crystal Ball Speaks!</h2>
            <p style="font-size: 1.15rem; text-align: center;">Here is the secret clue the crystal ball spotted inside your picture:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); padding: 1.4rem; border-radius: 18px; border: 2px solid #70d6ff; text-align: center; margin-top: 1rem;">
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
    # LOADING SCREEN 2: Story Weaving Chamber
    # -----------------------------------------------------
    elif st.session_state.step == "loading_story":
        q, a = random.choice(RIDDLES)
        st.markdown(f"""
        <div class="spell-chamber">
            <div class="magic-crystal">📜✨🧚‍♀️</div>
            <h2 style="color: #ff477e !important;">Weaving Golden Story Threads...</h2>
            <p style="font-size: 1.25rem; color: #6a0572; font-weight: bold;">
                The royal elves are dipping enchanted quills into rainbow starlight ink!
            </p>
            <div style="background: rgba(255,255,255,0.75); border-radius: 18px; padding: 1.2rem; margin: 1.5rem 0; border: 2px dashed #ffb3c6;">
                <h3 style="color: #7209b7 !important; margin: 0 0 0.5rem 0;">🌟 Fairy Riddle Time!</h3>
                <p style="font-size: 1.2rem; color: #3d405b; font-weight: bold;">{q}</p>
                <p style="color: #ff499e; font-size: 1.1rem;"><i>✨ Chant along: "Abracadabra, alakazam, weave a story as fast as you can!" ✨</i></p>
            </div>
            <p style="color: #4361ee; font-weight: bold; font-size: 1.1rem;">📖 Crafting a magical 50-100 word adventure... 📖</p>
        </div>
        """, unsafe_allow_html=True)

        _, _, story_pipe = load_models()
        st.session_state.story = get_story(st.session_state.caption, story_pipe)
        
        time.sleep(2.0)
        st.session_state.step = 3
        st.rerun()

    # -----------------------------------------------------
    # CHAPTER 3: The Golden Story Scroll
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
    # CHAPTER 4: The Voice Harp (With Image, Audio & Story Text)
    # -----------------------------------------------------
    elif st.session_state.step == 4:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 4: The Voice Harp</p>", unsafe_allow_html=True)
        st.progress(0.8)

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎶 Chapter 4: The Voice Harp</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Transform your written adventure into spoken audio narration!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Top row: Image on left, Audio Generation / Player on right
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="The Storybook Scene", use_container_width=True)
        
        with col2:
            if not st.session_state.audio_path or not os.path.exists(st.session_state.audio_path):
                st.markdown("<p style='text-align:center;'>Click below to summon the royal narrator!</p>", unsafe_allow_html=True)
                if st.button("🧚 Cast Voice Spell"):
                    with st.spinner("Recording fairy voice..."):
                        st.session_state.audio_path = text_to_speech(st.session_state.story)
                    st.rerun()
            else:
                st.success("✨ Fairy Audio Narrated Successfully!")
                st.audio(st.session_state.audio_path, format="audio/mp3")

        # Full story displayed clearly inside Chapter 4
        st.markdown(f"""
        <div class="story-reader-box">
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
                if st.button("🎬 Generate Storybook Video (Chapter 5)"):
                    st.session_state.step = 5
                    st.rerun()
            with btn_c2:
                if st.button("📜 Back to Story Scroll"):
                    st.session_state.step = 3
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 5: The Storybook Video (.mp4 Generation)
    # -----------------------------------------------------
    elif st.session_state.step == 5:
        st.markdown("<p style='text-align: center; color: #6a0572; font-weight: 700;'>Chapter 5: The Storybook Video</p>", unsafe_allow_html=True)
        st.progress(1.0)
        st.balloons()

        st.markdown("""
        <div class="magic-parchment">
            <h2>🎬 Chapter 5: The Living Storybook Video</h2>
            <p style="font-size: 1.15rem; text-align: center;">
                Your video brings together the picture, synchronized voiceover, and full story subtitles!
            </p>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.video_path or not os.path.exists(st.session_state.video_path):
            with st.spinner("Rendering your storybook video with synchronized audio..."):
                video_file = make_storybook_video("temp_scene.png", st.session_state.audio_path)
                st.session_state.video_path = video_file
                st.rerun()

        st.video(st.session_state.video_path)

        st.markdown(f"""
        <div class="subtitles-banner">
            💛 <strong>Storybook Subtitles:</strong><br>"{st.session_state.story}"
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        _, btn_c, _ = st.columns([1, 2, 1])
        with btn_c:
            if st.button("🏰 Create a Brand New Fairytale"):
                for f in ["temp_scene.png", st.session_state.audio_path, st.session_state.video_path]:
                    if os.path.exists(f):
                        os.remove(f)
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.session_state.video_path = ""
                st.rerun()


if __name__ == "__main__":
    main()
