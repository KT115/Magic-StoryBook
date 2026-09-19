import os
import time
import random
import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS

# ---------------------------------------------------------
# Page Configuration & Fairy Tale Theme Styling
# ---------------------------------------------------------
st.set_page_config(page_title="The Whispering Storybook", page_icon="🏰", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:ital,wght@0,400;0,700;1,400&family=Cinzel:wght@600;700&display=swap');

/* Night sky backdrop */
.stApp {
    background: radial-gradient(circle at 50% 20%, #2b1055 0%, #150930 60%, #090314 100%);
    color: #f7ede2;
    font-family: 'Comic Neue', cursive;
}

/* Titles */
h1, h2, h3 {
    font-family: 'Cinzel', serif !important;
    color: #ffd166 !important;
    text-shadow: 0 0 12px rgba(255, 209, 102, 0.4);
    text-align: center;
}

/* Parchment Card */
.magic-parchment {
    background: #fffdf7;
    color: #2b2d42;
    padding: 2rem;
    border-radius: 20px;
    border: 3px solid #e0a96d;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(255, 209, 102, 0.25);
    margin: 1.2rem 0;
    animation: fadeIn 0.6s ease-in-out;
}

/* Interactive Wait Box */
.riddle-box {
    background: rgba(255, 255, 255, 0.12);
    border: 2px dashed #f4a261;
    border-radius: 16px;
    padding: 1.2rem;
    margin: 1rem 0;
    text-align: center;
    color: #fceade;
    font-size: 1.15rem;
}

/* Force all Streamlit button blocks to center horizontally */
div[data-testid="stColumn"] {
    display: flex;
    justify-content: center;
    align-items: center;
}

div.stButton {
    display: flex;
    justify-content: center;
    margin: 0.5rem auto;
}

div.stButton > button {
    background: linear-gradient(135deg, #f72585 0%, #7209b7 50%, #4361ee 100%) !important;
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    border: 2px solid #ffd166 !important;
    box-shadow: 0 0 15px rgba(255, 209, 102, 0.4) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div.stButton > button:hover {
    transform: scale(1.04) !important;
    box-shadow: 0 0 25px rgba(255, 209, 102, 0.8) !important;
}

/* Animated Storybook Video / Canvas Card */
.cinema-container {
    background: #110924;
    border: 3px solid #ffd166;
    border-radius: 20px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255, 209, 102, 0.4);
    animation: breatheGlow 4s infinite alternate;
}

@keyframes breatheGlow {
    from { box-shadow: 0 0 15px rgba(255, 209, 102, 0.3); }
    to { box-shadow: 0 0 30px rgba(247, 37, 133, 0.6); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 1. Models Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    story_model = pipeline("text-generation", model="gpt2")
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
    prompt = f"Once upon a time, there was {caption}. One sunny morning, a brave little friend found"
    output = story_pipe(
        prompt,
        max_new_tokens=85,
        min_length=50,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2
    )
    raw_story = output[0]["generated_text"]
    return raw_story[:raw_story.rfind(".") + 1] if "." in raw_story else raw_story


def text_to_speech(text, filename="story_sound.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. Interactive Kid Activities
# ---------------------------------------------------------
MAGIC_RIDDLES = [
    ("🧙‍♂️ 'What has hands but cannot clap?'", "A clock! ⏰"),
    ("🌟 'What gets wetter the more it dries?'", "A towel! 🛁"),
    ("🐉 'What has a head and a tail, but no body?'", "A coin! 🪙"),
    ("🦄 'What goes up but never comes down?'", "Your age! 🎂"),
    ("🧚 'What is full of holes but still holds water?'", "A sponge! 🧽")
]

def show_interactive_loader(magic_action):
    riddle_q, riddle_a = random.choice(MAGIC_RIDDLES)
    box = st.empty()
    
    with box.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>✨ Casting: {magic_action}... ✨</h3>
            <p><strong>Riddle Time:</strong> {riddle_q}</p>
            <p><i>Blow softly on your screen to help make the magic work! 💨</i></p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1.8)
    
    with box.container():
        st.markdown(f"""
        <div class="riddle-box">
            <h3>🪄 The Magic Worked! 🪄</h3>
            <p><strong>Answer:</strong> {riddle_a}</p>
            <p>✨ <i>Abracadabra!</i> ✨</p>
        </div>
        """, unsafe_allow_html=True)
    
    time.sleep(1.0)
    box.empty()


# ---------------------------------------------------------
# 4. State Initialisation
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


# ---------------------------------------------------------
# 5. Main Application Flow
# ---------------------------------------------------------
def main():
    st.markdown("<h1>🏰 The Whispering Storybook 🏰</h1>", unsafe_allow_html=True)
    
    chapters = ["1. Enchanted Mirror", "2. Crystal Ball", "3. Magic Scroll", "4. Story Cinema"]
    st.markdown(f"<p style='text-align: center; color: #ffd166; font-size: 1.1rem;'>Chapter: <b>{chapters[st.session_state.step - 1]}</b></p>", unsafe_allow_html=True)
    st.progress(st.session_state.step / 4)

    # -----------------------------------------------------
    # CHAPTER 1: Image Upload
    # -----------------------------------------------------
    if st.session_state.step == 1:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Step 1: Feed the Magic Mirror</h2>
            <p style="font-size: 1.2rem; text-align: center;">
                Upload an illustration, toy photo, or drawing! The mirror will gaze inside.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload a photo or drawing (PNG/JPG):", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_img = image
            st.image(image, caption="Your Enchanted Picture", use_container_width=True)

            center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
            with center_col2:
                if st.button("🪄 Awaken the Magic Mirror 🪄"):
                    st.balloons()
                    show_interactive_loader("Mirror Vision")
                    
                    processor, caption_model, _ = load_models()
                    with st.spinner("Deciphering what's in the picture..."):
                        st.session_state.caption = get_caption(image, processor, caption_model)
                    
                    st.session_state.step = 2
                    st.rerun()

    # -----------------------------------------------------
    # CHAPTER 2: Caption Inspection
    # -----------------------------------------------------
    elif st.session_state.step == 2:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🔮 Step 2: The Crystal Ball Speaks!</h2>
            <p style="font-size: 1.15rem; text-align: center;">Here is what the magical mirror discovered:</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="Your Clue", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="background: #eef1f6; padding: 1.4rem; border-radius: 14px; border: 2px solid #a2d2ff; text-align: center; margin-top: 1rem;">
                <h3 style="color: #1d3557 !important; margin: 0;">✨ The Mirror Saw:</h3>
                <p style="font-size: 1.25rem; font-weight: bold; color: #457b9d; margin-top: 0.5rem;">
                    "{st.session_state.caption.capitalize()}"
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("📜 Spin a Bedtime Tale from This Clue! 📜"):
                st.snow()
                show_interactive_loader("Story Weaving")
                
                _, _, story_pipe = load_models()
                with st.spinner("Weaving words into a story..."):
                    st.session_state.story = get_story(st.session_state.caption, story_pipe)
                
                st.session_state.step = 3
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 3: Story Scroll + Image Together
    # -----------------------------------------------------
    elif st.session_state.step == 3:
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">📜 Step 3: The Golden Story Scroll</h2>
        </div>
        """, unsafe_allow_html=True)

        # Image and narrative together in the layout
        img_col, story_col = st.columns([1, 1])
        with img_col:
            if st.session_state.uploaded_img:
                st.image(st.session_state.uploaded_img, caption="The Illustrated Scene", use_container_width=True)
        
        with story_col:
            st.markdown(f"""
            <div style="background: #fffdf7; border: 2px dashed #e0a96d; border-radius: 15px; padding: 1.2rem; height: 100%;">
                <p style="font-size: 1.15rem; line-height: 1.75; color: #2b2d42;">
                    {st.session_state.story}
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        # Centered action buttons
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("🎬 Turn Into Story Cinema (Audio & Show)"):
                show_interactive_loader("Cinema Animation")
                with st.spinner("Summoning fairyland cinema..."):
                    st.session_state.audio_path = text_to_speech(st.session_state.story)
                st.session_state.step = 4
                st.rerun()
        with btn_col2:
            if st.button("🔄 Start New Journey"):
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.rerun()

    # -----------------------------------------------------
    # CHAPTER 4: Story Cinema (Animated Video / Audio Mode)
    # -----------------------------------------------------
    elif st.session_state.step == 4:
        st.balloons()
        st.markdown("""
        <div class="magic-parchment">
            <h2 style="color: #6b2d5c !important;">🎬 Step 4: Fairyland Story Cinema</h2>
            <p style="font-size: 1.15rem; text-align: center; color: #3d405b;">
                Watch the magical scene come alive as the story is read to you!
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Cinema Box with animated glow and synchronized audio
        st.markdown('<div class="cinema-container">', unsafe_allow_html=True)
        if st.session_state.uploaded_img:
            st.image(st.session_state.uploaded_img, caption="🎬 Fairyland Living Picture", use_container_width=True)
        
        if os.path.exists(st.session_state.audio_path):
            st.audio(st.session_state.audio_path, format="audio/mp3")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="magic-parchment" style="margin-top: 1rem;">
            <h3 style="color: #7209b7 !important;">📖 Full Narrative</h3>
            <p style="font-size: 1.1rem; line-height: 1.7; color: #2b2d42;">
                {st.session_state.story}
            </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🏰 Brew Another Story"):
                if os.path.exists(st.session_state.audio_path):
                    os.remove(st.session_state.audio_path)
                st.session_state.step = 1
                st.session_state.uploaded_img = None
                st.session_state.caption = ""
                st.session_state.story = ""
                st.session_state.audio_path = ""
                st.rerun()


if __name__ == "__main__":
    main()
