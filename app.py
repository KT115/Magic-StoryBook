import os
import streamlit as st
from PIL import Image
from transformers import pipeline
from gtts import gTTS

# ---------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Loads and caches the Hugging Face pipelines:
      - Image Captioning: Salesforce/blip-image-captioning-base
      - Text Generation:  gpt2
    """
    # "image-captioning" task is used to avoid KeyError in transformers
    caption_model = pipeline(
        "image-captioning",
        model="Salesforce/blip-image-captioning-base"
    )
    story_model = pipeline(
        "text-generation",
        model="gpt2"
    )
    return caption_model, story_model


# ---------------------------------------------------------
# 2. Pipeline Processing Functions
# ---------------------------------------------------------
def get_caption(image, caption_pipe):
    """
    Takes a PIL Image object and generates a descriptive text caption.
    """
    result = caption_pipe(image)
    return result[0]["generated_text"]


def get_story(caption, story_pipe):
    """
    Expands the image caption into a child-friendly story of 50-100 words.
    """
    # Prompt framed for children aged 3-10
    prompt = f"Once upon a time, there was {caption}. One sunny day,"
    
    output = story_pipe(
        prompt,
        max_new_tokens=80,
        min_length=50,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2
    )
    
    raw_story = output[0]["generated_text"]
    
    # Trim to the last complete sentence for a clean ending
    if "." in raw_story:
        story = raw_story[:raw_story.rfind(".") + 1]
    else:
        story = raw_story
        
    return story


def text_to_speech(text, filename="story_audio.mp3"):
    """
    Converts text to an MP3 audio file using gTTS (Google Text-to-Speech).
    """
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename


# ---------------------------------------------------------
# 3. Streamlit User Interface
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Magic Storybook", page_icon="📖", layout="centered")
    
    st.title("📖 Magic Storybook for Kids")
    st.write("Upload a picture to turn it into a bedtime story with audio narration!")

    # Load pipelines once
    caption_pipe, story_pipe = load_models()

    # Image upload widget
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Picture", use_container_width=True)

        # Trigger story generation
        if st.button("✨ Generate Story ✨"):
            with st.spinner("Analyzing image, writing story, and generating voice..."):
                # Step A: Image Captioning
                caption = get_caption(image, caption_pipe)
                st.info(f"**Image Caption:** {caption.capitalize()}")

                # Step B: Story Generation
                story = get_story(caption, story_pipe)
                st.subheader("Your Story:")
                st.write(story)

                # Step C: Text-to-Speech Conversion
                audio_path = text_to_speech(story)
                st.subheader("🔊 Listen Along:")
                st.audio(audio_path, format="audio/mp3")

                # Clean up local audio file after Streamlit serves it
                if os.path.exists(audio_path):
                    os.remove(audio_path)


if __name__ == "__main__":
    main()
