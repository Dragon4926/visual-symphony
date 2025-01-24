import os
import warnings
import streamlit as st
from PIL import Image
import tempfile
import logging
from dotenv import find_dotenv, load_dotenv
from app import image2text, gen_story, gen_tts

# Configure environment and logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Streamlit page configuration
st.set_page_config(
    page_title="Visual Story Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("Visual Narrative Generator")
    st.markdown("Transform images into immersive audio stories using AI")
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("GROQ API Key", type="password", 
                              help="Get your API key from https://console.groq.com/keys")
        
        st.markdown("---")
        st.caption("Advanced Settings")
        max_length = st.slider("Max Story Length", 10, 50, 20)
        temperature = st.slider("Creativity Level", 0.1, 2.0, 1.0)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"],
                                       help="Supported formats: JPG, PNG, JPEG")
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                with st.spinner("Analyzing image..."):
                    caption = image2text(tmp_path)
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                st.success("Image analysis complete!")
                st.subheader("Image Insights")
                st.markdown(f"**Caption:** {caption}")
                
            except Exception as e:
                st.error(f"Image processing error: {str(e)}")
                return

    with col2:
        if uploaded_file and caption:
            st.subheader("2. Generate Story")
            if st.button("Create Narrative", help="Generate story from image analysis"):
                try:
                    with st.spinner("Crafting story..."):
                        story = gen_story(caption)
                    
                    st.subheader("Generated Story")
                    st.markdown(f"*{story}*")
                    st.success("Narrative generated successfully!")
                    
                    st.subheader("3. Audio Conversion")
                    with st.spinner("Generating audio..."):
                        gen_tts(story)
                        
                    st.audio("bark_generation.wav")
                    with open("bark_generation.wav", "rb") as f:
                        st.download_button("Download Audio", f, 
                                        file_name="generated_story.wav",
                                        mime="audio/wav")
                        
                except Exception as e:
                    st.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()
