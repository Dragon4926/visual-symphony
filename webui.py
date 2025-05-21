"""
webui.py: Streamlit Web User Interface for the Visual Narrative Generator.

This script creates an interactive web application that allows users to:
1. Upload an image.
2. View an AI-generated caption for the image.
3. Generate a short story based on the image caption.
4. Listen to an audio narration of the generated story.

It utilizes functions from `app.py` for the core generation tasks
(image captioning, story generation, and text-to-speech) and uses
Streamlit for building the user interface components and managing
the application flow.
"""
import os
import warnings
import streamlit as st
from PIL import Image
import tempfile
import logging
from dotenv import find_dotenv, load_dotenv
from app import image2text, gen_story, gen_tts

# Configure environment and logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations for TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logging (show only errors).
warnings.filterwarnings('ignore')  # Ignore all warnings to keep the UI clean.
logging.getLogger("transformers").setLevel(logging.ERROR) # Set Hugging Face Transformers logging to ERROR.

# Streamlit page configuration
st.set_page_config(
    page_title="Visual Story Generator",  # Title appearing on the browser tab.
    page_icon="🎨",  # Favicon for the page.
    layout="wide",  # Use wide layout for the page content.
    initial_sidebar_state="expanded"  # Keep the sidebar expanded by default.
)

def main():
    """
    Constructs and runs the Streamlit web application for the Visual Narrative Generator.

    This function defines the entire UI layout and manages the interactive workflow:
    - Sets the main page title and a brief description.
    - Creates a sidebar for API key configuration and advanced settings (story length, creativity).
    - Divides the main page into two columns:
        - Column 1: Handles image upload and displays the generated caption.
        - Column 2: Handles story generation from the caption and TTS conversion,
                    allowing users to listen to and download the audio.
    - Utilizes `image2text`, `gen_story`, and `gen_tts` from `app.py` to perform
      the core AI-powered generation tasks.
    - Implements error handling to display issues gracefully within the UI.
    """
    st.title("Visual Narrative Generator")  # Main title of the web application.
    st.markdown("Transform images into immersive audio stories using AI") # Subtitle or description.
    
    # Sidebar for configuration options
    with st.sidebar:
        st.header("Configuration")
        # Input field for GROQ API Key
        api_key = st.text_input("GROQ API Key", type="password", 
                              help="Get your API key from https://console.groq.com/keys")
        
        st.markdown("---") # Visual separator
        st.caption("Advanced Settings")
        # Slider for maximum story length (currently not directly used by gen_story in app.py)
        max_length = st.slider("Max Story Length", 10, 50, 20)
        # Slider for creativity level (temperature for LLM, currently not directly used by gen_story)
        temperature = st.slider("Creativity Level", 0.1, 2.0, 1.0)
    
    # Main page layout with two columns
    col1, col2 = st.columns([1, 2]) # Define column proportions
    
    # Column 1: Image Upload and Captioning
    with col1:
        st.subheader("1. Upload Image")
        # File uploader widget for image selection
        uploaded_file = st.file_uploader("Choose an image...", 
                                       type=["jpg", "jpeg", "png"], # Allowed file types
                                       help="Supported formats: JPG, PNG, JPEG")
        
        # Logic to process the uploaded file
        if uploaded_file:
            # Use a temporary file to store the uploaded image data.
            # This is necessary because image2text expects a file path.
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name # Get the path to the temporary file
            
            try:
                # Display a spinner while the image is being analyzed
                with st.spinner("Analyzing image..."):
                    caption = image2text(tmp_path) # Call image captioning function
                
                # Display the uploaded image and its generated caption
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                st.success("Image analysis complete!")
                st.subheader("Image Insights")
                st.markdown(f"**Caption:** {caption}")
                
            except Exception as e:
                # Display an error message if image processing fails
                st.error(f"Image processing error: {str(e)}")
                # Clean up the temporary file if it exists
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                return # Stop further execution in this column
            finally:
                # Ensure temporary file is deleted after use
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # Column 2: Story Generation and Text-to-Speech
    with col2:
        # This section only appears if an image has been uploaded and a caption generated
        if uploaded_file and 'caption' in locals() and caption:
            st.subheader("2. Generate Story")
            # Button to trigger story generation
            if st.button("Create Narrative", help="Generate story from image analysis"):
                try:
                    # Display a spinner while the story is being crafted
                    with st.spinner("Crafting story..."):
                        story = gen_story(caption) # Call story generation function
                    
                    st.subheader("Generated Story")
                    st.markdown(f"*{story}*") # Display the generated story
                    st.success("Narrative generated successfully!")
                    
                    st.subheader("3. Audio Conversion")
                    # Display a spinner while audio is being generated
                    with st.spinner("Generating audio..."):
                        gen_tts(story) # Call TTS function
                        
                    # Audio player widget for the generated speech
                    st.audio("bark_generation.wav")
                    # Provide a download button for the audio file
                    with open("bark_generation.wav", "rb") as f:
                        st.download_button("Download Audio", f, 
                                        file_name="generated_story.wav", # Suggested download filename
                                        mime="audio/wav") # Mime type for WAV audio
                        
                except Exception as e:
                    # Display an error message if story/audio generation fails
                    st.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    # This ensures that main() is called only when the script is executed directly
    main()
