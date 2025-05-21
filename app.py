"""
app.py: Core logic for the multimodal narrative generation pipeline.

This script orchestrates a three-step process:
1. Image Captioning: Generates a descriptive text caption for a given image URL
   using a pre-trained model.
2. Story Generation: Takes the image caption (or any text input) and generates
   a short, emotive story snippet using a large language model.
3. Text-to-Speech (TTS): Converts the generated story into speech and saves it
   as an audio file.

The pipeline leverages several libraries and external services, including:
- Hugging Face Transformers for image captioning.
- LangChain and Groq for story generation.
- Bark for text-to-speech synthesis.

Environment variables are used for API keys and model configurations.
"""
import os
import warnings
import tensorflow as tf
import torch.nn.utils.parametrizations
import logging

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations for TensorFlow to avoid potential conflicts or issues.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ logging, showing only errors.
os.environ["SUNO_OFFLOAD_CPU"] = "True"  # Offload Bark TTS computations to CPU if CUDA is not available or to save GPU memory.
os.environ["SUNO_USE_SMALL_MODELS"] = "True"  # Use smaller Bark TTS models to reduce computational load and memory usage.
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism to prevent potential issues with some environments.

# Warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Ignore deprecation warnings to keep the output clean.
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore user warnings.
warnings.filterwarnings('ignore', category=FutureWarning)  # Ignore future warnings.
warnings.simplefilter('ignore')  # Ignore all simple warnings.

# Logging configuration
logging.getLogger("torch").setLevel(logging.ERROR)  # Set PyTorch logging to ERROR level to minimize verbosity.
tf.get_logger().setLevel('ERROR')

# Torch specific settings
torch.set_warn_always(False)

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# Download and load all Bark TTS models into memory upon script initialization.
# This can take some time but speeds up subsequent TTS generation calls.
preload_models()

# Set transformers logging to error only
logging.set_verbosity_error()

load_dotenv(find_dotenv())

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

#img2txt
def image2text(url):
    """
    Generates a descriptive caption for an image from a given URL.

    This function uses the 'Salesforce/blip-image-captioning-base' model via
    the Hugging Face pipeline to perform image-to-text captioning.

    Args:
        url (str): The URL of the image to caption.

    Returns:
        str: The generated text caption for the image.
    """
    pipe = pipeline("image-to-text", 
                   model="Salesforce/blip-image-captioning-base", # Model for image captioning
                   use_fast=True) 

    # max_new_tokens is set to 20 to keep captions concise. This can be adjusted.
    result = pipe(url, max_new_tokens=20)[0]['generated_text']  
    
    # print(result)
    return result

#LLM
def gen_story(scene):
    """
    Generates a short, emotive story based on a given scene description.

    This function uses the 'mixtral-8x7b-32768' model via ChatGroq (LangChain)
    to create a narrative. The prompt is structured to guide the LLM to produce
    a story with emotional cues and a specific style.

    Args:
        scene (str): The input text (e.g., an image caption) to base the story on.

    Returns:
        str: The generated story content.
    """
    # Template for the story generation prompt.
    # - The "system" message sets the persona of the LLM (a female webnovel writer).
    # - The "human" message provides the instruction and the scene input, asking for
    #   a short narrative with emotional cues.
    template = [
        ("system", "You are a professional and imaginative female webnovel writer, known for your ability to craft captivating and emotionally rich stories."),
        ("human", """Compose a compelling and vivid narrative, incorporating appropriate emotional cues such as ([laughter], [laughs], [sighs], [music], [gasps], [clears throat], '—' or '...' for hesitations, '♪' for song lyrics, CAPITALIZATION for emphasis, etc.) in under 20 words about: {scene}.""")
    ]

    prompt = ChatPromptTemplate.from_messages(template)
    
    llm = ChatGroq(
        temperature=1,
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768"
    )
    
    # Created the chain 
    chain = prompt | llm
    
    # Get story response
    response = chain.invoke({"scene": scene})
    return response.content

#tts
def gen_tts(text):
    """
    Converts input text into speech and saves it as 'bark_generation.wav'.

    This function utilizes the 'bark' library for text-to-speech synthesis.
    The generated audio is saved at a predefined sample rate.

    Args:
        text (str): The text to be converted to speech.

    Raises:
        Exception: Propagates exceptions that occur during audio generation
                   or file writing, after printing an error message.
    """
    try:
       # Generate audio data from the input text using Bark.
       audio_array = generate_audio(text)
       # Save the generated audio data as a WAV file.
       write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)   
    except Exception as e:
        # If any error occurs during TTS generation or saving, print an error and re-raise.
        print(f"Error generating speech: {str(e)}")
        raise

if __name__ == "__main__":
    # This block executes when the script is run directly.
    # It provides an example usage of the pipeline:
    # 1. Generate a caption for the image "im1.jpg".
    # 2. Generate a story based on the caption.
    # 3. Print the story (stripping any surrounding quotes).
    # 4. Convert the story to speech and save it as "bark_generation.wav".
    story = gen_story(image2text("im1.jpg"))
    print(story.strip("\""))
    gen_tts(story)
