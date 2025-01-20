import os
import warnings
import tensorflow as tf
import torch.nn.utils.parametrizations
import logging

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ["SUNO_OFFLOAD_CPU"] = "True"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter('ignore')

# Logging configuration
logging.getLogger("torch").setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# Torch specific settings
torch.set_warn_always(False)

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav

# download and load all models
preload_models()

# Set transformers logging to error only
logging.set_verbosity_error()

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#img2txt
def image2text(url):
    pipe = pipeline("image-to-text", 
                   model="Salesforce/blip-image-captioning-base",
                   use_fast=True) 

    result = pipe(url, max_new_tokens=20)[0]['generated_text']  
    
    # print(result)
    return result

#LLM
def gen_story(scene):
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
    try:
       audio_array = generate_audio(text)
       write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)   
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        raise

if __name__ == "__main__":
    story = gen_story(image2text("im1.jpg"))
    print(story.strip("\""))
    gen_tts(story)