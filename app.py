import os
import warnings
import tensorflow as tf

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warnings, 3=errors

# Warning filters
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')  # Only show errors, not warnings

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, logging

# Set transformers logging to error only
logging.set_verbosity_error()

load_dotenv(find_dotenv())

#img2txt
def image2text(url):
    pipe = pipeline("image-to-text", 
                   model="Salesforce/blip-image-captioning-base",
                   use_fast=True) 

    result = pipe(url, max_new_tokens=20)[0]['generated_text']  
    
    print(result)
    return result

image2text("im1.jpg")
