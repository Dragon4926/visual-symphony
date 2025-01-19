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
from langchain import PromptTemplate, LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Set transformers logging to error only
logging.set_verbosity_error()

load_dotenv(find_dotenv())

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
        ("system", "You are an experienced webnovel writer."),
        ("human", """Create an enchanting tale in under 20 words about: {scene}.""")
    ]

    prompt = ChatPromptTemplate.from_messages(template)
    
    llm = ChatGroq(
        temperature=1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )
    
    # Created the chain 
    chain = prompt | llm
    
    # Get story response
    response = chain.invoke({"scene": scene})
    return response.content

#tts

if __name__ == "__main__":
    story = gen_story(image2text("im1.jpg")).strip("\"")
    print(story)