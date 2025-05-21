# Visual Symphony - Multimodal Narrative Generator

### Demo
https://github.com/user-attachments/assets/0d0aa59e-4b55-4d80-9f33-4ef76fee6b63

## Project Overview
Visual Symphony is an AI-powered pipeline that transforms visual inputs into immersive audio narratives. It achieves this through a three-step process:

1.  **Image Captioning**: Utilizes the BLIP model to generate a descriptive text caption for an input image.
2.  **Contextual Story Generation**: Employs the Mixtral-8x7B large language model (via Groq API) to craft a short, emotive story based on the image caption.
3.  **Emotional Speech Synthesis**: Leverages the Bark library to convert the generated story into natural-sounding speech with emotional cues.

The project includes a Streamlit-based web user interface (`webui.py`) for easy interaction and a core logic script (`app.py`) that can be used independently.

## Features
- **Multimodal Processing Chain**: Seamlessly converts images to text, text to story, and story to speech.
- **Context-Aware Narratives**: Generates stories that are relevant to the visual input.
- **Emotive Speech**: Produces audio output with natural intonation and emotional expression.
- **Interactive Web UI**: Provides an easy-to-use interface (via Streamlit) for uploading images and experiencing the generated narratives.
- **Modular Core Logic**: The backend functions in `app.py` can be integrated into other Python projects.
- **Optimized for Performance**: Configured for efficient inference, considering TensorFlow/PyTorch interplay and memory usage (e.g., using smaller Bark models, offloading to CPU if needed).

## Installation
```bash
# Clone the repository (if you haven't already)
# git clone <repository_url>
# cd <repository_directory>

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
# .venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the project root directory by copying the `.env.template` or creating a new one:
```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```
Replace `your_groq_api_key_here` with your actual Groq API key.

## Usage

### Through the Web UI
1.  Ensure your `.env` file is configured with the `GROQ_API_KEY`.
2.  Run the Streamlit application:
    ```bash
    streamlit run webui.py
    ```
3.  Open the provided URL in your web browser.
4.  Upload an image, and the application will guide you through generating the story and audio.

### Programmatic Usage (Core Logic)
The core functions are available in `app.py`:
```python
from app import image2text, gen_story, gen_tts

# Example workflow:
try:
    image_path = "path/to/your/image.jpg" # Replace with your image path
    
    # 1. Generate caption from image
    caption = image2text(image_path)
    print(f"Generated Caption: {caption}")
    
    # 2. Generate story from caption
    story = gen_story(caption)
    print(f"Generated Story: {story}")
    
    # 3. Generate audio from story
    # This will save 'bark_generation.wav' in the current directory
    gen_tts(story)
    print("Audio generated as bark_generation.wav")
    
except Exception as e:
    print(f"An error occurred: {e}")

```

## Architecture
The pipeline follows a sequential flow:
```mermaid
graph TD
    A[Image Input (.jpg, .png)] -->|webui.py or app.py| B(BLIP Image Captioning);
    B -- Caption --> C(Mixtral-8x7B Story Generation via Groq);
    C -- Story Text --> D(Bark Speech Synthesis);
    D -- Audio Data --> E[Audio Output (bark_generation.wav)];
```

## Code Documentation
This project emphasizes readable and maintainable code through comprehensive internal documentation:

-   **`app.py`**: This file contains the backend logic for the Visual Narrative Generator. It includes functions for:
    -   `image2text()`: Analyzing an image and generating a text caption.
    -   `gen_story()`: Taking a text input (like a caption) and generating a short story.
    -   `gen_tts()`: Converting the generated story text into an audio file.
    The module itself, each function, and key implementation details are documented with docstrings and inline comments. These explain the purpose, parameters, return values, and models/libraries used.

-   **`webui.py`**: This script builds the interactive web user interface using Streamlit.
    -   The `main()` function orchestrates the UI layout, handles user inputs (like image uploads and API keys), manages the application flow through different stages (image analysis, story creation, audio playback/download), and calls the backend functions from `app.py`.
    Detailed docstrings for the module and the `main()` function, along with inline comments, explain the UI components, their configurations, and the logic flow.

We encourage you to explore the code. The embedded documentation should provide a clear understanding of how each part of the Visual Symphony works.

## License
Apache 2.0 - See the `LICENSE` file for details.
