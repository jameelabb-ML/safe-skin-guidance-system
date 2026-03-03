"""
Safe Skin Guidance System

MedGemma Impact Challenge Submission

This application provides safety-first, non-diagnostic skincare guidance
using a structured multi-agent pipeline:
1. Image validation
2. Similarity estimation via embeddings
3. Controlled follow-up questioning
4. Safe multimodal guidance generation

The system does NOT diagnose medical conditions and does not prescribe treatment.
It is intended for informational purposes only.
"""

import gradio as gr
import time
import re  # Used for text cleaning
from PIL import Image

import agent01_image_validation as agent1
import agent02_image_similarity as agent2
import agent03_followup_questions as agent3
import agent04_guidance_agent as agent4

from text_templates import IMAGE_REJECTED, IMAGE_ACCEPTED, IMAGE_REVIEWED, MEDICAL_DISCLAIMER



# Utility Functions

def clean_guidance_text(text):
    """
    Cleans raw AI guidance text by removing:
    - Bounding box JSON artifacts
    - Trailing numbers or noise
    - Duplicate keywords like "ADVICE"
    - Stray brackets or quotes
    - Double spaces

    Parameters:
        text (str): Raw AI-generated guidance text

    Returns:
        str: Sanitized guidance text ready for display
    """
    if not isinstance(text, str):
        return str(text)

    # 1. Remove all Bounding Box JSON patterns completely
    text = re.sub(r'\[?\s*\{\s*"box_2d":.*?\}\s*\]?', '', text, flags=re.DOTALL)

    # 2. Remove trailing numbers/noise from JSON remnants
    text = re.sub(r'["\s:]+\d+,\s*\d+,\s*\d+,\s*\d+.*', '', text)

    # 3. Remove stray brackets and quotes
    text = re.sub(r'[\[\]\{\}"\']', '', text)

    # 4. Remove duplicate "ADVICE:" keywords
    text = re.sub(r'(?i)advice:', '', text)

    # 5. Fix double spaces and strip
    text = text.replace('  ', ' ').strip()

    return text


# Centered spinner HTML for AI processing
SPINNER_HTML = """
<div style='display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px;'>
    <div style="border: 8px solid #f3f3f3; border-top: 8px solid #3498db; border-radius: 50%; width: 60px; height: 60px; animation: spin 2s linear infinite;"></div>
    <h3 style="margin-top: 20px; font-family: sans-serif; color: #555;">Analyzing your image and responses...</h3>
    <style> @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } } </style>
</div>
"""


def format_success(msg):
    """Formats a success message in HTML with a green banner."""
    return f"""<div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px; border: 1px solid #c3e6cb; margin-bottom: 10px;">
                <strong>✅ SUCCESS:</strong> {msg}</div>"""


def format_error(msg):
    """Formats an error message in HTML with a red banner."""
    return f"""<div style="background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; margin-bottom: 10px;">
                <strong>❌ REJECTED:</strong> {msg}</div>"""



# Pipeline Functions

def start_pipeline(image, state):
    """
    Starts the AI processing pipeline after image upload.

    Steps:
    1. Validate the uploaded image
    2. Compute similarity level
    3. Select first follow-up question
    4. Display status messages and question box

    Parameters:
        image (PIL.Image): Uploaded user image
        state (dict): Current session state

    Returns:
        tuple: Updated UI components and state
    """
    if image is None:
        return gr.update(visible=False), "", {}, gr.update(visible=False)

    state = {"answers": {}, "similarity": None, "current_q": None, "image": image}

    # 1. Image validation
    if not agent1.validate_skin_image(image):
        return gr.update(visible=False), gr.HTML(format_error(IMAGE_REJECTED)), state, gr.update(visible=False)

    # 2. Similarity check
    sim = agent2.similarity_level(image).strip().upper()
    state["similarity"] = sim

    # 3. Get first follow-up question
    first_q = agent3.select_next_question(sim, state["answers"])
    state["current_q"] = first_q

    # 4. Show success banner
    success_html = format_success(f"{IMAGE_ACCEPTED}<br>{IMAGE_REVIEWED}")

    return (
        gr.update(visible=True),
        gr.HTML(success_html),
        state,
        gr.update(label=f"Question: {first_q}", value="", visible=True, interactive=True)
    )


def handle_next_step(user_answer, state):
    """
    Handles submission of a follow-up question.

    Updates state with user's answer, loads the next question if available,
    or shows the final guidance button if questions are done.

    Parameters:
        user_answer (str): User's answer to the current question
        state (dict): Current session state

    Returns:
        tuple: Updated UI components and state
    """
    
    if not state:
        state = {"answers": {}}

    
    # If the user submitted an empty box, do nothing and keep the same question visible
    if not user_answer or user_answer.strip() == "":
        return (
            gr.update(visible=True),                # Keep question container visible
            gr.update(label=f"Question: {state['current_q']}"), # Keep current label
            state,                                  # Don't change state
            gr.update(visible=False)                # Keep 'Get Guidance' button hidden
        )

    # If text exists, proceed with normal logic
    state["answers"][state["current_q"]] = user_answer
    next_q = agent3.select_next_question(state["similarity"], state["answers"])
    state["current_q"] = next_q

    if next_q:
        return gr.update(visible=True), gr.update(label=f"Question: {next_q}", value=""), state, gr.update(visible=False)
    else:
        # Hide question box and show "Get Guidance" button when questions are finished
        return gr.update(visible=False), gr.update(value=""), state, gr.update(visible=True)

def final_guidance_stream(state):
    """
    Generates and streams the final personalized guidance from MedGemma.

    Streaming ensures a smooth display while processing.

    Parameters:
        state (dict): Current session state containing answers and image

    Yields:
        tuple: Updated UI components for streaming
    """
    # Show spinner
    yield gr.update(visible=False), gr.update(visible=True, value=SPINNER_HTML), ""

    # Generate guidance
    raw_message = agent4.generate_guidance(state["answers"], state["image"])
    message = clean_guidance_text(raw_message)

    icon = "🩺"
    safety_footer = (
        "\n\n---\n### 📢 Important Safety Reminder\n"
        "**Please visit a dermatologist immediately if your symptoms worsen, change rapidly, or cause you pain.** "
        "This tool is for guidance only and does not replace professional medical advice."
    )

    full_output = f"### {icon} Personalized Guidance\n\n{message}{safety_footer}"

    # Stream text character by character
    typed_text = ""
    for char in full_output:
        typed_text += char
        yield gr.update(visible=False), gr.update(visible=False), typed_text
        time.sleep(0.005)  # Keep same as original

def reset_app():
    """
    Resets the app to its initial state.

    Returns:
        tuple: Reset values for all UI components
    """
    return (None, "", {}, gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False, value=""))



# Gradio UI Layout

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    """
    Main Gradio interface layout:
    - Left column: image upload + status + reset
    - Right column: follow-up questions + guidance
    - State object maintains session memory across steps
    """
    state = gr.State({})

    gr.Markdown("# 🩺 Safe Skin Guidance System")
    gr.Markdown(f"### ⚠️ Medical Disclaimer\n{MEDICAL_DISCLAIMER}")
    gr.HTML("<hr>")

    with gr.Row():
        # Left column: image upload, status, reset
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Skin Image", sources=["upload", "webcam"])
            status = gr.Markdown()
            clear_btn = gr.Button("🔄 Start Over", variant="secondary")

        # Right column: follow-up questions + guidance
        with gr.Column():
            with gr.Column(visible=False) as q_area:
                gr.Markdown("### Follow Up Questions")
                q_text = gr.Textbox(label="Question", placeholder="Type your answer here...")
                q_btn = gr.Button("Submit Answer", variant="primary")

            final_btn = gr.Button("🔍 Generate Safe Guidance", variant="stop", visible=False)
            loading_box = gr.HTML(visible=False)
            output = gr.Markdown()

    # Event mappings
    img_input.change(start_pipeline, [img_input, state], [q_area, status, state, q_text])
    q_btn.click(handle_next_step, [q_text, state], [q_area, q_text, state, final_btn])
    final_btn.click(final_guidance_stream, [state], [final_btn, loading_box, output])
    clear_btn.click(reset_app, None, [img_input, status, state, q_area, final_btn, output, loading_box])

# Launch the Gradio app
demo.launch()