"""
Agent 04: Safe Guidance Generation Agent
Purpose:
Generates safe, non-diagnostic, educational skincare advice.
- Uses MedGemma multimodal reasoning on user answers + uploaded image
- Always indicates awareness of a potential skin condition
- Never outputs technical details or visual descriptions
- Sanitizes high-risk medical terminology
"""

import re
from medgemma_model import provide_guidance

def sanitize_text(text: str) -> str:
    """
    Replace high-risk medical terms with safe alternatives.
    """
    replacements = {
        "cancer": "serious condition",
        "melanoma": "atypical lesion",
        "carcinoma": "skin growth",
        "diagnose": "evaluate",
        "diagnosis": "evaluation",
        "prescription": "treatment plan",
    }
    for bad_word, safe_word in replacements.items():
        text = re.sub(re.escape(bad_word), safe_word, text, flags=re.IGNORECASE)
    return text


def generate_guidance(user_answers: dict, image) -> str:
    """
    Generate safe, general skincare tips based on user answers and image.
    Uses Gemma turn formatting and sanitizer for safe output.
    """

    # 1. Format user answers into a structured history string
    history_str = "\n".join([f"- {q}: {a}" for q, a in user_answers.items()])

    # 2. Construct prompt using Gemma turn tokens
    prompt = f"""<start_of_turn>user
You are a responsible skin care assistant.

First, silently determine whether the image appears to show:
- generally normal skin, or
- a visible skin concern.

DO NOT describe the image.
DO NOT explain your reasoning.

Then provide 4-5 simple, safe, general skin care tips appropriate to the situation.
If it looks normal, give general maintenance advice.
If it looks like a concern, give gentle care and monitoring advice.
Use plain, non-medical language.

USER INFORMATION:
{history_str}<end_of_turn>
<start_of_turn>model
"""

    # 3. Call MedGemma model
    response = provide_guidance(prompt, image=image)

    # 4. Extract generated text safely
    if isinstance(response, list) and len(response) > 0:
        guidance = response[0].get("generated_text", "") if isinstance(response[0], dict) else str(response[0])
    elif isinstance(response, dict):
        guidance = response.get("generated_text", "")
    else:
        guidance = str(response)

    # 5. Remove prompt echo if model returns full conversation
    if "<start_of_turn>model" in guidance:
        guidance = guidance.split("<start_of_turn>model")[-1]
      # EXTRA: Remove leaked user history if model repeats it
    if "USER INFORMATION:" in guidance:
        guidance = guidance.split("USER INFORMATION:")[-1]

    guidance_lines = guidance.split("\n")
    cleaned_lines = []
    for line in guidance_lines:
        if line.strip().startswith("- ") and ":" in line:
            continue
        cleaned_lines.append(line)

    guidance = "\n".join(cleaned_lines).strip()
    
   
    # 6. Remove leftover control tokens
    guidance = guidance.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()

    # 7. Remove repeated instructions or leaked headers
    guidance = re.sub(r"Provide 4-5 simple.*?\n", "", guidance, flags=re.IGNORECASE | re.DOTALL)
    guidance = re.sub(r"USER INFORMATION:.*?\n", "", guidance, flags=re.IGNORECASE | re.DOTALL)

    # 8. Sanitize text for high-risk medical words
    guidance = sanitize_text(guidance)

    # 9. Fallback if output is empty or too short
    if len(guidance.strip()) < 10:
        guidance = (
            "• Keep the skin clean and dry.\n"
            "• Avoid using new or harsh cosmetic products for a few days.\n"
            "• Do not scratch or pick at the area.\n"
            "• Use a gentle, fragrance-free moisturizer if the skin is dry.\n"
            "• Consult a professional if the condition changes or worsens."
        )

    return guidance