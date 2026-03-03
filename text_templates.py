"""
text_templates.py

Purpose:
Contains all user-facing text used by the application.

Why separate text from logic?
- Improves maintainability
- Reduces hallucination risk
- Centralizes safety control
- Enables easy auditing

Medical Safety Principles:
- No diagnosis
- No medication instructions
- No certainty claims
- Always encourage professional consultation
"""


# IMAGE VALIDATION MESSAGES (Agent 1)

IMAGE_ACCEPTED = (
    "Thanks for uploading the image. I’ll take a closer look to understand "
    "general visual characteristics of the affected skin area."
)

IMAGE_REJECTED = (
    "The uploaded image does not appear to clearly show a skin area. "
    "Please upload a clear photo of the affected skin region under good lighting."
)

# message after similarity level
IMAGE_REVIEWED = ("Your image has been reviewed."
                 " Please answer a few brief questions to help provide more personalized guidance")

# IMAGE SIMILARITY EXPLANATION (Agent 2)

SIMILARITY_LOW_EXPLANATION = (
    "The visual pattern in the image does not closely resemble common reference patterns. "
    "This usually suggests a lower level of concern, but visual analysis alone is not enough."
)

SIMILARITY_MEDIUM_EXPLANATION = (
    "The image shows some similarities with known skin-related patterns. "
    "Additional information will help provide better guidance."
)

SIMILARITY_HIGH_EXPLANATION = (
    "The image shows strong similarities with known skin-related patterns. "
    "It is important to gather more details to understand the situation safely."
)


# FOLLOW-UP QUESTIONS (Agent 3)
# These are phrased to be neutral, non-diagnostic, and inclusive

FOLLOWUP_QUESTIONS = {
    "LOW": [
        "Have you noticed any itching or mild discomfort in the area?",
        "How long have you noticed this change on your skin?",
        "Have you recently used any new skincare or cosmetic products?"
    ],

    "MEDIUM": [
        "Is the affected area itchy, burning, or uncomfortable?",
        "Do you feel pain or tenderness when touching the area?",
        "Has the area changed in size, color, or texture over time?",
        "Have you been exposed to sunlight for long periods recently?",
        "Did you eat anything unusual before noticing this skin change?"
    ],

    "HIGH": [
        "Is the area painful, swollen, or warm to the touch?",
        "Have you noticed rapid spreading or worsening of the condition?",
        "Is there any oozing, bleeding, or severe irritation?",
        "Did this appear suddenly after an insect bite or sting?",
        "Are there any additional symptoms such as fever or fatigue?"
    ]
}


# UNIVERSAL DISCLAIMER (Always shown)

MEDICAL_DISCLAIMER = (
    "This tool does not provide medical diagnoses or treatment."
    "It is intended for informational purposes only and should not replace professional medical advice."
)
