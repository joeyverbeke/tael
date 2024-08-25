def validate_transcription(text):
    """Validate if the transcribed text is valid."""
    invalid_phrases = [
        "too blurry", 
        "Setting `pad_token_id`", 
        "no text visible", 
        "error",
        "does not contain",
        "too dark",
        "cannot transcribe",
        "the text in the image",
        "the image",
    ]
    return not any(phrase in text.lower() for phrase in invalid_phrases)
