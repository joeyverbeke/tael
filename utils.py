def validate_transcription(current_text, last_valid_text):
    """Validate if the transcribed text is valid and return either the valid current or the last valid text."""
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
        "i'm sorry",
        "transcribe",
    ]

    # Check if the current text is valid
    is_valid = not any(phrase in current_text.lower() for phrase in invalid_phrases)
    
    if is_valid:
        return current_text  # Return the current text if valid
    else:
        return last_valid_text  # Return the last valid text if current text is not valid
