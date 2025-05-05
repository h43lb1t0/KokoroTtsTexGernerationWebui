def extract_quotes_with_speaker_tags(text):
    """
    Extracts quotes with speaker tags from a text and returns a list containing 
    tuples with text and speaker tags.
    
    Args:
        text (str): The input text containing quotes with speaker tags in the format:
                   "quoted text"[speaker_tag]
    
    Returns:
        list: A list where all elements are represented as tuples:
              - Regular text: (text, None)
              - Quotes with tags: (quote, speaker_tag)
    """
    result = []
    current_text = ""
    i = 0
    
    while i < len(text):
        # If we find an opening quote
        if text[i] == '"':
            # Add any accumulated text before the quote as tuple with None
            if current_text.strip():
                result.append((current_text.strip(), None))
                current_text = ""
            
            # Extract the quote
            quote_start = i
            i += 1  # Move past the opening quote
            
            while i < len(text) and text[i] != '"':
                i += 1
                
            if i < len(text):  # Found closing quote
                quote_end = i
                quote = text[quote_start:quote_end+1]  # Include quotes
                
                # Look for speaker tag
                if i+1 < len(text) and text[i+1] == '[':
                    tag_start = i+1
                    i += 2  # Move past the '['
                    
                    while i < len(text) and text[i] != ']':
                        i += 1
                        
                    if i < len(text):  # Found closing bracket
                        tag_end = i
                        speaker_tag = text[tag_start+1:tag_end]  # Extract tag without brackets
                        result.append((quote, speaker_tag))
                        i += 1  # Move past the closing ']'
                    else:
                        # No closing bracket found, treat as regular text with None tag
                        result.append((quote, None))
                else:
                    # No speaker tag, add quote with None
                    result.append((quote, None))
            else:
                # No closing quote found, add accumulated text
                current_text += text[quote_start:i]
        else:
            current_text += text[i]
            i += 1
    
    # Add any remaining text as tuple with None
    if current_text.strip():
        result.append((current_text.strip(), None))
        
    return result


