import pathlib
import html
import time
from extensions.KokoroTtsTexGernerationWebui.src.generate import run, load_voice, set_plitting_type
from extensions.KokoroTtsTexGernerationWebui.src.voices import VOICES
import gradio as gr
import time

from modules import shared

def input_modifier(string, state):

    shared.processing_message = "*Is recording a voice message...*"
    return string


def voice_update(voice):
    load_voice(voice)
    return gr.Dropdown(choices=VOICES, value=voice, label="Voice", info="Select Voice", interactive=True)

def voice_preview():
    run("This is a preview of the selected voice", preview=True)
    audio_dir = pathlib.Path(__file__).parent / 'audio' / 'preview.wav'
    audio_url = f'{audio_dir.as_posix()}?v=f{int(time.time())}'
    return f'<audio controls><source src="file/{audio_url}" type="audio/mpeg"></audio>'
   

def ui():
    info_voice = """Select a Voice. \nThe default voice is a 50-50 mix of Bella & Sarah\nVoices starting with 'a' are American
     english, voices with 'b' are British english"""
    with gr.Accordion("Kokoro"):
        voice = gr.Dropdown(choices=VOICES, value=VOICES[0], label="Voice", info=info_voice, interactive=True)

        preview = gr.Button("Voice preview", type="secondary")

        preview_output = gr.HTML()

        info_splitting ="""Kokoro only supports 510 tokens. One method to split the text is by sentence (default), the otherway
        is by word up to 510 tokens. """
        spltting_method = gr.Radio(["Split by sentence", "Split by Word"], info=info_splitting, value="Split by sentence", label_lines=2, interactive=True)


    voice.change(voice_update, voice)
    preview.click(fn=voice_preview, outputs=preview_output)

    spltting_method.change(set_plitting_type, spltting_method)

    
def input_modifier(string, state, is_chat=False):

    voices_string = ', '.join(VOICES)

    prompt = f"""
    **Instructions:**
    **Task: Add Speaker Tags for Text-to-Speech**

    **Objective:** Append a specific speaker tag IMMEDIATELY after each segment of direct speech in the provided text.

    **Available Voice Names:**

    ```
    {voices_string}
    ```

    **Precise Rules:**

    1.  **Identify Direct Speech:** Locate all text enclosed in quotation marks (`"`).
    2.  **Identify Speaker:** For each quote, determine which character is speaking.
    3.  **Assign Unique Voice Name:**
        *   For each distinct character identified as a speaker, assign ONE unique voice name from the `Available Voice Names` list.
        *   Use names starting with `af_` or `bf_` for characters perceived as female.
        *   Use names starting with `am_` or `bm_` for characters perceived as male.
    4.  **Maintain Consistency:** Once a character is assigned a voice name, use THAT SAME name every time they speak.
    5.  **Append Tag - CRITICAL PLACEMENT:**
        *   Append the assigned voice name in the exact format `[name]`.
        *   Place this tag **IMMEDIATELY** after the closing quotation mark (`"`) of the direct speech. **NO SPACES** between the `"` and the `[`.
        *   **Example:** `"Quote goes here."[assigned_name]`
    6.  **DO NOT MODIFY ORIGINAL TEXT:**
        *   **Crucially:** Do NOT add character names *before* the quotation marks or anywhere else in the narrative text.
        *   The tag `[name]` is the *only* allowed addition to the text.

    **Example:**

    *Original Snippet:*

    ```
    "Hello there," Alice said. "How are you?"
    "I'm fine," replied Bob. "And you?"
    Alice smiled. "Doing well!"
    ```

    *Correct Output (Assuming Alice -> af_alloy, Bob -> am_adam):*

    ```
    "Hello there,"[af_alloy] Alice said. "How are you?"[af_alloy]
    "I'm fine,"[am_adam] replied Bob. "And you?"[am_adam]
    Alice smiled. "Doing well!"[af_alloy]
    ```
    """

    string += prompt

    
    
    return string

def load_and_update_grammar(grammar_filepath: str, names_list: list[str]) -> str:
    """
    Loads a GBNF grammar from a file and updates the 'speaker-name' rule
    with the provided list of names.

    Args:
        grammar_filepath: The path to the GBNF grammar file.
        names_list: A list of strings, where each string is a speaker name.

    Returns:
        A string containing the updated GBNF grammar.

    Raises:
        FileNotFoundError: If the grammar file does not exist.
        ValueError: If the 'speaker-name ::=' rule is not found in the grammar file
                    or if the names_list is empty.
        IOError: If there's an error reading the file.
    """
    import re
    import os
    if not names_list:
        raise ValueError("The list of names cannot be empty.")

    if not os.path.exists(grammar_filepath):
        raise FileNotFoundError(f"Grammar file not found: {grammar_filepath}")

    try:
        with open(grammar_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        raise IOError(f"Error reading grammar file {grammar_filepath}: {e}")

    # Prepare the new speaker names string part
    # Ensure names are quoted and joined by " | "
    formatted_names = [f'"{name}"' for name in names_list]
    speaker_names_definition = " | ".join(formatted_names)
    new_speaker_rule_line = f"speaker-name ::= {speaker_names_definition}\n"

    # Find and replace the speaker-name rule line
    speaker_rule_found = False
    for i, line in enumerate(lines):
        # Use strip() to handle potential leading/trailing whitespace
        # Use regex for a slightly more flexible match (optional, simple startswith is often fine)
        # if line.strip().startswith("speaker-name ::="):
        if re.match(r"^\s*speaker-name\s*::=", line):
            lines[i] = new_speaker_rule_line
            speaker_rule_found = True
            break # Assume only one definition

    if not speaker_rule_found:
        raise ValueError(
            "Could not find the 'speaker-name ::=' rule in the grammar file."
        )

    # Join the modified lines back into a single string
    updated_grammar = "".join(lines)

    return updated_grammar

def state_modifier(state):
    
    # get the path to the grammar file
    grammar_filepath = pathlib.Path(__file__).parent / 'kokoro_grammar.gbnf'

    grammar = load_and_update_grammar(grammar_filepath, VOICES)

    state['grammar_string'] = grammar

    return state


def output_modifier(string, state):


    # Escape the string for HTML safety
    string_for_tts = html.unescape(string)
    string_for_tts = string_for_tts.replace('*', '')
    string_for_tts = string_for_tts.replace('`', '')

    print(string_for_tts)

 
    # Run your custom logic to generate audio
    msg_id = run(string_for_tts)

    # Construct the correct path to the 'audio' directory
    audio_dir = pathlib.Path(__file__).parent / 'audio' / f'{msg_id}.wav'


    # Add the audio playback HTML to the output string
    string += f'<audio controls><source src="file/{audio_dir.as_posix()}" type="audio/mpeg"></audio>'

    return string
