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

def state_modifier(state):
    # Build a pipe-separated list of raw voice tokens (no extra quotes)
    qouted_voices = [f'"{v}"' for voice in VOICES]
    voices_string = ' | '.join(qouted_voices)

    # A GBNF that forces the tag immediately after the quote
    grammar = f"""
    # Root: sequence of non-quotes and tagged quotes
    root         ::= sequence

    # A sequence is any mix of text outside quotes or properly tagged quotes
    sequence     ::= (non_quote | tagged_quote)*

    # Anything except a double-quote
    non_quote    ::= [^"]+

    # A quoted segment plus its tag
    tagged_quote ::= '"' quoted_content '"' speaker_tag

    # The text inside quotes (no unescaped ")
    quoted_content ::= [^"]*

    # Tag that must follow immediately after the closing quote
    speaker_tag  ::= '[' voice_name ']'

    # Allowed voice names
    voice_name   ::= {voices_string}
    """

    #state['grammar_string'] = grammar.strip()
    #print("State modified with grammar:")
    #print(state['grammar_string'])
    return state

def output_modifier(string, state):


    # Escape the string for HTML safety
    string_for_tts = html.unescape(string)
    string_for_tts = string_for_tts.replace('*', '')
    string_for_tts = string_for_tts.replace('`', '')

 
    # Run your custom logic to generate audio
    msg_id = run(string_for_tts)

    # Construct the correct path to the 'audio' directory
    audio_dir = pathlib.Path(__file__).parent / 'audio' / f'{msg_id}.wav'


    # Add the audio playback HTML to the output string
    string += f'<audio controls><source src="file/{audio_dir.as_posix()}" type="audio/mpeg"></audio>'

    return string
