import pathlib
import html
import time
from extensions.KokoroTtsTexGernerationWebui_tts.src.generate import run, load_voice, set_plitting_type
from extensions.KokoroTtsTexGernerationWebui_tts.src.voices import VOICES
import gradio as gr
import time
import os

from modules import shared
from extensions.KokoroTtsTexGernerationWebui_tts.rvc import refresh_rvc_models

def input_modifier(string, state):

    shared.processing_message = "*Is recording a voice message...*"
    return string


def voice_update(voice):
    load_voice(voice)
    return gr.Dropdown(choices=VOICES, value=voice, label="Voice", info="Select Voice", interactive=True)

def voice_preview():
    run("This is a preview of the selected voice", 
        preview=True, 
        rvc_params={
            **RVC_PARAMS,
            'transpose': 2  # Match edge_tts default pitch shift
        })
    audio_dir = pathlib.Path(__file__).parent / 'audio' / 'preview.wav'
    audio_url = f'{audio_dir.as_posix()}?v=f{int(time.time())}'
    return f'<audio controls><source src="file/{audio_url}" type="audio/mpeg"></audio>'
#Change these to make your config default
RVC_PARAMS = {
    'rvc': False,
    'rvc_model': None,  #The filename with.pt
    'transpose': 0,
    'index_rate': 0.0,
    'protect': 0.0,
    'f0_method': 'rmvpe'
}
   

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
    with gr.Accordion("RVC Settings", open=False):
        rvc_enable = gr.Checkbox(value=RVC_PARAMS['rvc'], label='Enable RVC Voice Conversion')
        rvc_model = gr.Dropdown(choices=[], value=RVC_PARAMS['rvc_model'], label="RVC Model")
        transpose = gr.Slider(-24, 24, value=RVC_PARAMS['transpose'], step=1, label='Pitch Shift')
        index_rate = gr.Slider(0, 1, value=RVC_PARAMS['index_rate'], label='Index Rate')
        protect = gr.Slider(0, 0.5, value=RVC_PARAMS['protect'], label='Protect')
        refresh_btn = gr.Button("Refresh Models")
        
    # Add event handlers
    rvc_enable.change(lambda x: RVC_PARAMS.update({'rvc': x}), rvc_enable, None)
    rvc_model.change(lambda x: RVC_PARAMS.update({'rvc_model': x}), rvc_model, None)
    transpose.change(lambda x: RVC_PARAMS.update({'transpose': x}), transpose, None)
    index_rate.change(lambda x: RVC_PARAMS.update({'index_rate': x}), index_rate, None)
    protect.change(lambda x: RVC_PARAMS.update({'protect': x}), protect, None)
    refresh_btn.click(refresh_rvc_models, outputs=rvc_model)
    



    
#Personal preference and make it compatible with another extension
def output_modifier(string, state):
    # Escape and clean the text
    string_for_tts = html.unescape(string).replace('*', '').replace('`', '')
    
    # Generate audio file
    msg_id = run(string_for_tts, rvc_params=RVC_PARAMS)
    
    # Create relative path from webui root directory
    audio_path = pathlib.Path(__file__).parent / 'audio' / f'{msg_id}.wav'
    
    # Get relative path from webui working directory
    relative_path = os.path.relpath(audio_path, start=os.getcwd())
    
    # Convert to web-style path and add cache busting
    web_path = f"file/{relative_path.replace(os.sep, '/')}?v={int(time.time())}"
    
    # Add audio element with proper relative path
    return f'{string}<audio controls><source src="{web_path}" type="audio/mpeg"></audio>'
