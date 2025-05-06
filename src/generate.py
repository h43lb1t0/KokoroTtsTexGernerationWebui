import gc
import shutil
import uuid
from .models import build_model
import torch
import os
import numpy as np
from pydub import AudioSegment
import pathlib
from huggingface_hub import snapshot_download
from modules import shared
from nltk.tokenize import sent_tokenize
import nltk

# Download the Kokoro weights

def download_kokoro_weights():
    """Download the Kokoro weights."""
    
    snapshot_path_base = pathlib.Path(__file__).parent / 'models--hexgrad--Kokoro-82M' / 'snapshots'
    try:
        snapshot_paths = os.listdir(snapshot_path_base)
    except FileNotFoundError:
        snapshot_paths = None
    if snapshot_paths:
        for snapshot_path in snapshot_paths:
            if (snapshot_path_base / snapshot_path / 'kokoro-v0_19.pth').is_file():
                shutil.rmtree(snapshot_path_base)
                break


    snapshot_download(repo_id="hexgrad/Kokoro-82M", cache_dir =pathlib.Path(__file__).parent)
    if not snapshot_paths:
        snapshot_paths = os.listdir(snapshot_path_base)
    from .voices import VOICES

    return snapshot_path_base / snapshot_paths[0]

snapshot_path = download_kokoro_weights()

# Download the models for sentence splitting
nltk.download('punkt')
nltk.download('punkt_tab')

# Set the environment variables for eSpeak NG on Windows
if os.name == 'nt':
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from .kokoro import generate, tokenize, phonemize

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = 'cuda:0'
    else:
        device = 'cuda'
else:
    device = 'cpu'


    


model_path = snapshot_path / 'kokoro-v1_0.pth'
MODEL = None

voice_name, voicepack = None, None

def load_voice(voice=None):
    """Load a voice by name.
    
    Args:
        voice (str, optional): The name of the voice to load. Defaults to `af`.
    """
    global voice_name, voicepack
    voice_name = voice or ('af_bella' or VOICES[0])
    voise_path = snapshot_path / 'voices' / f'{voice_name}.pt'
    voicepack = torch.load(voise_path, weights_only=True).to(device)
    print(f'Loaded voice: {voice_name}')


load_voice()


def run(text, preview=False, rvc_params=None):
    """Generate audio from text.

    Args:
        text (str): The text to generate audio from.
        preview (bool, optional): Whether to generate a preview audio. Defaults to False.

    Returns:
        str: The message ID.
    """
    if rvc_params is None:
        rvc_params = {'rvc': False, 'rvc_model': None}  # Safe default
    global MODEL, voicepack
    MODEL = build_model(model_path, device)
    msg_id = str(uuid.uuid4())
    text_chunks = split_text(text)
    try:
        segments = generate_audio_chunks(text_chunks)
    except IndexError:
        if sentence_based:
            set_plitting_type("Word")
            text_chunks = split_text(text)
            segments = generate_audio_chunks(text_chunks)
            set_plitting_type()

    full_adio = concatenate_audio_segments(segments)
    audio_path = pathlib.Path(__file__).parent / '..' / 'audio' / f'{"preview" if preview else msg_id}.wav'
    full_adio.export(audio_path, format="wav")

    del MODEL
    gc.collect()
    if rvc_params.get('rvc', False) and rvc_params.get('rvc_model'):
        from extensions.KokoroTtsTexGernerationWebui_tts.rvc import process_with_rvc  # Use relative import
        
        process_with_rvc(
            audio_path,
            rvc_params['rvc_model'],
            transpose=rvc_params.get('transpose', 0),
            index_rate=rvc_params.get('index_rate', 0.75),
            protect=rvc_params.get('protect', 0.33)
        )

    return msg_id

sentence_based = True

def set_plitting_type(method="Split by sentence"):
    """Set the splitting method for the text.

    Args:
        method (str, optional): The splitting method. Defaults to "Split by sentence".
    """
    global sentence_based
    sentence_based = True if method == "Split by sentence" else False
    print(f'Splitting method: {"sentence" if sentence_based else "Word"}')

set_plitting_type()

def split_text(text):
    """Split the text into chunks of sentences or word up to 510 token.

    Args:
        text (str): The text to split.

    Returns:
        list: The text chunks.
    """

    global MODEL
    
    max_token = 510
    text_parts = sent_tokenize(text) if sentence_based else text.split()
    current_text_parts = []
    chunks = []
    current_chunk_len = 0



    for text_part in text_parts:
        tokenized_textpart = tokenize(phonemize(text_part, lang=voice_name[0]))
        additional_tokens = len(tokenized_textpart) + 1

        if current_chunk_len + additional_tokens > max_token and current_text_parts:
            # Create the chunk from what's accumulated so far
            current_text = ' '.join(current_text_parts)
            tokenized_chunk = tokenize(phonemize(current_text, lang=voice_name[0]))
            chunks.append(tokenized_chunk)

            # Reset trackers
            current_text_parts = []
            current_chunk_len = 0

            
        current_text_parts.append(text_part)
        current_chunk_len += additional_tokens


    # Add remaining words as the final chunk if any
    if current_text_parts:
        current_text = ' '.join(current_text_parts)
        tokenized_chunk = tokenize(phonemize(current_text, lang=voice_name[0]))
        chunks.append(tokenized_chunk)


    del text_parts

    return chunks


def generate_audio_chunks(chunks):
    """Generate audio chunks from the text chunks.

    Args:
        chunks (list): The text chunks.

    Returns:
        list: The audio segments.
    """

    out = {'out': [], 'ps': []}
    for i, chunk in enumerate(chunks):
        out_chunk, ps = generate(MODEL, chunk, voicepack, lang=voice_name[0])
        out['out'].append(out_chunk)
        out['ps'].append(ps)

    segments = []

    for i, chunk in enumerate(out['out']):
        # Normalize to 16-bit PCM
        normalized_audio = np.int16(chunk / np.max(np.abs(chunk)) * 32767)

        # Create an AudioSegment
        segments.append(AudioSegment(
            data=normalized_audio.tobytes(),
            sample_width=normalized_audio.dtype.itemsize,  # 2 bytes for int16
            frame_rate=24000,
            channels=1
        ))

    return segments

def concatenate_audio_segments(segments):
    """Concatenate audio segments.

    Args:
        segments (list): The audio segments to concatenate.

    Returns:
        AudioSegment: The concatenated audio segment.
    """

    # Concatenate all segments
    audio_segment = segments[0]
    for segment in segments[1:]:
        audio_segment += segment

    return audio_segment

if __name__ == '__main__':
    run("Hello, this is an example of a text for Kokoro")
