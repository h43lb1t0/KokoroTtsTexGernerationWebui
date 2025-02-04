import torch
import librosa
import numpy as np
import gradio as gr
from pathlib import Path
from fairseq import checkpoint_utils
from extensions.KokoroTtsTexGernerationWebui_tts.rmvpe import RMVPE
from extensions.KokoroTtsTexGernerationWebui_tts.vc_infer_pipeline import VC
from extensions.KokoroTtsTexGernerationWebui_tts.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from multiprocessing import cpu_count
from scipy.io import wavfile
# Shared RVC models
hubert_model = None
rmvpe_model = None

class RVCConfig:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = True if torch.cuda.is_available() else False
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            # Adjust settings based on GPU capabilities
            if ("16" in self.gpu_name and "V100" not in self.gpu_name.upper()) \
                or "P40" in self.gpu_name.upper() \
                or "1060" in self.gpu_name \
                or "1070" in self.gpu_name \
                or "1080" in self.gpu_name:
                self.is_half = False
        else:
            self.device = "cpu"
            self.is_half = False

        # Set default values
        if self.is_half:
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        # Adjust for low VRAM
        if self.gpu_mem and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max

# Shared models
hubert_model = None
rmvpe_model = None
rvc_config = RVCConfig()

def load_models():
    global hubert_model, rmvpe_model
    if hubert_model is None:
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            ["extensions/KokoroTtsTexGernerationWebui_tts/models/hubert_base.pt"],
            suffix="",
        )
        hubert_model = models[0].to(rvc_config.device)
        hubert_model = hubert_model.half() if rvc_config.is_half else hubert_model.float()
    
    if rmvpe_model is None:
        # CORRECTED RMVPE INITIALIZATION
        rmvpe_model = RMVPE(
            "extensions/KokoroTtsTexGernerationWebui_tts/models/rmvpe.pt",
            rvc_config.is_half,
            rvc_config.device
        )

def process_with_rvc(audio_path, model_name, transpose=0, index_rate=0.75, protect=0.33):
    load_models()
    
    try:
        # Load RVC model
        model_path = Path(f"extensions/KokoroTtsTexGernerationWebui_tts/rvc_models/{model_name}")
        cpt = torch.load(model_path, map_location="cpu")
        
        # Get model configuration
        tgt_sr = cpt["config"][-1]
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")

        # Load the correct model architecture
        if version == "v1":
            if if_f0 == 1:
                net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=rvc_config.is_half)
            else:
                net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
        else:
            if if_f0 == 1:
                net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=rvc_config.is_half)
            else:
                net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])

        net_g.load_state_dict(cpt["weight"], strict=False)
        net_g.eval().to(rvc_config.device)
        net_g = net_g.half() if rvc_config.is_half else net_g.float()

        # Load and process audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        vc = VC(tgt_sr, rvc_config)
        
        # Add resampling similar to edge_tts
        resample_sr = 0  # Set to target sample rate if needed
        
        audio_processed = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            str(audio_path),
            [0, 0, 0],
            transpose,
            'rmvpe',
            "",
            index_rate,
            if_f0,
            3,
            tgt_sr,
            resample_sr,
            0.25,
            version,
            protect,
            None
        )
        
        # Save with proper sample rate
        wavfile.write(audio_path, tgt_sr, audio_processed.astype(np.int16))
        
    except Exception as e:
        print(f"Error in RVC processing: {str(e)}")
        raise e

def load_rvc_model(cpt):
    # Model loading logic similar to edge_tts implementation
    if cpt.get("version", "v1") == "v1":
        if cpt.get("f0", 1) == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=True)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    else:
        if cpt.get("f0", 1) == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=True)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    
    net_g.load_state_dict(cpt["weight"], strict=False)
    return net_g.half().to('cuda')

def refresh_rvc_models():
    models = []
    for file in Path("extensions/KokoroTtsTexGernerationWebui_tts/rvc_models").glob("*.pth"):
        models.append(file.name)
    return gr.Dropdown.update(choices=models)