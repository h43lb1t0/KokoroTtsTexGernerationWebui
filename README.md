# Kokoro-Based TTS Extension for [Oobabooga Text Generation WebUI](https://github.com/oobabooga/text-generation-webui)

Enhance your text generation experience with the Kokoro TTS extension, seamlessly integrating with the Oobabooga Text Generation WebUI.

## License

- **Project License:** This extension is released under the [MIT License](LICENSE) and is built upon the [Original Kokoro 82M Inference Code](https://huggingface.co/hexgrad/Kokoro-82M/tree/c97b7bbc3e60f447383c79b2f94fee861ff156ac).

- **Model Weights:** The model weights are **not** covered by the MIT License. They are licensed under the [Apache 2.0 License](https://huggingface.co/hexgrad/Kokoro-82M) and will be directly downloaded from Hugging Face.

## Features

**Current Version:** Kokoro v1
**Supported Languages:** English

Kokoro TTS is limited to inputs up to **510 tokens**. *Note that Kokoro tokens differ from LLM tokens.* This extension allows you to generate longer audio outputs by splitting the input text into segments and concatenating the resulting audio.

## Audio Play
> [!NOTE] 
> Currently, the Audio Play feature is not available in the main branch as it is still in development, but you can see [here](https://github.com/h43lb1t0/KokoroTtsTexGernerationWebui/blob/personas/Audio_Play.md) how to use it.

Unique TTS voices are assigned to each speaker in dialogue to get a audio play like experience. 

### Text Splitting Methods

- **Split by Sentence:** Divides the text into chunks of complete sentences, each chunk containing fewer than or 510 tokens.
    - This method may fail to split the text into sentences if the input text contains unusual punctuation or formatting. In such cases, the extension will fall back to the "Split by Word" method for this one text.
- **Split by Word:** Divides the text into chunks of individual words, each chunk containing fewer than or 510 tokens.

*I recommend using the "Split by Sentence" method to maintain context and ensure higher quality audio output.*

## Installation

### Prerequisites

Before installing the extension, ensure you have the following dependencies installed:

- **eSpeak:** Download from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases).
- **FFmpeg:** Download from [FFmpeg Downloads](https://ffmpeg.org/download.html).

### Python Dependencies

Install the required Python packages using the appropriate script for your operating system.

#### Windows

1. Run the Windows setup script:
    ```cmd
    .\cmd_windows.bat
    ```
2. Install the Python dependencies:
    ```cmd
    pip install -r extensions\KokoroTtsTextGenerationWebUI\requirements.txt
    ```

#### Linux

1. Run the Linux setup script:
    ```bash
    ./cmd_linux.sh
    ```
2. Install the Python dependencies:
    ```bash
    pip install -r extensions/KokoroTtsTextGenerationWebUI/requirements.txt
    ```

## Multiple GPU Support

By default, the extension utilizes the first available GPU. To specify a different GPU, modify the `device` variable in `src/generate.py` to your desired GPU identifier.

## Roadmap

- [x] Implement the extension
- [x] ~~Kokoro v0.19~~
- [x] Kokoro v1
- [x] Support for all operating systems
- [x] Voice selection feature
- [ ] Only switch to "Split by Word" for the failing text part, not the entire text
- [ ] Support for other languages than English
- [ ] Support for future versions of Kokoro

## Contributing

I welcome contributions to improve this project! If you'd like to contribute, please create a pull request or open an issue. Your improvements and suggestions are highly appreciated.
