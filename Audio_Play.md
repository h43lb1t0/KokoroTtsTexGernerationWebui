# Audio Play


---

## Overview

* **Feature name:** Audio Play
* **Purpose:** Automatically assign a unique TTS voice to each speaker in dialogue by forcing the LLM to append explicit speaker tags after every quotation.
* **Status:** Experimental—behavior and quality depend heavily on the underlying LLM’s capabilities.

---

## How to Use

This feature is currently not available in the main branch as it is still in development (But usable as long as you don't need custom grammar).

1. Go to the `extensions\KokoroTtsTexGernerationWebui` directory in your WebUI installation.
2. Open the terminal and run the following command to use the experimental branch:
   ```bash
   git pull
   git checkout personas
   ```
3. Restart your WebUI.
4. Select the `Audio Play` option in the extension settings.

**Revert back to the main branch:**
```bash
git checkout main
```

## How It Works

1. **Prompt Augmentation**
   When you enable Audio Play, the extension adds instructions to your prompts. The injected instructions tell the LLM to:

   * Insert a speaker tag (e.g., `[Alice:]`, `[Bob:]`, etc.) immediately after every closing quotation mark in its **generated** output.
   * Generate tags consistently so that each dialogue participant gets the same tag (and thus the same voice) throughout the generation.

2. **TTS Processing**  \*\*

   * After text generation, the extension parses out each `[Speaker:]` tag.
   * It assigns each unique speaker tag to a distinct Kokoro voice profile.
   * It then synthesizes the speech segments in sequence, stitching them together into one audio file.

---


## Caveats & Known Issues

* **Experimental Quality:**
  The effectiveness of speaker tagging depends on how reliably your LLM follows the injected instructions. You may see missing or mis‑attributed tags if the model deviates.

* **Increased LLM Context Usage:**
  Audio Play’s extra prompt instructions consume part of your LLM’s context window as extra instructions are added to every prompt. This will hopefully be improved in future updates.

* **Grammar File Overwrite:**
  If you use a custom grammar file in your WebUI, Audio Play will overwrite it with its own grammar file.

* **Missaligned Volume:**
  The volume of the generated audio may not be consistent across different speakers. This is a known issue and will be addressed in future updates.

* **Narrorator Voice:**
  The narrator voice (the voice used for non-dialogue text) is currently *not* excluded from the avilable voices for the speakers. This means that the narrator voice may be used for dialogue text as well.

---

## Feedback & Contributions

This is an experimental feature and your feedback is invaluable!