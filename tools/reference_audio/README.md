# Reference Audio for Example Speakers

Place reference audio files here for creating bundled example speaker profiles.

## Requirements per speaker

- **Audio**: WAV file, 3-10 seconds of clear speech, 16kHz+ sample rate
- **Transcript**: Exact text spoken in the audio
- **Metadata**: Name, language, gender, description

## File naming

Name files as `{speaker_name}.wav`, e.g. `alice.wav`, `bob.wav`.

## Creating profiles

After placing audio files here, run:

```bash
python tools/create_example_speakers.py
```

This extracts speaker embeddings and saves profiles to `src/babelvox/data/speakers/`.
The .wav files are NOT shipped with the package — only the extracted embeddings.
