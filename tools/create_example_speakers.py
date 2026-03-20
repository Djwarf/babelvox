#!/usr/bin/env python3
"""Create bundled example speaker profiles from reference audio.

Place audio files in tools/reference_audio/ and configure SPEAKERS below,
then run this script to extract embeddings and save profiles to
src/babelvox/data/speakers/.

Requires a working BabelVox installation with models downloaded.
"""

import os
import sys

# ── Configure example speakers here ──────────────────────────────────
# Each entry: (filename, transcript, language, gender, description, tags)
SPEAKERS = [
    ("steve.mp3",
     "By the late summer the news of what had happened on Animal Farm had spread across half the county.",
     "English", "male", "Male narrator, literary fiction", ["narrator", "example"]),
    ("lou.mp3",
     "Detective sergeant Bennett mentally notes the blanket, pillows on the couch the makeshift ashtray on the table and probably half a dozen other things I haven't even thought of.",
     "English", "female", "Female narrator, detective fiction", ["narrator", "example"]),
    ("cindy.mp3",
     "On the bleak greyness of a Cambridge spring day we set off in a cortege black car towards great Saint Mary's church.",
     "English", "female", "Female narrator, memoir", ["narrator", "example"]),
    ("phil.mp3",
     "The second half of the test was as easy as the first and he found he did a passable job on the essay kept it short.",
     "English", "male", "Male narrator, contemporary fiction", ["narrator", "example"]),
]
# ─────────────────────────────────────────────────────────────────────

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "reference_audio")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__),
                          "..", "src", "babelvox", "data", "speakers")


def main():
    if not SPEAKERS:
        print("No speakers configured. Edit SPEAKERS in this script first.")
        print("See tools/reference_audio/README.md for instructions.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from babelvox import BabelVox
    from babelvox.speakers import SpeakerLibrary

    print("Loading BabelVox (CPU, int8)...")
    tts = BabelVox(precision="int8", use_cp_kv_cache=True)
    tts.speaker_library = SpeakerLibrary(OUTPUT_DIR)

    for wav_name, ref_text, language, gender, description, tags in SPEAKERS:
        audio_path = os.path.join(AUDIO_DIR, wav_name)
        if not os.path.isfile(audio_path):
            print(f"  SKIP {wav_name} — file not found at {audio_path}")
            continue

        name = os.path.splitext(wav_name)[0].lower()
        print(f"  Extracting speaker '{name}' from {wav_name}...")

        profile = tts.save_speaker(
            name, audio_path,
            language=language, gender=gender,
            description=description, tags=tags,
        )
        print(f"  Saved {name}.json + {name}.npy "
              f"(embedding shape: {profile.embedding.shape})")

    print(f"\nDone. Profiles saved to {OUTPUT_DIR}")
    print("Commit the .json and .npy files to bundle them with the package.")


if __name__ == "__main__":
    main()
