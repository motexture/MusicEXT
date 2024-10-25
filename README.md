# MusicEXT-1.0

Using MusicGen model from Facaebook, create infinite length music or play live music with smooth transitions. Both thanks to the model which allows for generating music with conditioning.

## Installation

```bash
git clone https://github.com/motexture/MusicEXT
cd MusicEXT
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python app.py  # For standard generation
# or
python live.py  # For live generation
```

## Info

To switch to a lighter model, replace the facebook/musicgen-large string with a different MusicGen model.

Note: Running live music generation with either the melody or large model requires an NVIDIA RTX 4090.

## License

The code in this repository is released under the MIT license as found in the LICENSE file.
The models weights in this repository are released under the CC-BY-NC 4.0 license as found in the LICENSE_weights file.
