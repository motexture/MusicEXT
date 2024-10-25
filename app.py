import gradio as gr
import numpy as np
import torch
from musicext.models import MusicGen

# Global variables
audio_clip = None  # Initialize empty buffer for audio clip
model = None  # Global model variable

# Initialize the MusicGen model only once at the start
def initialize_model():
    global model
    device = 'cuda'
    model = MusicGen.get_pretrained('facebook/musicgen-large', device=device)

# Function to generate initial music and store in the audio buffer
def generate_music(description, duration, top_k):
    global audio_clip, model

    model.set_generation_params(use_sampling=True, top_k=top_k, duration=duration)

    music = model.generate(descriptions=[description], progress=True)
    music = music.cpu().numpy()

    if music.dtype != np.float32:
        music = music.astype(np.float32)
        music /= 32768.0

    audio_clip = music.squeeze().squeeze()
    
    return model.sample_rate, audio_clip

# Function to extend the existing audio buffer with new generated music and crossfade
def extend_music(description, duration, conditioning_seconds, top_k, crossfade_duration):
    global audio_clip, model
    crossfade_duration = 500

    fade_out = np.linspace(1, 0, crossfade_duration)
    fade_in = np.linspace(0, 1, crossfade_duration)

    model.set_generation_params(use_sampling=True, top_k=top_k, duration=duration)

    input_audio_clip = np.expand_dims(audio_clip, axis=0)
    input_audio_clip = np.expand_dims(input_audio_clip, axis=0)

    tensor_clip = torch.tensor(input_audio_clip, dtype=torch.float32)

    music = model.generate_continuation(tensor_clip[:, :, -model.sample_rate * conditioning_seconds:], model.sample_rate, [description], progress=True)
    music = music.cpu().numpy()

    if music.dtype != np.float32:
        music = music.astype(np.float32)
        music /= 32768.0

    if len(input_audio_clip) >= crossfade_duration:
        input_audio_clip[-crossfade_duration:] = (input_audio_clip[-crossfade_duration:] * fade_out) + (music[:crossfade_duration] * fade_in)
        audio_clip = np.append(input_audio_clip, music[crossfade_duration:])
    else:
        audio_clip = np.append(input_audio_clip, music).squeeze().squeeze()

    return model.sample_rate, audio_clip

# Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("## Music Generation & Extension Interface")

    # Input parameters for music generation
    description_input = gr.Textbox(label="Music Description", value="Minimal, Techno, Tech House, 120 bpm")
    duration_input = gr.Slider(minimum=10, maximum=30, value=30, step=5, label="Duration of Initial Clip (Seconds)")
    top_k_input = gr.Slider(minimum=50, maximum=500, value=250, step=50, label="Top-K Sampling Value")

    # Generate music button
    generate_button = gr.Button("Generate Initial Music")
    audio_player = gr.Audio(label="Generated Music", type="numpy")  # Use numpy as the data type

    # Parameters for extending music
    conditioning_seconds_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Conditioning Duration (Seconds)")
    extension_duration_input = gr.Slider(minimum=10, maximum=30, value=30, step=5, label="Extension Duration (Seconds)")
    crossfade_duration_input = gr.Slider(minimum=100, maximum=1000, value=500, step=50, label="Crossfade Duration (Samples)")

    # Extend music button
    extend_button = gr.Button("Extend Music")

    # Initialize the MusicGen model before launching the app
    initialize_model()

    # Event handlers
    generate_button.click(
        fn=generate_music,
        inputs=[description_input, duration_input, top_k_input],
        outputs=[audio_player]
    )
    extend_button.click(
        fn=extend_music,
        inputs=[description_input, extension_duration_input, conditioning_seconds_input, top_k_input, crossfade_duration_input],
        outputs=[audio_player]
    )

# Launch the Gradio app
app.launch()
