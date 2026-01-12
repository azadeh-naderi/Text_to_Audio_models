import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio


!pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft

from audiocraft.models import AudioGen

model = AudioGen.get_pretrained('facebook/audiogen-medium')

# Line 97 of https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/models/audiogen.py


model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=5
)



def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=16000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

# Here we use a synthetic signal to prompt the generated audio.
res = model.generate_continuation(
    get_bip_bip(0.125).expand(2, -1, -1),
    16000, ['Whistling with wind blowing',
            'Typing on a typewriter'],
    progress=True)
display_audio(res, 16000)

# You can also use any audio from a file. Make sure to trim the file if it is too long!
prompt_waveform, prompt_sr = torchaudio.load("../assets/sirens_and_a_humming_engine_approach_and_pass.mp3")
prompt_duration = 2
prompt_waveform = prompt_waveform[..., :int(prompt_duration * prompt_sr)]
output = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=True)
display_audio(output, sample_rate=16000)

from audiocraft.utils.notebook import display_audio

output = model.generate(
    descriptions=[
        'Subway train blowing its horn',
        'A cat meowing',
    ],
    progress=True
)

display_audio(output, sample_rate=16000)

from audiocraft.utils.notebook import display_audio

output = model.generate(
    descriptions=[
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
        'Subway train blowing its horn',
    ],
    progress=True
)

display_audio(output, sample_rate=16000)

output = model.generate(
    descriptions=
        ['sound of hand saw' for _ in range(40)] , progress=True)
#display_audio(output, sample_rate=16000)

import IPython.display as ipd

index = 1
for audio_tensor in output:
    # Move the tensor to CPU and convert it to a NumPy array
    audio_np = audio_tensor.cpu().numpy()

    myaudioobject = ipd.Audio(audio_np, rate=16000)
    with open(f'hand_saw_index_{index}.wav', 'wb') as f:
        f.write(myaudioobject.data)
    index += 1

from google.colab import files

num_audio_files = 40
file_paths = [f"/content/hand_saw_index_{i}.wav" for i in range(1, num_audio_files + 1)]

# Download each file
for file_path in file_paths:
    files.download(file_path)
