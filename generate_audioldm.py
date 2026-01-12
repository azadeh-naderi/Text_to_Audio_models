

pip install --upgrade diffusers transformers accelerate

from diffusers import AudioLDMPipeline
import torch
import scipy
from IPython.display import Audio
import soundfile as sf

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


prompt = "sound of dog"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

Audio(audio, rate=16000)

#scipy.io.wavfile.write("dog_index_1.wav", rate=16000, data=audio)

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generating 40 audio files for dog barking
num_audio_files = 40
for i in range(num_audio_files):
    prompt = "sound of fireworks"
    audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

    # Saving each audio file with a unique name
    file_name = f"fireworks_index_{i + 1}.wav"  # Naming each file uniquely

    scipy.io.wavfile.write(f"fireworks_index_{i + 1}.wav", rate=16000, data=audio)



from google.colab import files

#num_audio_files = 40
file_paths = [f"/content/fireworks_index_{i}.wav" for i in range(1, num_audio_files + 1)]

# Download each file
for file_path in file_paths:
    files.download(file_path)

from google.colab import files

num_audio_files = 40
file_paths = [f"/content/crackling_fire_index_{i}.wav" for i in range(1, num_audio_files + 1)]

# Download each file
for file_path in file_paths:
    files.download(file_path)
