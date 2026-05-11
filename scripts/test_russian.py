import torch
import torchaudio
import os

text = "Интерес к психологии детской игры возник у меня в самом начале 1930-х годов в ходе наблюдений над игрой дочурок и в связи с чтением лекций по детской психологии. Записи этих наблюдений затерялись во время войны в блокированном Ленинграде, и в памяти остались лишь некоторые эпизоды. Вот два из них."

language = 'ru'
model_id = 'v3_1_ru'
# speaker = 'xenia'  # or 'aidar', 'baya', 'kseniya', 'eugene'
speaker = 'eugene'

model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                          model='silero_tts',
                          language=language,
                          speaker=model_id)

audio = model.apply_tts(text=text,
                        speaker=speaker,
                        sample_rate=48000)

# Save the audio to a file
output_file = "output_russian.wav"
torchaudio.save(output_file, audio.unsqueeze(0), 48000)

# Play the audio file
os.system(f"afplay {output_file}")