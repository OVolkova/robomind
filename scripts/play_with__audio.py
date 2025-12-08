# import torchaudio
#
# print(torchaudio.list_audio_backends())
# import soundfile as sf
# import io
#
# waveform, sample_rate = torchaudio.load(
#     "/Users/olly/Documents/projects/robodog/robodog/robomind/assets/exampleaudio.mp3"
# )
# print(waveform.shape, sample_rate)
# print(type(sample_rate))
#
# memory_file = io.BytesIO()
# sf.write(
#     memory_file, output_signal, output_frequency, format="wav"
# )
# memory_file.seek(0)  # Reset pointer to the beginning of the BytesIO object
#
# waveform, sample_rate = sf.read(memory_file, format="mp3")
# print(waveform.shape, sample_rate)
# print(type(sample_rate))