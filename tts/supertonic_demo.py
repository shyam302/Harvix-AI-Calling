import numpy as np
from supertonic import TTS

tts = TTS(auto_download=True)
style = tts.get_voice_style(voice_name="F3")

text = "आपका स्वागत है शिक्षा के दौरान आपको एक बार फिर से बात करने का मौका देंगे।"
wav, duration = tts.synthesize(text, voice_style=style, lang="hi")

tts.save_audio(wav, "output.wav")
dur_sec = float(np.asarray(duration).squeeze())
print(f"Generated {dur_sec:.2f}s of audio")
