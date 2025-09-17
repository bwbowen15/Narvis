import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import convertToSpectrogram



#RECORD
#sampling frequency
freq = 44100

#recording duration
duration = 5

#start recorder
recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)

#record for duration
sd.wait()


#SAVE
wv.write("./io/recording.wav", recording, freq, sampwidth=1)