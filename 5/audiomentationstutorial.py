import librosa
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter

from helper import _plot_signal_and_augmented_signal

# install pydub for using HighPassFilter
# install audiomentations

# Raw audio augmentation
augment_raw_audio = Compose(
    [
        AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1),
        PitchShift(min_semitones=-8, max_semitones=8, p=1),
        HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=4000, p=1)
    ]
)

if __name__ == "__main__":
    signal, sr = librosa.load("scale.wav")
    augmented_signal = augment_raw_audio(signal, sr)
    sf.write("augmented_audio.wav", augmented_signal, sr)
    _plot_signal_and_augmented_signal(signal, augmented_signal, sr)