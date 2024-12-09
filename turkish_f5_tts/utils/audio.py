import torch
import numpy as np
import librosa
import soundfile as sf

class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mel_channels=80,
                 mel_fmin=0.0, mel_fmax=8000.0):
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        
        # Mel filtre bankası oluştur
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=1024,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
    
    def load_wav(self, path):
        """Ses dosyasını yükle"""
        wav, _ = librosa.load(path, sr=self.sample_rate)
        return wav
    
    def save_wav(self, wav, path):
        """Ses dosyasını kaydet"""
        sf.write(path, wav, self.sample_rate)
    
    def mel_spectrogram(self, wav):
        """Ses dalgasından mel spektrogramı hesapla"""
        # STFT hesapla
        D = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024)
        # Genlik spektrogramı
        S = np.abs(D)
        # Mel spektrogramı
        mel = np.dot(self.mel_basis, S)
        # Log-mel spektrogramı
        mel = np.log10(np.maximum(mel, 1e-5))
        return torch.FloatTensor(mel)
    
    def normalize(self, wav):
        """Ses dalgasını normalize et"""
        return wav / np.max(np.abs(wav))
    
    def trim_silence(self, wav, threshold_db=-40):
        """Sessiz kısımları kırp"""
        return librosa.effects.trim(wav, top_db=threshold_db)[0]
