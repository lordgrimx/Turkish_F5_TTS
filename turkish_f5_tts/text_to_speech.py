import torch
import numpy as np
from .models.fastspeech2 import FastSpeech2
from .models.hifigan import HiFiGAN
from .text import text_to_sequence
from .utils.audio import AudioProcessor

class TextToSpeech:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.fastspeech = FastSpeech2().to(device)
        self.vocoder = HiFiGAN().to(device)
        self.audio_processor = AudioProcessor()
        
        if model_path:
            self._load_model(model_path)
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.fastspeech.load_state_dict(checkpoint['model'])
        
    def synthesize(self, text, output_path=None, speed_ratio=1.0):
        """
        Türkçe metni ses dosyasına dönüştürür.
        
        Args:
            text (str): Türkçe metin
            output_path (str, optional): Çıktı ses dosyasının yolu
            speed_ratio (float): Konuşma hızı çarpanı (1.0 = normal hız)
        
        Returns:
            numpy.ndarray: Ses dalga formu
        """
        # Metni fonem dizisine dönüştür
        phonemes = text_to_sequence(text)
        phoneme_tensor = torch.tensor(phonemes).unsqueeze(0).to(self.device)
        
        # FastSpeech2 ile mel-spektrogramı üret
        with torch.no_grad():
            mel_spec = self.fastspeech.inference(
                phoneme_tensor,
                speed_ratio=speed_ratio
            )
            
            # HiFiGAN ile ses dalgası üret
            waveform = self.vocoder.inference(mel_spec)
        
        # Numpy dizisine dönüştür
        waveform = waveform.squeeze().cpu().numpy()
        
        # Eğer çıktı yolu belirtildiyse kaydet
        if output_path:
            self.audio_processor.save_wav(waveform, output_path)
            
        return waveform
