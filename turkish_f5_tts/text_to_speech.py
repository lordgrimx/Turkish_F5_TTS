import torch
import numpy as np
from .models.fastspeech2 import FastSpeech2
from .models.hifigan import HiFiGAN
from .text import text_to_sequence
from .utils.audio import AudioProcessor
from .utils.constants import ModelConfig

class TextToSpeech:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu', config=None):
        self.device = device
        self.config = config if config is not None else ModelConfig()
        
        # Initialize models
        self._init_models()
        
        # Load checkpoint if provided
        if model_path:
            self._load_model(model_path)
    
    def _init_models(self):
        """Initialize FastSpeech2 and HiFiGAN models"""
        self.fastspeech = FastSpeech2(self.config)
        self.fastspeech = self.fastspeech.to(self.device)
        self.vocoder = HiFiGAN()
        self.vocoder = self.vocoder.to(self.device)
        self.audio_processor = AudioProcessor()
    
    def _load_model(self, model_path):
        """Load model checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.fastspeech.load_state_dict(checkpoint['model_state_dict'])
        
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
