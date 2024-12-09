import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from turkish_f5_tts.models.fastspeech2 import FastSpeech2
from turkish_f5_tts.models.hifigan import HiFiGAN
from turkish_f5_tts.utils.constants import ModelConfig
from turkish_f5_tts.utils.audio import AudioProcessor
from turkish_f5_tts.text import text_to_sequence
import numpy as np
from tqdm import tqdm
import logging
import torchaudio

class TurkishTTSDataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.audio_processor = AudioProcessor()
        self.load_metadata()
        
    def load_metadata(self):
        """Veri seti metadatasını yükle"""
        self.metadata = []
        metadata_path = os.path.join(self.data_path, 'metadata.csv')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.csv dosyası bulunamadı: {metadata_path}")
            
        logging.info(f"Metadata dosyası yükleniyor: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logging.info(f"Toplam {len(lines)} satır okundu")
            
            for i, line in enumerate(lines):
                parts = line.strip().split('|')
                if len(parts) == 2:
                    audio_path = os.path.join(self.data_path, parts[0])
                    text = parts[1]
                    if os.path.exists(audio_path):
                        self.metadata.append((audio_path, text))
                    else:
                        logging.warning(f"Ses dosyası bulunamadı: {audio_path}")
                else:
                    logging.warning(f"Geçersiz metadata formatı, satır {i+1}: {line.strip()}")
        
        logging.info(f"Toplam {len(self.metadata)} geçerli örnek bulundu")
        if len(self.metadata) == 0:
            raise ValueError("Hiç geçerli örnek bulunamadı! Lütfen veri seti yapısını kontrol edin.")
                        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        """Veri setinden bir örnek döndür"""
        audio_path, text = self.metadata[idx]
        
        # Ses dosyasını yükle
        waveform, sr = torchaudio.load(audio_path)
        
        # Tek kanala dönüştür
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Örnekleme hızını kontrol et
        if sr != self.config.sampling_rate:
            waveform = torchaudio.transforms.Resample(sr, self.config.sampling_rate)(waveform)
            
        # Ses uzunluğunu kontrol et
        if waveform.size(1) > self.config.max_wav_length:
            waveform = waveform[:, :self.config.max_wav_length]
            
        # Metni fonemlere dönüştür
        try:
            phonemes = text_to_sequence(text)
            phonemes = torch.LongTensor(phonemes)
        except Exception as e:
            print(f"Metin dönüştürme hatası: {text}")
            print(f"Hata: {str(e)}")
            # Hata durumunda boş bir örnek döndür
            return {
                'waveform': torch.zeros(1, self.config.max_wav_length),
                'phonemes': torch.zeros(1, dtype=torch.long)
            }
            
        return {
            'waveform': waveform.squeeze(0),  # [T]
            'phonemes': phonemes  # [L]
        }

def train(config, model, train_loader, optimizer, criterion, device, epoch):
    """Bir epoch için eğitim"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Batch verilerini device'a taşı
        waveform = batch['waveform'].to(device)
        phonemes = batch['phonemes'].to(device)
        
        # Forward pass
        output = model(phonemes)
        
        # Loss hesapla
        loss = criterion(output['mel_pred'], waveform)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_thresh)
        
        # Optimize
        optimizer.step()
        
        # Loss güncelle
        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        
        # Progress bar güncelle
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
    
    return total_loss / len(train_loader)

def main():
    # Config
    config = ModelConfig()
    
    # Logging ayarla
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Dataset yolunu kontrol et
    dataset_path = './dataset'
    dataset_path = os.path.abspath(dataset_path)
    logger.info(f'Dataset yolu: {dataset_path}')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset dizini bulunamadı: {dataset_path}')
    
    # Model
    model = FastSpeech2(config).to(device)
    vocoder = HiFiGAN().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    
    # Loss
    criterion = nn.MSELoss()
    
    # Dataset ve DataLoader
    dataset = TurkishTTSDataset(dataset_path, config)
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f'Dataset size: {len(dataset)} samples')
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.epochs):
        loss = train(config, model, train_loader, optimizer, criterion, device, epoch)
        logger.info(f'Epoch {epoch+1}, Loss: {loss:.4f}')
        
        # Her 10 epoch'ta bir checkpoint kaydet
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            
            # Google Drive'a kaydet (eğer Colab'da çalışıyorsa)
            if os.environ.get('COLAB_GPU'):
                try:
                    drive_path = '/content/drive/MyDrive/turkish_f5_tts_checkpoints'
                    os.makedirs(drive_path, exist_ok=True)
                    drive_checkpoint_path = os.path.join(drive_path, checkpoint_path)
                    os.system(f'cp {checkpoint_path} {drive_checkpoint_path}')
                    logger.info(f'Checkpoint saved to Google Drive: {drive_checkpoint_path}')
                except Exception as e:
                    logger.warning(f'Could not save to Google Drive: {str(e)}')
        
        # En iyi modeli kaydet
        if loss < best_loss:
            best_loss = loss
            best_model_path = 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, best_model_path)
            
            # En iyi modeli Google Drive'a kaydet
            if os.environ.get('COLAB_GPU'):
                try:
                    drive_path = '/content/drive/MyDrive/turkish_f5_tts_checkpoints'
                    os.makedirs(drive_path, exist_ok=True)
                    drive_best_model_path = os.path.join(drive_path, 'best_model.pt')
                    os.system(f'cp {best_model_path} {drive_best_model_path}')
                    logger.info(f'Best model saved to Google Drive: {drive_best_model_path}')
                except Exception as e:
                    logger.warning(f'Could not save best model to Google Drive: {str(e)}')

if __name__ == '__main__':
    main()
