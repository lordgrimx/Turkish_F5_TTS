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

def pad_sequence(waveform, target_length):
    """Ses dosyasını hedef uzunluğa getir"""
    current_length = waveform.size(-1)
    if current_length > target_length:
        return waveform[..., :target_length]
    else:
        padding = torch.zeros((*waveform.shape[:-1], target_length - current_length))
        return torch.cat([waveform, padding], dim=-1)

class TurkishTTSDataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.target_length = int(config.max_seq_len * config.sample_rate / 1000)  # ms to samples
        self.audio_processor = AudioProcessor()
        self.metadata = self.load_metadata()
        
        # Add mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            n_mels=config.n_mel_channels,
            f_min=config.mel_fmin,
            f_max=config.mel_fmax
        )
        
    def load_metadata(self):
        """Metadata dosyasını yükle"""
        metadata_path = os.path.join(self.data_path, 'metadata.csv')
        logging.info(f"Metadata dosyası yükleniyor: {metadata_path}")
        
        metadata = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'audio_path' in line:  # Başlık satırını atla
                    continue
                    
                parts = line.strip().split('|')
                if len(parts) == 2:
                    audio_path, text = parts
                    
                    # Tam yolu oluştur
                    full_audio_path = os.path.join(self.data_path, audio_path)
                    
                    # Ses dosyasının varlığını kontrol et
                    if not os.path.exists(full_audio_path):
                        logging.warning(f"Ses dosyası bulunamadı: {full_audio_path}")
                        continue
                        
                    metadata.append((full_audio_path, text))
                else:
                    logging.warning(f"Geçersiz metadata formatı, satır {len(metadata)}: {line.strip()}")
                    
        logging.info(f"Toplam {len(metadata)} geçerli örnek bulundu")
        return metadata
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        """Veri setinden bir örnek döndür"""
        audio_path, text = self.metadata[idx]
        
        try:
            # Ses dosyasını yükle
            waveform, sr = torchaudio.load(audio_path)
            
            # Tek kanala dönüştür
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Örnekleme hızını kontrol et
            if sr != self.config.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.config.sample_rate)(waveform)
                
            # Ses uzunluğunu ayarla
            waveform = pad_sequence(waveform, self.target_length)
            
            # Convert to mel-spectrogram
            mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)  # [time, n_mels]
            
            # Metni fonemlere dönüştür
            phonemes = text_to_sequence(text)
            phonemes = torch.LongTensor(phonemes)
            
            return {
                'waveform': mel_spec,  # Now returning mel-spectrogram instead of waveform
                'phonemes': phonemes
            }
            
        except Exception as e:
            print(f"Hata oluştu - Dosya: {audio_path}, Metin: {text}")
            print(f"Hata: {str(e)}")
            # Hata durumunda boş bir örnek döndür
            return {
                'waveform': torch.zeros(1000, self.config.n_mel_channels),  # Adjusted for mel-spec shape
                'phonemes': torch.zeros(1, dtype=torch.long)
            }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Filter out any None values (failed samples)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Get max lengths
    max_phoneme_len = max(x['phonemes'].size(0) for x in batch)
    max_mel_len = max(x['waveform'].size(0) for x in batch)
    
    # Prepare tensors
    mel_specs = torch.zeros(len(batch), max_mel_len, batch[0]['waveform'].size(1))
    phonemes_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    
    # Pad sequences
    for i, x in enumerate(batch):
        mel_len = x['waveform'].size(0)
        phoneme_len = x['phonemes'].size(0)
        mel_specs[i, :mel_len] = x['waveform']
        phonemes_padded[i, :phoneme_len] = x['phonemes']
    
    return {
        'waveform': mel_specs,  # This is actually mel-spectrogram now
        'phonemes': phonemes_padded
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
        pin_memory=True,
        collate_fn=collate_fn
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
