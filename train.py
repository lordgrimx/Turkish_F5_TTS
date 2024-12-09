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
from torch import F

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
        
        # Duration ratio for initial duration targets
        self.duration_ratio = 256  # hop_length of mel transform
        
    def normalize_mel(self, mel):
        """Normalize mel-spectrogram"""
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return (mel - mel.mean()) / (mel.std() + 1e-8)
        
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
            
            # Normalize mel-spectrogram
            mel_spec = self.normalize_mel(mel_spec)
            
            # Metni fonemlere dönüştür
            phonemes = text_to_sequence(text)
            phonemes = torch.LongTensor(phonemes)
            
            # Calculate initial duration targets
            mel_len = mel_spec.size(0)
            phone_len = len(phonemes)
            duration = torch.zeros(phone_len)
            
            # Distribute mel frames evenly across phonemes
            frames_per_phone = mel_len / phone_len
            remaining_frames = mel_len
            
            for i in range(phone_len):
                if i == phone_len - 1:
                    # Last phoneme gets all remaining frames
                    duration[i] = remaining_frames
                else:
                    # Round to nearest integer, ensuring at least 1 frame per phoneme
                    dur = max(1, round(frames_per_phone))
                    duration[i] = min(dur, remaining_frames)
                    remaining_frames -= duration[i]
            
            return {
                'phonemes': phonemes,
                'durations': duration,
                'mel': mel_spec
            }
            
        except Exception as e:
            print(f"Hata oluştu - Dosya: {audio_path}, Metin: {text}")
            print(f"Hata: {str(e)}")
            return {
                'phonemes': torch.zeros(1, dtype=torch.long),
                'durations': torch.zeros(1),
                'mel': torch.zeros(1000, self.config.n_mel_channels)
            }

def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    # Filter out any None values (failed samples)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    # Get max lengths
    max_phoneme_len = max(x['phonemes'].size(0) for x in batch)
    max_mel_len = max(x['mel'].size(0) for x in batch)
    
    # Prepare tensors
    phonemes_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
    durations_padded = torch.zeros(len(batch), max_phoneme_len)
    mel_specs = torch.zeros(len(batch), max_mel_len, batch[0]['mel'].size(1))
    
    # Pad sequences
    for i, x in enumerate(batch):
        phoneme_len = x['phonemes'].size(0)
        mel_len = x['mel'].size(0)
        phonemes_padded[i, :phoneme_len] = x['phonemes']
        durations_padded[i, :phoneme_len] = x['durations']
        mel_specs[i, :mel_len] = x['mel']
    
    return {
        'phonemes': phonemes_padded,
        'durations': durations_padded,
        'mel': mel_specs
    }

def train_step(batch, model, optimizer, criterion, device):
    # Move batch to device
    phonemes = batch['phonemes'].to(device)
    durations = batch['durations'].to(device)
    mel_target = batch['mel'].to(device)
    
    # Create masks (if needed)
    src_mask = None  # Add proper mask creation if needed
    mel_mask = None  # Add proper mask creation if needed
    
    # Forward pass
    model_output = model(
        src_seq=phonemes,
        src_mask=src_mask,
        mel_mask=mel_mask,
        duration_target=durations,
        max_len=mel_target.size(1)
    )
    
    mel_pred = model_output['mel_pred']
    duration_pred = model_output['duration_pred']
    
    # Calculate losses
    mel_loss = criterion(mel_pred, mel_target)
    duration_loss = F.mse_loss(duration_pred, durations.float())
    
    # Total loss (you can adjust the weights)
    total_loss = mel_loss + duration_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'mel_loss': mel_loss.item(),
        'duration_loss': duration_loss.item()
    }

def train(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    
    for epoch in range(num_epochs):
        epoch_stats = {
            'total_loss': 0.0,
            'mel_loss': 0.0,
            'duration_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            step_stats = train_step(batch, model, optimizer, criterion, device)
            
            # Update epoch stats
            for k, v in step_stats.items():
                epoch_stats[k] += v
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - '
                      f'Loss: {step_stats["total_loss"]:.4f} '
                      f'(Mel: {step_stats["mel_loss"]:.4f}, Dur: {step_stats["duration_loss"]:.4f})')
        
        # Print epoch summary
        avg_total_loss = epoch_stats['total_loss'] / len(train_loader)
        avg_mel_loss = epoch_stats['mel_loss'] / len(train_loader)
        avg_duration_loss = epoch_stats['duration_loss'] / len(train_loader)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Total Loss: {avg_total_loss:.4f}')
        print(f'Average Mel Loss: {avg_mel_loss:.4f}')
        print(f'Average Duration Loss: {avg_duration_loss:.4f}\n')

def main():
    # Config
    config = ModelConfig()
    config.epochs = 10
    
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
        loss = train(model, train_loader, optimizer, criterion, 1, device)
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
