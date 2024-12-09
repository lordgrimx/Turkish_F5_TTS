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
import wandb

class TurkishTTSDataset(Dataset):
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.audio_processor = AudioProcessor()
        self.load_metadata()
        
    def load_metadata(self):
        """Veri seti metadatasını yükle"""
        self.metadata = []
        with open(os.path.join(self.data_path, 'metadata.csv'), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    audio_path = os.path.join(self.data_path, 'wavs', parts[0])
                    text = parts[1]
                    if os.path.exists(audio_path):
                        self.metadata.append((audio_path, text))
                        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        audio_path, text = self.metadata[idx]
        
        # Metni fonem dizisine dönüştür
        phonemes = text_to_sequence(text)
        
        # Ses dosyasını yükle ve mel spektrogramını hesapla
        wav = self.audio_processor.load_wav(audio_path)
        mel = self.audio_processor.mel_spectrogram(wav)
        
        return {
            'text': torch.LongTensor(phonemes),
            'mel': mel,
            'duration': torch.ones(len(phonemes))  # Placeholder for duration
        }

def train(config, model, train_loader, optimizer, criterion, device, epoch):
    """Bir epoch için eğitim"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad()
        
        # Batch verilerini device'a taşı
        text = batch['text'].to(device)
        mel = batch['mel'].to(device)
        duration = batch['duration'].to(device)
        
        # Forward pass
        output = model(text, duration_target=duration)
        
        # Loss hesapla
        loss = criterion(output['mel_pred'], mel)
        
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
        
        # Wandb'ye log gönder
        if wandb.run is not None:
            wandb.log({
                'batch_loss': loss.item(),
                'epoch_loss': current_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    return total_loss / len(train_loader)

def main():
    # Config
    config = ModelConfig()
    
    # Logging ayarla
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Wandb başlat
    if os.environ.get('COLAB_GPU'):  # Google Colab'da çalışıyorsa
        wandb.init(project="turkish-f5-tts", name="training_run")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
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
    dataset = TurkishTTSDataset('path/to/your/dataset', config)
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
