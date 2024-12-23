import torch
import torch.nn as nn
import numpy as np
from .transformer import FFTBlock
from ..utils.constants import *
import torch.nn.functional as F

class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        
        self.input_size = model_config.encoder_hidden
        self.filter_size = model_config.variance_predictor_filter_size
        self.kernel = model_config.variance_predictor_kernel_size
        self.conv_output_size = model_config.variance_predictor_filter_size
        self.dropout = model_config.dropout
        
        self.conv_net = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.input_size if i == 0 else self.filter_size,
                    self.filter_size,
                    kernel_size=self.kernel,
                    padding=(self.kernel - 1) // 2
                ),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ) for i in range(2)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.InstanceNorm1d(self.filter_size)
            for _ in range(2)
        ])
        
        self.linear_layer = nn.Linear(self.conv_output_size, 1)
    
    def forward(self, encoder_output, mask=None):
        out = encoder_output.transpose(1, 2)  # [batch, channels, length]
        
        for conv, norm in zip(self.conv_net, self.norm_layers):
            out = conv(out)  # Apply conv and activation
            out = norm(out)  # Apply normalization
        
        out = out.transpose(1, 2)  # [batch, length, channels]
        out = self.linear_layer(out)  # [batch, length, 1]
        
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0)
        
        return out.squeeze(-1)  # [batch, length]

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self, x, duration_predictor_output, max_len=None):
        batch_size, seq_len, hidden_size = x.size()
        
        # Calculate expanded lengths for each sequence in the batch
        expanded_lens = torch.sum(duration_predictor_output, dim=1)  # [batch_size]
        
        # If max_len is provided, use it; otherwise use the maximum expanded length
        if max_len is not None:
            max_output_len = max_len
        else:
            max_output_len = int(expanded_lens.max().item())
        
        # Initialize output tensor
        output = torch.zeros(batch_size, max_output_len, hidden_size).to(x.device)
        
        # Process each sequence in the batch
        for i in range(batch_size):
            current_seq = x[i]
            current_durations = duration_predictor_output[i]
            
            pos = 0
            for j, duration in enumerate(current_durations):
                duration = int(duration.item())
                if duration > 0:
                    # Create the expanded features correctly
                    expanded = current_seq[j:j+1].expand(duration, -1)
                    # Make sure we don't exceed the output length
                    if pos + duration <= max_output_len:
                        output[i, pos:pos + duration] = expanded
                    else:
                        # If we would exceed max_output_len, only take what fits
                        remaining = max_output_len - pos
                        if remaining > 0:
                            output[i, pos:max_output_len] = expanded[:remaining]
                        break
                    pos += duration
        
        return output

class FastSpeech2(nn.Module):
    def __init__(self, model_config):
        super(FastSpeech2, self).__init__()
        
        # Add phoneme embedding layer
        self.phoneme_embedding = nn.Embedding(
            model_config.vocab_size,  # Size of phoneme vocabulary
            model_config.encoder_hidden
        )
        
        self.encoder = nn.ModuleList([
            FFTBlock(
                model_config.encoder_hidden,
                model_config.encoder_conv1d_filter_size,
                model_config.encoder_head,
                model_config.encoder_hidden // model_config.encoder_head,
                model_config.encoder_hidden // model_config.encoder_head,
                model_config.encoder_conv1d_kernel_size,
                model_config.encoder_conv1d_padding,
                model_config.dropout
            ) for _ in range(model_config.encoder_n_layer)
        ])
        
        self.variance_adaptor = nn.ModuleList([
            VariancePredictor(model_config) for _ in range(3)  # Duration, Pitch, Energy
        ])
        
        # Add length regulator
        self.length_regulator = LengthRegulator()
        
        self.decoder = nn.ModuleList([
            FFTBlock(
                model_config.decoder_hidden,
                model_config.decoder_conv1d_filter_size,
                model_config.decoder_head,
                model_config.decoder_hidden // model_config.decoder_head,
                model_config.decoder_hidden // model_config.decoder_head,
                model_config.decoder_conv1d_kernel_size,
                model_config.decoder_conv1d_padding,
                model_config.dropout
            ) for _ in range(model_config.decoder_n_layer)
        ])
        
        self.mel_linear = nn.Linear(
            model_config.decoder_hidden,
            model_config.n_mel_channels
        )

    def normalize_mel(self, mel):
        """Normalize mel-spectrogram output"""
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return (mel - mel.mean()) / (mel.std() + 1e-8)

    def forward(self, src_seq, src_mask=None, mel_mask=None, duration_target=None,
                pitch_target=None, energy_target=None, max_len=None):
        
        # Encoder
        encoder_output = self.phoneme_embedding(src_seq)
        for encoder_layer in self.encoder:
            encoder_output = encoder_layer(encoder_output, src_mask)
            
        # Variance Adaptor
        duration_pred = self.variance_adaptor[0](encoder_output, src_mask)
        
        # Print shapes for debugging
        print(f"Encoder output shape: {encoder_output.shape}")
        print(f"Initial duration pred shape: {duration_pred.shape}")
        
        # Ensure duration prediction matches input sequence length
        if duration_pred.size(1) != src_seq.size(1):
            print(f"Adjusting duration pred from {duration_pred.size(1)} to {src_seq.size(1)}")
            if duration_pred.size(1) < src_seq.size(1):
                duration_pred = F.pad(duration_pred, (0, src_seq.size(1) - duration_pred.size(1)))
            else:
                duration_pred = duration_pred[:, :src_seq.size(1)]
        
        pitch_pred = self.variance_adaptor[1](encoder_output, src_mask)
        energy_pred = self.variance_adaptor[2](encoder_output, src_mask)
        
        # Length Regulator
        if duration_target is not None:
            output = self.length_regulator(encoder_output, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(duration_pred), min=0)
            output = self.length_regulator(encoder_output, duration_rounded, max_len)
        
        # Decoder
        for decoder_layer in self.decoder:
            output = decoder_layer(output, mel_mask)
            
        # Project to mel-spectrogram and normalize
        mel_pred = self.mel_linear(output)
        mel_pred = self.normalize_mel(mel_pred)
        
        return {
            'mel_pred': mel_pred,
            'duration_pred': duration_pred,
            'pitch_pred': pitch_pred,
            'energy_pred': energy_pred
        }
        
    def inference(self, src_seq, speed_ratio=1.0):
        src_mask = torch.ones(1, src_seq.size(1)).bool()
        
        # Encoder
        encoder_output = self.phoneme_embedding(src_seq)
        for encoder_layer in self.encoder:
            encoder_output = encoder_layer(encoder_output, src_mask)
            
        # Variance Prediction
        duration_pred = self.variance_adaptor[0](encoder_output, src_mask)
        pitch_pred = self.variance_adaptor[1](encoder_output, src_mask)
        energy_pred = self.variance_adaptor[2](encoder_output, src_mask)
        
        # Adjust duration for speed
        duration_rounded = torch.clamp(torch.round(duration_pred / speed_ratio), min=0)
        
        # Length Regulation
        output = self.length_regulator(encoder_output, duration_rounded)
        
        # Decoder
        for decoder_layer in self.decoder:
            output = decoder_layer(output)
            
        # Project to mel-spectrogram
        mel_pred = self.mel_linear(output)
        mel_pred = self.normalize_mel(mel_pred)
        
        return mel_pred
