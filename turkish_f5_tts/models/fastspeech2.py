import torch
import torch.nn as nn
import numpy as np
from .transformer import FFTBlock
from ..utils.constants import *

class VariancePredictor(nn.Module):
    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()
        
        self.input_size = model_config.encoder_hidden
        self.filter_size = model_config.variance_predictor_filter_size
        self.kernel = model_config.variance_predictor_kernel_size
        self.conv_output_size = model_config.variance_predictor_filter_size
        self.dropout = model_config.variance_predictor_dropout
        
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=(self.kernel-1)//2
            ),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            nn.ReLU(),
            nn.LayerNorm(self.filter_size),
            nn.Dropout(self.dropout)
        )
        
        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask=None):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
            
        return out

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

    def forward(self, src_seq, src_mask=None, mel_mask=None, duration_target=None,
                pitch_target=None, energy_target=None, max_len=None):
        
        # Encoder
        encoder_output = self.phoneme_embedding(src_seq)
        for encoder_layer in self.encoder:
            encoder_output = encoder_layer(encoder_output, src_mask)
            
        # Variance Adaptor
        duration_pred = self.variance_adaptor[0](encoder_output, src_mask)
        pitch_pred = self.variance_adaptor[1](encoder_output, src_mask)
        energy_pred = self.variance_adaptor[2](encoder_output, src_mask)
        
        # Length Regulator
        if duration_target is not None:
            output = self.length_regulator(encoder_output, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(duration_pred) - 1)), min=0
            )
            output = self.length_regulator(encoder_output, duration_rounded, max_len)
            
        # Decoder
        for decoder_layer in self.decoder:
            output = decoder_layer(output, mel_mask)
            
        # Mel Linear
        output = self.mel_linear(output)
        
        return {
            "mel_pred": output,
            "duration_pred": duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred
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
        duration_rounded = torch.clamp(
            (torch.round(torch.exp(duration_pred) - 1) / speed_ratio), min=0
        )
        
        # Length Regulation
        output = self.length_regulator(encoder_output, duration_rounded)
        
        # Decoder
        for decoder_layer in self.decoder:
            output = decoder_layer(output)
            
        # Mel Linear
        output = self.mel_linear(output)
        
        return output
