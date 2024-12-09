class ModelConfig:
    # Encoder parameters
    encoder_hidden = 256
    encoder_head = 2
    encoder_n_layer = 4
    encoder_conv1d_filter_size = 1024
    encoder_conv1d_kernel_size = 9
    encoder_conv1d_padding = 4
    vocab_size = 256  # Size of phoneme vocabulary
    
    # Decoder parameters
    decoder_hidden = 256
    decoder_head = 2
    decoder_n_layer = 4
    decoder_conv1d_filter_size = 1024
    decoder_conv1d_kernel_size = 9
    decoder_conv1d_padding = 4
    
    # Variance predictor
    variance_predictor_filter_size = 256
    variance_predictor_kernel_size = 3
    variance_predictor_dropout = 0.5
    
    # Other
    dropout = 0.1
    n_mel_channels = 80
    max_seq_len = 3000  # Uzun metinler için artırıldı
    
    # Training
    batch_size = 16
    epochs = 1000
    learning_rate = 0.001
    weight_decay = 0.0001
    grad_clip_thresh = 1.0
    
    # Audio parameters
    sample_rate = 22050
    mel_fmin = 0.0
    mel_fmax = 8000.0
