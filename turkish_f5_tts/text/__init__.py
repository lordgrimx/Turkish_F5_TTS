import re
from phonemizer import phonemize

# Türkçe fonem seti
_pad = '_'
_punctuation = ';:,.!?¡¿—…"«»"" '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÇçĞğİıÖöŞşÜü'
_silences = ['sp', 'spn', 'sil']

# Sembolleri birleştir
symbols = [_pad] + list(_punctuation) + list(_letters) + _silences

# Symbol to id ve id to symbol dönüşüm sözlükleri
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text):
    """Metni sayı dizisine dönüştürür"""
    # Metni fonemlere dönüştür
    phonemes = phonemize(
        text,
        language='tr',
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=False
    )
    
    # Fonemleri ID'lere dönüştür
    sequence = []
    for symbol in phonemes:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
    
    return sequence

def sequence_to_text(sequence):
    """Sayı dizisini metne dönüştürür"""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result.append(_id_to_symbol[symbol_id])
    return ''.join(result)
