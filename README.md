# Turkish F5-TTS

Bu proje, F5-TTS modelinin Türkçe dili için özelleştirilmiş bir versiyonudur. Uzun metinleri (30+ dakika) yüksek kaliteli Türkçe ses dosyalarına dönüştürmek için tasarlanmıştır.

## Özellikler

- Sadece Türkçe dil desteği
- Uzun metin girişi desteği (30+ dakika)
- Yüksek kaliteli ses çıktısı
- FastSpeech 2 tabanlı mimari

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Türkçe ses veri setini indirin ve hazırlayın
3. Modeli eğitin veya önceden eğitilmiş modeli kullanın

## Kullanım

```python
from turkish_f5_tts import TextToSpeech

# TTS modelini başlat
tts = TextToSpeech()

# Metinden ses üret
tts.synthesize("Türkçe metin buraya", output_path="output.wav")
```
