# Tesseract CNN OCR

Hybrid OCR system combining traditional optical character recognition with a compact convolutional neural recogniser trained on generated text images.


## Overview
- Synthetic character images are generated using PIL:
    - Grayscale normalisation: `x ∈ [0,1]`

- A shallow CNN learns `glyph-level` classification:
    - Convolution → ReLU → Pooling → Linear

- Classical OCR is attempted first:
    - `Tesseract(image) → text`

- CNN prediction is used as fallback or correction:
    - `ŷ = argmax(softmax(f(x)))`
  
- All data and inference remain fully local.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
python3 tesseract_cnn_ocr.py
```