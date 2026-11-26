# Urban Scene Segmentation

Automatically segments different regions in synthetic urban scenes: Sky, Buildings, and Road.


## Overview
- **Synthetic Scene Generation**: Produces a small image with `sky`, `buildings`, and `road`.

- **Model**: Simple convolutional encoder-decoder `(SegNet-style)`.
    - Encoder: 
        $$
        x_{enc} = \text{ReLU}(\text{Conv}(x))
        $$
    - Pooling: Reduces `spatial` dimensions.
    - Decoder: Upsamples and predicts `per-pixel` class:
        $$
        x_{dec} = \text{ConvTranspose}(x_{pooled})
        $$

- **Segmentation**: `Argmax` over predicted channels assigns class labels.

- **Colours**: Each class mapped to `RGB` for visualisation.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```