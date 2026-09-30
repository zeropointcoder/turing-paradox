# Image Style Transfer

A neural image transformation project that merges structural content with stylistic texture using convolutional feature statistics.


## Overview
- A convolutional neural network extracts feature maps from images

- `Content loss` preserves spatial structure:
  
    `‖F_target − F_content‖²`

- `Style loss` matches texture using `Gram` matrices:

    `G = F × Fᵀ`

- Optimisation updates the target image directly to minimise combined loss

- Entire pipeline runs locally with no pretrained weights or downloads
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Style is applied, but too weakly
### Random, shallow networks preserve content by default
### Increase style weight or start from noise to see visible change
```