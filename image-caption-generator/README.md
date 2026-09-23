# Image Caption Generator

An image captioning system that trains a neural network on procedurally generated images and produces learned captions.


## Overview
- Synthetic images of simple objects are generated offline

- A `CNN` is trained to classify image content

- Classes are mapped to natural-language `caption` templates

- The trained model performs inference
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```