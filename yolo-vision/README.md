# YOLO Vision

An object detection system using a YOLOv5-style architecture trained on generated images.


## Overview
- Synthetic images with rectangular objects are generated locally

- A compact `YOLOv5`-inspired convolutional network predicts bounding boxes

- Bounding box regression uses mean squared error:

    `L = (x̂ − x)² + (ŷ − y)² + (ŵ − w)² + (ĥ − h)²`

- Training runs instantly on CPU using small images

- Detection results are rendered directly in the browser 


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```