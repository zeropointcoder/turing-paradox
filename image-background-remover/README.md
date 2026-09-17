# Image Background Remover

An application that performs background removal using a deep-learning image segmentation model (U-Net) trained on labeled data.


## Overview
- The user uploads an image.

- The image is prepared and passed to a deep-learning segmentation model.

- The model predicts which pixels belong to the foreground.

- A binary mask is created from the prediction.

- The mask is applied to remove the background.

- The final image is displayed and can be downloaded.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### A pretrained U-Net model is used for background removal.
### Due to domain mismatch and lack of task-specific fine-tuning, results vary across images.
### The project focuses on ML-based segmentation deployment and evaluation rather than heuristic CV methods.
```