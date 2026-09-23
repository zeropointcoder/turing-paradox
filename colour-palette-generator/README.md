# Image Colour Palette Generator

Extracts `dominant` colour palettes from images for design and creative purposes.


## Overview
- User `uploads` an image via the Streamlit interface.

- The image is resized to `150x150` pixels for fast processing.

- Pixel data is reshaped into a `2D` array of `RGB` values.

- KMeans clustering identifies the most dominant colours:
  - Formula: `KMeans` minimises sum of squared distances:
    $$ \sum_{i=1}^{n} \sum_{j=1}^{k} ||x_i - \mu_j||^2 $$

    where \(x_i\) is a pixel and \(mu_j\) is cluster centre.

- `Dominant` colours are displayed as `swatches` in the browser.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```