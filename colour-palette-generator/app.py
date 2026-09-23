import streamlit as st
from PIL import Image 
import numpy as np
from sklearn.cluster import KMeans


class PaletteGenerator:
    def __init__(self, n_colors=5):
        self.n_colors = n_colors
        self.model = KMeans(n_clusters=n_colors)

    def generate_palette(self, image: Image.Image):
        img = image.resize((150, 150))
        img_array = np.array(img)
        pixels = img_array.reshape(-1, 3)
        self.model.fit(pixels)
        colours = self.model.cluster_centers_.astype(int)
        return colours


class PaletteDisplay:
    @staticmethod
    def show_palette(colours):
        st.write("### Extracted Colour Palette")
        cols = st.columns(len(colours))
        for idx, colour in enumerate(colours):
            hex_colour = '#%02x%02x%02x' % tuple(colour)
            cols[idx].markdown(f'<div style="background-color:{hex_colour};height:100px"></div>', unsafe_allow_html=True)


def main():
    st.title("Image Colour Palette Generator")
    st.write("Upload an image to extract its dominant colour palette.")

    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=400)

        n_colors = st.slider("Number of colours", 2, 10, 5)
        generator = PaletteGenerator(n_colors)
        colours = generator.generate_palette(image)
        PaletteDisplay.show_palette(colours)


if __name__ == "__main__":
    main()