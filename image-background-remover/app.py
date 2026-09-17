import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp


# Model Loader
class SegmentationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict_mask(self, image: Image.Image):
        original_size = image.size

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        pred = self.model(tensor)

        mask = torch.sigmoid(pred)[0, 0].cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)

        mask = Image.fromarray(mask * 255).resize(original_size)
        return np.array(mask) // 255

# Image Utilities
class ImageProcessor:
    @staticmethod
    def remove_background(image_np, mask):
        result = image_np.copy()
        result[mask == 0] = [0, 0, 0]
        return result


# Streamlit App
class BackgroundRemoverApp:
    def __init__(self):
        self.model = SegmentationModel()
        self.processor = ImageProcessor()

    def run(self):
        st.title("ML Background Remover")
        st.write("Deep-learning-based image segmentation using U-net.")

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"]
        )

        if not uploaded_file:
            return

        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.image(image, caption="Original Image")

        if st.button("Remove Background"):
            mask = self.model.predict_mask(image)
            result = self.processor.remove_background(image_np, mask)

            st.image(result, caption="Background Removed")

            output = Image.fromarray(result)
            st.download_button(
                "Download PNG",
                data=output.tobytes(),
                file_name="background_removed.png",
                mime="image/png"
            )

if __name__ == "__main__":
    BackgroundRemoverApp().run()