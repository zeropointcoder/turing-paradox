import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from streamlit_drawable_canvas import st_canvas


# Dataset class
class SketchDataset:
    def __init__(self, num_samples=500, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = [self.generate_sample() for _ in range(num_samples)]

    def generate_sample(self):
        img = Image.new('RGB', (self.img_size, self.img_size), 'white')
        draw = ImageDraw.Draw(img)
        for _ in range(np.random.randint(1,4)):
            x0, y0 = np.random.randint(0,32,2)
            x1, y1 = x0 + np.random.randint(10,32), y0 + np.random.randint(10,32)
            colour = tuple(np.random.randint(0, 255, 3))
            draw.rectangle([x0,y0,x1,y1], fill=colour)
        sketch = img.convert('L')
        return np.array(sketch)/255.0, np.array(img)/255.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sketch, colour = self.data[idx]
        return torch.tensor(sketch, dtype=torch.float32).unsqueeze(0), \
                torch.tensor(colour, dtype=torch.float32).permute(2,0,1)


# CNN Model
class ColouriserModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU()  # added extra layer
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1), nn.ReLU(),
            nn.Conv2d(64,32,3,padding=1), nn.ReLU(),
            nn.Conv2d(32,16,3,padding=1), nn.ReLU(),
            nn.Conv2d(16,3,3,padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Colouriser class
class Colouriser:
    def __init__(self, dataset, epochs=50):  # more epochs
        self.dataset = dataset
        self.model = ColouriserModel()
        self.optimiser = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.epochs = epochs
        self.train_model()

    def train_model(self):
        for epoch in range(self.epochs):
            total_loss = 0
            for sketch, target in self.dataset.data:
                sketch_tensor = torch.tensor(sketch, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                target_tensor = torch.tensor(target, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                self.optimiser.zero_grad()
                output = self.model(sketch_tensor)
                loss = self.loss_fn(output, target_tensor)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()
            if epoch % 5 == 0:
                st.write(f'Epoch {epoch}, Loss: {total_loss/len(self.dataset):.4f}')

    def transform(self, sketch_img):
        sketch_arr = np.array(sketch_img)/255.0
        sketch_tensor = torch.tensor(sketch_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.model(sketch_tensor).squeeze(0).permute(1,2,0).numpy()
        output = (np.clip(output,0,1)*255).astype(np.uint8)
        return Image.fromarray(output)


# Streamlit Interface
st.title("Sketch Colour Transformer (ML)")
st.write("Draw or upload a sketch below:")

canvas_size = 64

# Brush controls
brush_size = st.slider("Brush size", 1, 8, 4)
draw_colour = st.color_picker("Brush colour (black for sketching)", "#000000")

# Interactive canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=brush_size,
    stroke_color=draw_colour,
    background_color="white",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)

# Save canvas to session
if canvas_result.image_data is not None:
    img_arr = canvas_result.image_data.astype(np.uint8)
    img_pil = Image.fromarray(img_arr).convert('L').resize((64,64))
    st.session_state.draw_img = img_pil
else:
    if 'draw_img' not in st.session_state:
        st.session_state.draw_img = Image.new("L", (canvas_size, canvas_size), 255)

# Upload sketch
uploaded = st.file_uploader("Upload sketch (optional)", type=['png','jpg'])
if uploaded:
    draw_img = Image.open(uploaded).convert('L')
    st.session_state.draw_img = draw_img

# Display canvas
st.image(st.session_state.draw_img.resize((256, 256)), caption="Sketch Canvas", width=256)

# Train model if not loaded
if 'model' not in st.session_state:
    with st.spinner('Training model, please wait...'):
        dataset = SketchDataset(num_samples=500)
        st.session_state.model = Colouriser(dataset, epochs=50)
    st.success("Training complete!")

# Transform button
if st.button("Transform"):
    with st.spinner('Generating colourised image...'):
        output_img = st.session_state.model.transform(st.session_state.draw_img)
    st.image(output_img, caption="Colourised Image", width=256)
