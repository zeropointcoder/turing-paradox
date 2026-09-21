import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
import cv2
import random


# Generates augmented templates for training
class TemplateGenerator:
    def __init__(self, size=32, augment=20):
        self.size = size
        self.font = ImageFont.load_default()
        self.characters = list("0123456789+-*/()")
        self.augment = augment  # number of variations per character
        self.templates, self.labels = self._generate_templates()

    def _generate_templates(self):
        images, labels = [], []
        for ch in self.characters:
            for _ in range(self.augment):
                img = Image.new("L", (self.size, self.size), 255)
                draw = ImageDraw.Draw(img)

                # Random small shifts
                shift_x = random.randint(-2, 2)
                shift_y = random.randint(-2, 2)
                bbox = draw.textbbox((0,0), ch, font=self.font)
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]

                x = (self.size - w)//2 + shift_x
                y = (self.size - h)//2 + shift_y
                draw.text((x, y), ch, fill=0, font=self.font)

                # Random rotation
                angle = random.randint(-15, 15)
                img = img.rotate(angle, fillcolor=255)

                # Flatten and store
                images.append(np.array(img).flatten())
                labels.append(ch)
        return np.array(images), np.array(labels)                


# ML-based OCR
class OCRModel:
    def __init__(self, templates, labels):
        self.clf = RandomForestClassifier(n_estimators=150)
        self.clf.fit(templates, labels)

    def recognise(self, image):
        # Threshold and find contours
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[0])
        expression = ""
        for x, y, w, h in boxes:
            if w < 5 or h < 5:
                continue
            roi = cv2.resize(thresh[y:y+h, x:x+w], (32,32)).flatten().reshape(1,-1)
            pred = self.clf.predict(roi)[0]
            expression += pred
        return expression


# Safely evaluates expression
class Calculator:
    def evaluate(self, expression):
        try:
            allowed = "0123456789+-*/()."
            if all(c in allowed for c in expression):
                return eval(expression, {"__builtins__": {}})
        except Exception:
            return "Invalid expression"


# Full calculator pipeline
class VisionCalculator:
    def __init__(self):
        generator = TemplateGenerator()
        self.ocr = OCRModel(generator.templates, generator.labels)
        self.calculator = Calculator()

    def process(self, image):
        expr = self.ocr.recognise(image)
        result = self.calculator.evaluate(expr)
        return expr, result


# Streamlit UI
class App:
    def __init__(self):
        self.system = VisionCalculator()

    def run(self):
        st.title("ML-powered Calculator (offline and robust)")
        typed = st.text_input("Enter expression (printed font recommended)")
        if typed:
            img = Image.new("L", (300,60), 255)
            draw = ImageDraw.Draw(img)
            draw.text((10,10), typed, fill=0, font=ImageFont.load_default())
            image_np = np.array(img)
            expression, result = self.system.process(image_np)
            st.image(img, caption="Generated Image", width=400)
            st.markdown(f"**Recognised expression:** `{expression}`")
            st.markdown(f"**Result:** `{result}`")

        uploaded = st.file_uploader("Or upload a printed expression image", type=["png","jpg","jpeg"])
        if uploaded:
            image = Image.open(uploaded).convert("L")
            image_np = np.array(image)
            expression, result = self.system.process(image_np)
            st.image(image, caption="Uploaded image", width=400)
            st.markdown(f"**Recognised expression:** `{expression}`")
            st.markdown(f"**Result:** `{result}`")


if __name__ == "__main__":
    App().run()