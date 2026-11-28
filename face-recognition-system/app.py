import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, gray):
        return self.detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )


class FaceDataset:
    def __init__(self, root="faces"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def add_face(self, name, face_img):
        person_dir = os.path.join(self.root, name)
        os.makedirs(person_dir, exist_ok=True)
        count = len(os.listdir(person_dir))
        path = os.path.join(person_dir, f"{count}.png")
        cv2.imwrite(path, face_img)

    def load(self):
        images, labels, names = [], [], {}
        label_id = 0

        for person in sorted(os.listdir(self.root)):
            person_dir = os.path.join(self.root, person)
            if not os.path.isdir(person_dir):
                continue
            names[label_id] = person
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(label_id)
            label_id += 1

        return images, np.array(labels), names


class FaceRecogniser:
    def __init__(self):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}

    def train(self, images, labels, names):
        if len(images) == 0:
            return False
        self.model.train(images, labels)
        self.names = names
        return True

    def predict(self, face_img):
        label, confidence = self.model.predict(face_img)
        return self.names.get(label, "Unknown"), confidence


class App:
    def __init__(self):
        st.set_page_config(page_title="Real-time Face Recognition", layout="centered")
        self.detector = FaceDetector()
        self.dataset = FaceDataset()
        self.recogniser = FaceRecogniser()

    def run(self):
        st.title("Real-time Face Recognition")

        mode = st.radio("Mode", ["Register face", "Recognise face"])

        if mode == "Register face":
            self.register_mode()
        else:
            self.recognition_mode()

    def register_mode(self):
        name = st.text_input("Person name")
        img_file = st.camera_input("Capture face")

        if img_file and name:
            img = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                self.dataset.add_face(name, face)
                st.success("Face saved")

            st.image(frame, width=400)

    def recognition_mode(self):
        images, labels, names = self.dataset.load()
        trained = self.recogniser.train(images, labels, names)

        if not trained:
            st.warning("No registered faces found")
            return

        img_file = st.camera_input("Show face")

        if img_file:
            img = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            for (x, y, w, h) in faces:
                face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                name, conf = self.recogniser.predict(face)
                label = f"{name} ({conf:.1f})"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            st.image(frame, width=400)


if __name__ == "__main__":
    App().run()
