import streamlit as st
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
import time


# Synthetic video generator
class SyntheticVideo:
    def __init__(self, frame_size=(480, 640), num_objects=3):
        self.h, self.w = frame_size
        self.num_objects = num_objects
        self.reset()
        self.frozen = [False] * num_objects

    def reset(self):
        self.positions = np.random.randint(50, 400, (self.num_objects, 2)).astype(float)
        self.velocities = np.random.randint(-5, 5, (self.num_objects, 2)).astype(float)
        self.frozen = [False] * self.num_objects

    def next_frame(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.num_objects):
            if not self.frozen[i]:
                self.positions[i] += self.velocities[i]
                for d in range(2):
                    if self.positions[i][d] < 20 or self.positions[i][d] > (self.w if d == 0 else self.h) - 20:
                        self.velocities[i][d] *= -1
            cv2.circle(frame, tuple(self.positions[i].astype(int)), 15, (0, 255, 0), -1)
        return frame, self.positions.copy()


# Object detector
class ObjectDetector:
    @staticmethod
    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centres = []
        for cnt in contours:
            m = cv2.moments(cnt)
            if m["m00"] > 0:
                centres.append((int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])))
        return centres


# ML Tracker for rings
class MLTracker:
    def __init__(self, num_objects=3, history_len=3, speed=0.2):
        self.num_objects = num_objects
        self.history_len = history_len
        self.speed = speed
        self.positions = np.random.randint(50, 400, (num_objects,2)).astype(float)
        self.history = [np.zeros((history_len,2)) for _ in range(num_objects)]
        self.models_x = [LinearRegression() for _ in range(num_objects)]
        self.models_y = [LinearRegression() for _ in range(num_objects)]
        self.done = [False]*num_objects
        self.assigned_object = [None]*num_objects

    def update_history(self, detected):
        for i, pos in enumerate(detected):
            self.history[i] = np.roll(self.history[i], -1, axis=0)
            self.history[i][-1] = pos

    def history_ready(self):
        return all(np.count_nonzero(h)==self.history_len*2 for h in self.history)

    def train_predict(self):
        preds = []
        if self.history_ready():
            X = np.arange(self.history_len).reshape(-1,1)
            for i in range(self.num_objects):
                self.models_x[i].fit(X, self.history[i][:,0])
                self.models_y[i].fit(X, self.history[i][:,1])
                next_step = np.array([[self.history_len]])
                preds.append((self.models_x[i].predict(next_step)[0],
                              self.models_y[i].predict(next_step)[0]))
        else:
            preds = [tuple(self.history[i][-1]) for i in range(self.num_objects)]
        return preds

    def chase(self, detected, video_obj, threshold=5):
        preds = self.train_predict()
        available = set(range(len(detected)))

        # Assign rings uniquely
        for i in range(self.num_objects):
            if self.done[i]:
                continue
            if self.assigned_object[i] is None:
                min_dist, closest = float('inf'), None
                for obj_idx in available:
                    dist = np.linalg.norm(self.positions[i]-detected[obj_idx])
                    if dist < min_dist:
                        min_dist = dist
                        closest = obj_idx
                self.assigned_object[i] = closest
                available.remove(closest)

        # Move rings
        for i in range(self.num_objects):
            if self.done[i]:
                continue
            obj_idx = self.assigned_object[i]
            dx, dy = detected[obj_idx][0]-self.positions[i][0], detected[obj_idx][1]-self.positions[i][1]
            self.positions[i][0] += dx*self.speed
            self.positions[i][1] += dy*self.speed
            if np.linalg.norm(self.positions[i] - detected[obj_idx]) < threshold:
                self.done[i] = True
                self.positions[i] = np.array(detected[obj_idx])  # <- fixed here
                video_obj.frozen[obj_idx] = True


# Streamlit App
st.set_page_config(page_title="ML Tracker Rings", layout="wide")
st.title("ML-Based Tracker Rings Following Circular Objects")

# Session state
if 'video' not in st.session_state: st.session_state['video']=SyntheticVideo()
if 'tracker' not in st.session_state: st.session_state['tracker']=MLTracker()
if 'detector' not in st.session_state: st.session_state['detector']=ObjectDetector()
if 'running' not in st.session_state: st.session_state['running']=True
if 'last_frame' not in st.session_state: st.session_state['last_frame']=np.zeros((480,640,3),np.uint8)
if 'tracking_done' not in st.session_state: st.session_state['tracking_done']=False

video = st.session_state['video']
tracker = st.session_state['tracker']
detector = st.session_state['detector']
frame_slot = st.empty()

# Buttons
c1,c2,c3 = st.columns([1,1,1])
with c1:
    if st.button("Reload"):
        st.session_state['video'] = SyntheticVideo()
        st.session_state['tracker'] = MLTracker()
        st.session_state['running'] = True
        st.session_state['tracking_done'] = False
        st.session_state['last_frame'] = np.zeros((480,640,3),np.uint8)
with c2:
    if st.button("Pause"): st.session_state['running'] = False
with c3:
    if st.button("Play"): st.session_state['running'] = True

# Loop
while True:
    if st.session_state['running'] and not st.session_state['tracking_done']:
        frame, positions = video.next_frame()
        detected = detector.detect(frame)
        if len(detected) == tracker.num_objects:
            tracker.update_history(np.array(detected))
            tracker.chase(detected, video)
            st.session_state['tracking_done'] = all(tracker.done)
        for x,y in tracker.positions:
            cv2.circle(frame,(int(x),int(y)),20,(255,0,0),2)
        st.session_state['last_frame'] = frame.copy()
    else:
        frame = st.session_state['last_frame']

    frame_slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),width=700)
    time.sleep(0.03)
