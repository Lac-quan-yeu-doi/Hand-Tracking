Kick-start:
- in folder `Hand-Tracking\`, runs `pip install -e .` to install current directory as a python module. `pyproject.toml` has `where=['src']` to set global package-site setup so no need to import `sys` and append the path 

Google Mediapipe documentation: [https://ai.google.dev/edge/mediapipe/solutions/guide]
- Mediapipe Hand landmarker: [https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker]
Gesture recognition models reference from: [https://github.com/ahmetgunduz/Real-time-GesRec]

Finger counting:
- Dataset: [https://zenodo.org/records/3901659]
- Each image pass through the frame_detection to get coordinates => Train ML for detection