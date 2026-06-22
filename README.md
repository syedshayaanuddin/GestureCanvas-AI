🎨 GestureCanvas

AI-Powered Gesture Controlled Smart Drawing System
GestureCanvas is a computer vision application that transforms hand gestures into a natural drawing interface. Using real-time hand tracking, users can draw, erase, change colors, resize brushes, recognize geometric shapes, and interact with an intelligent virtual canvas without touching the screen.
The project combines OpenCV, MediaPipe, image processing, gesture recognition, contour analysis, and layered rendering into a complete Human-Computer Interaction (HCI) application.

Features:

✋ Real-Time Hand Tracking:

MediaPipe Hand Landmark Detection
21 hand landmarks
Single-hand gesture control
Stable real-time tracking

🎨 Smart Drawing:

Smooth brush rendering
Adaptive motion smoothing
Adjustable brush size
Adjustable eraser size
Multiple drawing colors
Anti-aliased strokes

🤏 Gesture Controls: 

Gesture	Action
Index Finger	Draw
Index + Middle	Toolbar Selection
Thumb + Index	Brush Size Adjustment
Three Fingers	Smart Fill Tool
Fist	AI Shape Recognition

🧠 AI Shape Recognition:

Recognizes rough hand-drawn shapes and converts them into clean geometric figures.
Supported shapes include:

Circle
Rectangle
Square
Triangle
Pentagon
Hexagon
Polygon (fallback)

Features:

Contour approximation
Circularity analysis
Convex hull processing
Shape confidence estimation
Animated snapping

🎯 Smart Fill Tool:

Gesture-based paint bucket implementation featuring

Flood-fill algorithm
Binary mask generation
Leak detection
Morphological operations
Layer-aware filling
Boundary preservation

🧱 Layer System:

GestureCanvas uses a custom multi-layer rendering architecture.

Drawing Layer
AI Preview Layer
Animation Layer
Composite Rendering

This allows

Smooth animations
Undo support
Non-destructive editing
Better rendering quality

⚡ Performance Optimizations:

Adaptive smoothing
Hover debouncing
Click cooldown
Gesture confirmation
FPS optimization
Anti-flicker toolbar
Stable gesture state machine
Technologies Used
Python
OpenCV
MediaPipe
NumPy
Computer Vision
Image Processing
Gesture Recognition
Contour Detection
Project Statistics
Metric	Value
Programming Language	Python
Camera Resolution	1280 × 720
Hand Landmarks	21
Supported Gestures	5+
Shape Types	6+
Drawing Layers	3
Brush Size Range	4–50 px
Eraser Size Range	20–120 px

Installation
git clone https://github.com/yourusername/GestureCanvas.git
cd GestureCanvas
pip install -r requirements.txt
python virtualpainter.py

Future Improvements:

OCR-based handwritten text recognition
Mathematical expression recognition
AI-assisted diagram correction
Multi-hand collaboration
Gesture-customization menu
Export to SVG/PDF
Pressure-sensitive virtual brush
Cloud save support
Voice commands
Challenges Faced

Some of the major engineering challenges during development included:

Stable real-time gesture recognition
Eliminating cursor jitter
Designing a reliable gesture state machine
Preventing accidental gesture activation
Shape classification using contour analysis
Accurate flood-fill without leakage
Managing rendering layers
Maintaining real-time performance
Learning Outcomes

Through this project I gained practical experience in:

Computer Vision
Human-Computer Interaction (HCI)
OpenCV
MediaPipe
Image Processing
Gesture Recognition
State Machine Design
Layer-based Rendering
Real-time Application Development

Main Interface:

Drawing Demo
Shape Recognition
Smart Fill
Brush Resize


License:
MIT License

🌟 If you found this project interesting, consider giving it a star!

requirements.txt
opencv-python
opencv-contrib-python
mediapipe
numpy
