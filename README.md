# Face Symmetry Analysis App

This application allows you to upload a face image and analyzes its symmetry using Machine Learning and Optimization techniques.

## Features
- **Face Segmentation**: Uses MediaPipe Face Mesh to isolate the face from hair, neck, and background.
- **Symmetry Optimization**: Uses `scipy.optimize` to find the optimal rotation and vertical axis that minimizes the pixel difference between the left and right sides of the face.
- **Visualization**: Shows the original, the segmentation mask, the optimization process, and the final symmetrized result with a heatmap of asymmetry.

## Installation

1.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:
```bash
streamlit run symmetry_app.py
```

Upload an image and click "Analyze Symmetry".
