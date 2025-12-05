import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from scipy.optimize import minimize
from PIL import Image
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_face_mask(image_rgb):
    """
    Uses MediaPipe to detect face landmarks and create a binary mask 
    excluding hair, neck, and background.
    """
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None, None

    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get the convex hull of the face landmarks
    # We use specific indices for the face oval to exclude ears/neck if possible, 
    # but convex hull of all landmarks is a good start for "face only".
    # MediaPipe's face oval indices: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    # Let's use the full set of points for a robust hull.
    points = []
    for lm in landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))
    
    points = np.array(points)
    hull = cv2.convexHull(points)
    
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Refine mask: erode slightly to avoid edge artifacts
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
    
    # Get the center of the face (centroid of landmarks) for initial guess
    center_x = np.mean([p[0] for p in points])
    center_y = np.mean([p[1] for p in points])
    
    # Calculate initial rotation and shift guess based on eyes and nose
    # Left Iris: 468, Right Iris: 473
    # We used refine_landmarks=True so irises should be there.
    landmarks_debug = {}
    try:
        left_iris = landmarks[468]
        right_iris = landmarks[473]
        nose_tip = landmarks[1]
        
        landmarks_debug['left_iris'] = (int(left_iris.x * w), int(left_iris.y * h))
        landmarks_debug['right_iris'] = (int(right_iris.x * w), int(right_iris.y * h))
        landmarks_debug['nose_tip'] = (int(nose_tip.x * w), int(nose_tip.y * h))

        # 1. Initial Angle
        # Calculate angle of the line connecting the eyes
        dy = (right_iris.y - left_iris.y) * h
        dx = (right_iris.x - left_iris.x) * w
        angle = np.degrees(np.arctan2(dy, dx))
        initial_angle = angle 
        
        # 2. Initial Shift
        # We want the symmetry axis to pass through the midpoint between eyes.
        # This is usually more robust than the nose for "symmetry".
        eye_midpoint_x = (left_iris.x + right_iris.x) / 2 * w
        eye_midpoint_y = (left_iris.y + right_iris.y) / 2 * h
        
        # We need the x-coordinate of this midpoint AFTER rotation by initial_angle around (center_x, center_y).
        M = cv2.getRotationMatrix2D((center_x, center_y), initial_angle, 1.0)
        midpoint_pt = np.array([eye_midpoint_x, eye_midpoint_y, 1.0])
        rotated_midpoint = M @ midpoint_pt
        rotated_midpoint_x = rotated_midpoint[0]
        
        initial_shift = rotated_midpoint_x - center_x
        
    except IndexError:
        # Fallback if iris landmarks are missing
        initial_angle = 0
        initial_shift = 0
    
    return mask, (center_x, center_y), (initial_angle, initial_shift), landmarks_debug

def rotate_image(image, angle, center=None):
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

def transform_and_compare(params, image_gray, mask, center_guess, debug_container=None):
    """
    params: [angle, x_shift]
    angle: rotation in degrees to correct tilt
    x_shift: horizontal shift from the center_guess to find the symmetry axis
    """
    angle, x_shift = params
    h, w = image_gray.shape
    
    # 1. Rotate the image and mask to correct tilt
    # We rotate around the estimated face center
    M_rot = cv2.getRotationMatrix2D(center_guess, angle, 1.0)
    img_rot = cv2.warpAffine(image_gray, M_rot, (w, h), flags=cv2.INTER_LINEAR)
    mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    
    # 2. Define the symmetry axis. 
    # After rotation, we assume the symmetry axis is vertical at x = center_guess[0] + x_shift
    axis_x = center_guess[0] + x_shift
    
    # 3. Create the flipped image
    # To flip around a specific vertical line x=c:
    # We can flip the whole image horizontally, then shift it.
    # Or simpler: extract a window, flip it? No, we need pixel-wise comparison.
    # Transformation matrix for flipping around x = axis_x:
    # x' = -(x - axis_x) + axis_x = -x + 2*axis_x
    # y' = y
    # [ -1  0  2*axis_x ]
    # [  0  1  0        ]
    
    M_flip = np.float32([[-1, 0, 2*axis_x], [0, 1, 0]])
    img_flipped = cv2.warpAffine(img_rot, M_flip, (w, h), flags=cv2.INTER_LINEAR)
    mask_flipped = cv2.warpAffine(mask_rot, M_flip, (w, h), flags=cv2.INTER_NEAREST)
    
    # 4. Compare only where both masks are active (intersection)
    # We want to minimize difference between img_rot and img_flipped
    
    intersection = cv2.bitwise_and(mask_rot, mask_flipped)
    
    # If intersection is too small, penalize heavily
    if np.sum(intersection) < 1000:
        return 1e9
        
    # Difference
    diff = cv2.absdiff(img_rot, img_flipped)
    
    # Mask the difference
    diff_masked = cv2.bitwise_and(diff, diff, mask=intersection)
    
    # Calculate Mean Squared Error or L1 norm
    # L1 might be more robust to outliers
    score = np.sum(diff_masked) / np.sum(intersection)
    
    # Visualization callback
    if debug_container is not None:
        # Create a visualization: Left half original, Right half flipped (composited)
        # Or just an overlay
        vis = cv2.addWeighted(img_rot, 0.5, img_flipped, 0.5, 0)
        # Draw the axis
        cv2.line(vis, (int(axis_x), 0), (int(axis_x), h), (255, 0, 0), 2)
        debug_container.image(vis, caption=f"Optimizing... Score: {score:.2f}", clamp=True)
        # time.sleep(0.05) # Slow down slightly to see animation
        
    return score

def main():
    st.set_page_config(page_title="Face Symmetry Finder", layout="wide")
    
    st.title("Face Symmetry Analysis AI")
    st.markdown("""
    Upload a photo of a face. This app uses Machine Learning (MediaPipe) to segment the face 
    and an optimization algorithm to find the perfect symmetry axis by minimizing pixel differences.
    """)
    
    uploaded_file = st.file_uploader("Choose a face image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize for performance if too large
        max_dim = 800
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
            
        st.image(image, caption="Original Image", use_container_width=True)
        
        if st.button("Analyze Symmetry"):
            with st.spinner("Detecting face and segmenting..."):
                mask, center_guess, initial_guess_params, landmarks_debug = get_face_mask(image)
                
            if mask is None:
                st.error("No face detected! Please try another image.")
                return
                
            st.success("Face detected! Starting optimization...")
            
            col1, col2 = st.columns(2)
            with col1:
                # Draw landmarks on a copy of the image for visualization
                img_debug = image.copy()
                if landmarks_debug:
                    cv2.circle(img_debug, landmarks_debug['left_iris'], 5, (0, 255, 0), -1)
                    cv2.circle(img_debug, landmarks_debug['right_iris'], 5, (0, 255, 0), -1)
                    cv2.circle(img_debug, landmarks_debug['nose_tip'], 5, (255, 0, 0), -1)
                    # Draw line between eyes
                    cv2.line(img_debug, landmarks_debug['left_iris'], landmarks_debug['right_iris'], (0, 255, 0), 2)
                
                st.image(img_debug, caption="Detected Landmarks (Eyes & Nose)", use_container_width=True)
            
            # Prepare for optimization
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Placeholder for live visualization
            with col2:
                st.markdown("### Optimization Progress")
                viz_placeholder = st.empty()
            
            # Optimization function wrapper to pass extra args
            def loss_func(params):
                return transform_and_compare(params, image_gray, mask, center_guess, viz_placeholder)
            
            # Initial guess: Use calculated params from landmarks
            initial_guess = initial_guess_params
            st.info(f"Initial Guess from Landmarks - Angle: {initial_guess[0]:.2f}°, Shift: {initial_guess[1]:.2f} px")
            
            # Bounds: Angle +/- 10 degrees, Shift +/- 20 pixels
            # We trust the landmarks significantly more now.
            # The optimization should only refine slightly.
            bounds = [
                (initial_guess[0] - 10, initial_guess[0] + 10), 
                (initial_guess[1] - 20, initial_guess[1] + 20)
            ]
            
            # Run optimization
            # Powell is a good method for this kind of parameter search without gradients
            res = minimize(loss_func, initial_guess, method='Powell', bounds=bounds, tol=1e-3)
            
            best_angle, best_shift = res.x
            best_score = res.fun
            
            st.success(f"Optimization Complete! Best Angle: {best_angle:.2f}°, Shift: {best_shift:.2f} px")
            
            # Generate Final Result
            h, w = image_gray.shape
            axis_x = center_guess[0] + best_shift
            
            # 1. Rotate Original Color Image
            M_rot = cv2.getRotationMatrix2D(center_guess, best_angle, 1.0)
            final_rot = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR)
            
            # 2. Create Symmetrical Face (Left side mirrored to Right)
            # We need to decide which side is "better" or just show the mirror.
            # Usually "Symmetry" implies constructing a face from one half.
            # But the prompt asks to "flip the face... to find how symmetrical the face is".
            # So showing the difference map is good, and maybe the "Perfectly Symmetrical" version.
            
            # Let's create the "Symmetrized" version by averaging the flipped and original
            M_flip = np.float32([[-1, 0, 2*axis_x], [0, 1, 0]])
            final_flipped = cv2.warpAffine(final_rot, M_flip, (w, h), flags=cv2.INTER_LINEAR)
            
            # Create a composite: Left half of Rotated + Right half of Flipped (which is Left side mirrored)
            # Actually, let's just show the "Symmetrical Composite" (Average)
            symmetrical_avg = cv2.addWeighted(final_rot, 0.5, final_flipped, 0.5, 0)
            
            # Difference Map (Heatmap)
            diff = cv2.absdiff(final_rot, final_flipped)
            diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            
            # Mask the heatmap
            mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
            mask_flipped = cv2.warpAffine(mask_rot, M_flip, (w, h), flags=cv2.INTER_NEAREST)
            intersection = cv2.bitwise_and(mask_rot, mask_flipped)
            diff_heatmap = cv2.bitwise_and(diff_heatmap, diff_heatmap, mask=intersection)

            st.markdown("---")
            st.markdown("### Results")
            
            r_col1, r_col2, r_col3 = st.columns(3)
            
            with r_col1:
                st.image(final_rot, caption="Aligned Original", use_container_width=True)
                
            with r_col2:
                st.image(symmetrical_avg, caption="Symmetrized Face (Average)", use_container_width=True)
                
            with r_col3:
                st.image(diff_heatmap, caption="Asymmetry Heatmap (Blue=Low, Red=High)", use_container_width=True)

if __name__ == "__main__":
    main()
