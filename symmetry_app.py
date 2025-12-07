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
    
    Also estimates the symmetry axis using multiple facial landmarks
    with weighted fitting and outlier rejection for robustness.
    """
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None, None, None, None

    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get the convex hull of the face landmarks for mask
    points = []
    for lm in landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))
    
    points = np.array(points)
    hull = cv2.convexHull(points)
    
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Refine mask: erode slightly to avoid edge artifacts
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
    
    # Get the center of the face (centroid of landmarks) for rotation pivot
    center_x = np.mean([p[0] for p in points])
    center_y = np.mean([p[1] for p in points])
    
    landmarks_debug = {}
    
    # Symmetry axis estimation using multiple anatomical landmarks
    # that should lie on the facial midline
    
    # MediaPipe landmark indices for midline points (top to bottom):
    # These landmarks should ideally lie on the vertical symmetry axis
    MIDLINE_LANDMARKS = {
        'forehead_top': 10,      # Top of forehead
        'glabella': 9,           # Between eyebrows (glabella)
        'nose_bridge_top': 168,  # Top of nose bridge
        'nose_bridge_mid': 6,    # Middle of nose bridge
        'nose_tip': 1,           # Tip of nose
        'philtrum': 164,         # Philtrum (above upper lip)
        'upper_lip': 0,          # Center of upper lip
        'lower_lip': 17,         # Center of lower lip
        'chin': 152,             # Chin point
    }
    
    # Symmetric landmark pairs (left, right) for angle estimation
    SYMMETRIC_PAIRS = [
        (468, 473),   # Irises (if available with refine_landmarks=True)
        (33, 263),    # Inner eye corners
        (133, 362),   # Outer eye corners
        (70, 300),    # Upper eyelid centers
        (105, 334),   # Lower eyelid centers
        (61, 291),    # Mouth corners
        (78, 308),    # Upper lip sides
        (95, 324),    # Lower lip sides
        (50, 280),    # Upper cheeks
        (101, 330),   # Lower cheeks
        (234, 454),   # Temple/jaw line
    ]
    
    try:
        # Collect midline points
        midline_points = []
        for name, idx in MIDLINE_LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                midline_points.append({
                    'name': name,
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'weight': 1.0
                })
        
        # Assign higher weights to more reliable midline landmarks
        weight_map = {
            'nose_bridge_top': 2.0,
            'nose_bridge_mid': 2.0,
            'nose_tip': 1.5,
            'philtrum': 1.5,
            'chin': 1.0,
            'forehead_top': 0.8,  # Less reliable
            'glabella': 1.2,
            'upper_lip': 1.3,
            'lower_lip': 1.0,
        }
        for pt in midline_points:
            pt['weight'] = weight_map.get(pt['name'], 1.0)
        
        # Calculate rotation from symmetric pairs
        angles = []
        angle_weights = []
        
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_lm = landmarks[left_idx]
                right_lm = landmarks[right_idx]
                
                lx, ly = left_lm.x * w, left_lm.y * h
                rx, ry = right_lm.x * w, right_lm.y * h
                
                # The line connecting symmetric points should be horizontal
                # after correction, so the angle of this line IS the needed rotation
                dx = rx - lx
                dy = ry - ly
                
                if abs(dx) > 5:  # Avoid division issues for nearly coincident points
                    pair_angle = np.degrees(np.arctan2(dy, dx))
                    
                    # Weight by distance (longer pairs are more reliable)
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Weight irises heavily, they're very reliable
                    pair_weight = dist
                    if (left_idx, right_idx) == (468, 473):
                        pair_weight *= 3.0  # Irises get 3x weight
                    elif (left_idx, right_idx) in [(33, 263), (133, 362)]:
                        pair_weight *= 2.0  # Eye corners get 2x weight
                    
                    angles.append(pair_angle)
                    angle_weights.append(pair_weight)
        
        # Robust angle estimation: weighted median-ish approach
        if len(angles) > 0:
            angles = np.array(angles)
            angle_weights = np.array(angle_weights)
            
            # Use weighted average but with outlier rejection
            # First pass: weighted mean
            weighted_mean_angle = np.average(angles, weights=angle_weights)
            
            # Second pass: reject outliers (> 5 degrees from weighted mean)
            inliers = np.abs(angles - weighted_mean_angle) < 5.0
            if np.sum(inliers) >= 3:
                initial_angle = np.average(angles[inliers], weights=angle_weights[inliers])
            else:
                initial_angle = weighted_mean_angle
        else:
            initial_angle = 0.0
        
        # Calculate shift from midline points
        # After rotating by initial_angle, the midline points should align vertically
        # Find the x-coordinate where they best align
        
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), initial_angle, 1.0)
        
        rotated_midline_x = []
        rotated_midline_weights = []
        
        for pt in midline_points:
            original_pt = np.array([pt['x'], pt['y'], 1.0])
            rotated_pt = M_rot @ original_pt
            rotated_midline_x.append(rotated_pt[0])
            rotated_midline_weights.append(pt['weight'])
        
        if len(rotated_midline_x) > 0:
            # Weighted median for robustness
            rotated_midline_x = np.array(rotated_midline_x)
            rotated_midline_weights = np.array(rotated_midline_weights)
            
            # Use weighted mean with outlier rejection
            weighted_mean_x = np.average(rotated_midline_x, weights=rotated_midline_weights)
            
            # Reject outliers (points too far from weighted mean - likely measurement error)
            deviations = np.abs(rotated_midline_x - weighted_mean_x)
            threshold = np.percentile(deviations, 75) * 2.5 + 3  # Robust threshold
            inliers = deviations < threshold
            
            if np.sum(inliers) >= 3:
                axis_x_after_rotation = np.average(
                    rotated_midline_x[inliers], 
                    weights=rotated_midline_weights[inliers]
                )
            else:
                axis_x_after_rotation = weighted_mean_x
            
            initial_shift = axis_x_after_rotation - center_x
        else:
            initial_shift = 0.0
        
        # Add symmetric pair midpoints for additional validation
        # The midpoint of each symmetric pair should also lie on the symmetry axis
        pair_midpoints_x = []
        
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_lm = landmarks[left_idx]
                right_lm = landmarks[right_idx]
                
                mid_x = (left_lm.x + right_lm.x) / 2 * w
                mid_y = (left_lm.y + right_lm.y) / 2 * h
                
                original_pt = np.array([mid_x, mid_y, 1.0])
                rotated_pt = M_rot @ original_pt
                pair_midpoints_x.append(rotated_pt[0])
        
        if len(pair_midpoints_x) > 0:
            # Combine midline points and pair midpoints for final estimate
            pair_midpoints_x = np.array(pair_midpoints_x)
            combined_x = np.concatenate([
                rotated_midline_x,
                pair_midpoints_x
            ])
            combined_weights = np.concatenate([
                rotated_midline_weights * 1.5,  # Midline points get higher weight
                np.ones(len(pair_midpoints_x))
            ])
            
            # Final robust estimate
            final_axis_x = np.average(combined_x, weights=combined_weights)
            
            # Outlier rejection
            deviations = np.abs(combined_x - final_axis_x)
            threshold = np.percentile(deviations, 80) * 2.0 + 3
            inliers = deviations < threshold
            
            if np.sum(inliers) >= 5:
                final_axis_x = np.average(combined_x[inliers], weights=combined_weights[inliers])
            
            initial_shift = final_axis_x - center_x
        
        # Store debug info
        try:
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            nose_tip = landmarks[1]
            
            landmarks_debug['left_iris'] = (int(left_iris.x * w), int(left_iris.y * h))
            landmarks_debug['right_iris'] = (int(right_iris.x * w), int(right_iris.y * h))
            landmarks_debug['nose_tip'] = (int(nose_tip.x * w), int(nose_tip.y * h))
            
            # Add midline debug visualization
            landmarks_debug['midline_points'] = [
                (int(pt['x']), int(pt['y'])) for pt in midline_points
            ]
        except IndexError:
            pass
        
    except Exception as e:
        # Ultimate fallback
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

def transform_and_compare(params, image_gray, mask, center_guess, edge_weights=None):
    """
    Compares an image with its reflection to score symmetry quality.
    Uses gradient-weighted comparison for better feature alignment.
    
    Args:
        params: [angle, x_shift] - rotation angle and horizontal shift
        image_gray: grayscale image to analyze
        mask: face region mask
        center_guess: estimated face center (x, y)
        edge_weights: pre-computed edge map for weighting (optional)
    
    Returns:
        float: symmetry score (lower is more symmetric)
    """
    angle, x_shift = params
    h, w = image_gray.shape
    
    # 1. Rotate the image and mask to correct tilt
    M_rot = cv2.getRotationMatrix2D(center_guess, angle, 1.0)
    img_rot = cv2.warpAffine(image_gray, M_rot, (w, h), flags=cv2.INTER_LINEAR)
    mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    
    # 2. Define the symmetry axis
    axis_x = center_guess[0] + x_shift
    
    # 3. Create the flipped image around the symmetry axis
    M_flip = np.float32([[-1, 0, 2*axis_x], [0, 1, 0]])
    img_flipped = cv2.warpAffine(img_rot, M_flip, (w, h), flags=cv2.INTER_LINEAR)
    mask_flipped = cv2.warpAffine(mask_rot, M_flip, (w, h), flags=cv2.INTER_NEAREST)
    
    # Also transform edge weights if provided
    if edge_weights is not None:
        edge_rot = cv2.warpAffine(edge_weights, M_rot, (w, h), flags=cv2.INTER_LINEAR)
        edge_flipped = cv2.warpAffine(edge_rot, M_flip, (w, h), flags=cv2.INTER_LINEAR)
    
    # 4. Compare only where both masks are active (intersection)
    intersection = cv2.bitwise_and(mask_rot, mask_flipped)
    
    # If intersection is too small, penalize heavily
    intersection_sum = np.sum(intersection > 0)
    if intersection_sum < 1000:
        return 1e9
    
    # 5. Calculate difference with edge weighting
    diff = np.abs(img_rot.astype(float) - img_flipped.astype(float))
    
    if edge_weights is not None:
        # Combine edge maps from both original and flipped (max of both)
        combined_edges = np.maximum(edge_rot, edge_flipped)
        # Normalize edge weights to [0.5, 1.5] range to not completely ignore flat areas
        normalized_edges = 0.5 + (combined_edges / 255.0)
        diff = diff * normalized_edges
    
    # Apply mask
    diff_masked = diff * (intersection / 255.0)
    
    # 6. Calculate score - use robust statistics
    valid_diffs = diff_masked[intersection > 0]
    
    # Use trimmed mean to reduce outlier sensitivity 
    sorted_diffs = np.sort(valid_diffs)
    trim_amount = int(len(sorted_diffs) * 0.05)  # Trim 5% from each end
    if trim_amount > 0:
        trimmed = sorted_diffs[trim_amount:-trim_amount]
    else:
        trimmed = sorted_diffs
    
    # Combine L1 (mean) and L2 (std) for balanced scoring
    l1_score = np.mean(trimmed)
    l2_score = np.std(trimmed)
    score = l1_score + 0.3 * l2_score  # Penalize high variance too
    
    return score


@st.cache_data
def run_analysis(image):
    """
    Analyzes facial symmetry using multi-scale optimization.
    
    Finds the optimal symmetry axis by minimizing pixel differences
    between the face and its reflection, weighted by edge importance.
    """
    mask, center_guess, initial_guess_params, landmarks_debug = get_face_mask(image)
    
    if mask is None:
        return None
        
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = image_gray.shape
    
    # Multi-scale optimization with edge weighting
    
    # Compute edge map for importance weighting (facial features matter more)
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_magnitude = np.clip(edge_magnitude, 0, 255).astype(np.uint8)
    # Smooth slightly to avoid noise
    edge_weights = cv2.GaussianBlur(edge_magnitude, (5, 5), 0)
    
    initial_guess = initial_guess_params
    
    # Coarse optimization: wider search at reduced resolution
    scale_factor = 0.5
    small_gray = cv2.resize(image_gray, None, fx=scale_factor, fy=scale_factor)
    small_mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    small_edges = cv2.resize(edge_weights, None, fx=scale_factor, fy=scale_factor)
    small_center = (center_guess[0] * scale_factor, center_guess[1] * scale_factor)
    
    def coarse_loss(params):
        # Scale shift parameter for smaller image
        scaled_params = [params[0], params[1] * scale_factor]
        return transform_and_compare(
            scaled_params, small_gray, small_mask, small_center, small_edges
        )
    
    # Coarse search with wider bounds
    coarse_bounds = [
        (initial_guess[0] - 15, initial_guess[0] + 15),  # ±15° angle
        (initial_guess[1] - 40, initial_guess[1] + 40)   # ±40px shift
    ]
    
    # Global search using differential evolution
    from scipy.optimize import differential_evolution, minimize
    
    try:
        coarse_result = differential_evolution(
            coarse_loss, 
            coarse_bounds, 
            maxiter=30,
            tol=0.1,
            seed=42,
            workers=1,
            polish=False
        )
        coarse_angle, coarse_shift = coarse_result.x
    except:
        # Fallback if differential_evolution fails
        coarse_angle, coarse_shift = initial_guess
    
    # Fine optimization: local refinement at full resolution
    def fine_loss(params):
        return transform_and_compare(
            params, image_gray, mask, center_guess, edge_weights
        )
    
    # Fine bounds centered on coarse result
    fine_bounds = [
        (coarse_angle - 3, coarse_angle + 3),   # ±3° around coarse result
        (coarse_shift - 10, coarse_shift + 10)  # ±10px around coarse result
    ]
    
    fine_result = minimize(
        fine_loss, 
        [coarse_angle, coarse_shift], 
        method='Nelder-Mead',
        options={'xatol': 0.01, 'fatol': 0.1}
    )
    
    best_angle, best_shift = fine_result.x

    
    # Generate Final Result
    axis_x = center_guess[0] + best_shift
    
    # 1. Rotate Original Color Image
    M_rot = cv2.getRotationMatrix2D(center_guess, best_angle, 1.0)
    final_rot = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR)
    
    # Create image with symmetry line
    final_rot_with_line = final_rot.copy()
    cv2.line(final_rot_with_line, (int(axis_x), 0), (int(axis_x), h), (0, 255, 0), 2)
    
    # 2. Create Symmetrical Face
    M_flip = np.float32([[-1, 0, 2*axis_x], [0, 1, 0]])
    final_flipped = cv2.warpAffine(final_rot, M_flip, (w, h), flags=cv2.INTER_LINEAR)
    
    # Average Symmetry
    symmetrical_avg = cv2.addWeighted(final_rot, 0.5, final_flipped, 0.5, 0)
    
    # Difference Map (Enhanced Heatmap for highlighting asymmetries)
    diff = cv2.absdiff(final_rot, final_flipped)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    
    # Mask the difference
    mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    mask_flipped = cv2.warpAffine(mask_rot, M_flip, (w, h), flags=cv2.INTER_NEAREST)
    intersection = cv2.bitwise_and(mask_rot, mask_flipped)
    
    # Apply mask before processing
    diff_masked = cv2.bitwise_and(diff_gray, diff_gray, mask=intersection)
    
    # Enhance contrast: normalize to full range within the masked area
    masked_values = diff_masked[intersection > 0]
    if len(masked_values) > 0:
        min_val = np.percentile(masked_values, 2)  # Use percentiles to avoid outliers
        max_val = np.percentile(masked_values, 98)
        if max_val > min_val:
            diff_normalized = np.clip((diff_masked.astype(float) - min_val) / (max_val - min_val), 0, 1)
        else:
            diff_normalized = diff_masked.astype(float) / 255.0
    else:
        diff_normalized = diff_masked.astype(float) / 255.0
    
    # Apply gamma correction to boost visibility of subtle differences
    gamma = 0.5  # Values < 1 boost darker regions (subtle asymmetries)
    diff_boosted = np.power(diff_normalized, gamma)
    diff_boosted = (diff_boosted * 255).astype(np.uint8)
    
    # Apply colormap - use INFERNO for better perceptual contrast (black -> purple -> red -> yellow)
    diff_heatmap = cv2.applyColorMap(diff_boosted, cv2.COLORMAP_INFERNO)
    diff_heatmap = cv2.bitwise_and(diff_heatmap, diff_heatmap, mask=intersection)
    
    # Create overlay version: heatmap blended with face for context
    final_rot_bgr = cv2.cvtColor(final_rot, cv2.COLOR_RGB2BGR)
    heatmap_overlay = cv2.addWeighted(final_rot_bgr, 0.4, diff_heatmap, 0.6, 0)
    # Restore areas outside mask to show original face
    heatmap_overlay = np.where(intersection[:, :, np.newaxis] > 0, heatmap_overlay, final_rot_bgr)
    heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)
    
    # Create "hot spots" version - only show the worst asymmetries
    threshold = np.percentile(masked_values, 75) if len(masked_values) > 0 else 128
    hot_spots_mask = (diff_masked > threshold).astype(np.uint8) * 255
    # Dilate to make spots more visible
    hot_spots_mask = cv2.dilate(hot_spots_mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Create hot spots visualization: original face with red overlay on problem areas
    hot_spots_vis = final_rot.copy()
    red_overlay = np.zeros_like(final_rot)
    red_overlay[:, :] = [255, 0, 0]  # Red color
    hot_spots_vis = np.where(hot_spots_mask[:, :, np.newaxis] > 0, 
                              cv2.addWeighted(final_rot, 0.4, red_overlay, 0.6, 0),
                              final_rot)
    
    return {
        "mask": mask,
        "landmarks_debug": landmarks_debug,
        "final_rot": final_rot,
        "final_rot_with_line": final_rot_with_line,
        "final_flipped": final_flipped,
        "symmetrical_avg": symmetrical_avg,
        "diff_heatmap": diff_heatmap,
        "heatmap_overlay": heatmap_overlay,
        "hot_spots_vis": hot_spots_vis,
        "best_angle": best_angle,
        "best_shift": best_shift,
        "initial_guess": initial_guess
    }

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
            
        st.image(image, caption="Original Image", width="stretch")
        
        # Use session state to store analysis results
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
            
        # Reset if new file uploaded
        if st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.analysis_results = None
            st.session_state.last_uploaded_file = uploaded_file.name

        if st.button("Analyze Symmetry"):
            with st.spinner("Analyzing..."):
                # We run the analysis. 
                # Note: We are calling the cached function.
                # If we want to show the "Optimization Progress" live, we can't use the cached function 
                # exactly as is because the visualization callback won't run on cached hits.
                # But for the "Restart" issue, caching is the solution.
                # We will prioritize the stability of the app over the live optimization view for subsequent runs.
                
                # To show live progress on the FIRST run, we can manually run the logic here, 
                # OR we can just accept that the "Optimization Progress" view is a nice-to-have that 
                # might not show on cached re-runs.
                
                # Let's run it directly here to populate session state.
                results = run_analysis(image)
                st.session_state.analysis_results = results
                
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            if results is None:
                st.error("No face detected! Please try another image.")
            else:
                st.success(f"Optimization Complete! Best Angle: {results['best_angle']:.2f}°, Shift: {results['best_shift']:.2f} px")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Draw landmarks
                    img_debug = image.copy()
                    landmarks_debug = results['landmarks_debug']
                    if landmarks_debug:
                        cv2.circle(img_debug, landmarks_debug['left_iris'], 5, (0, 255, 0), -1)
                        cv2.circle(img_debug, landmarks_debug['right_iris'], 5, (0, 255, 0), -1)
                        cv2.circle(img_debug, landmarks_debug['nose_tip'], 5, (255, 0, 0), -1)
                        cv2.line(img_debug, landmarks_debug['left_iris'], landmarks_debug['right_iris'], (0, 255, 0), 2)
                    st.image(img_debug, caption="Detected Landmarks", width="stretch")
                
                with col2:
                    st.image(results['mask'], caption="Face Segmentation Mask", width="stretch")

                st.markdown("---")
                st.markdown("### Results")
                
                r_col1, r_col2 = st.columns(2)
                
                with r_col1:
                    st.image(results['final_rot_with_line'], caption="Aligned Original with Symmetry Axis", width="stretch")
                    
                with r_col2:
                    st.image(results['symmetrical_avg'], caption="Symmetrized Face (Average)", width="stretch")

                st.markdown("---")
                st.markdown("### 🔴 Asymmetry Analysis")
                st.markdown("""
                These visualizations highlight areas where your face differs from its mirror image. 
                **Brighter/warmer colors = greater asymmetry.**
                """)
                
                h_col1, h_col2, h_col3 = st.columns(3)
                
                with h_col1:
                    st.image(results['heatmap_overlay'], caption="Heatmap Overlay (asymmetries on face)", width="stretch")
                    
                with h_col2:
                    st.image(results['hot_spots_vis'], caption="🚨 Problem Areas (top 25% differences)", width="stretch")
                    
                with h_col3:
                    st.image(results['diff_heatmap'], caption="Pure Asymmetry Heatmap", width="stretch")

                st.markdown("### Flipped Comparison")
                st.markdown("Here is the original face next to the fully flipped version.")
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.image(results['final_rot'], caption="Original", width="stretch")
                with f_col2:
                    st.image(results['final_flipped'], caption="Flipped", width="stretch")

                st.markdown("### Rapid Flip Animation")
                st.markdown("Click the button below to rapidly toggle between the original and flipped image.")
                if st.button("Start Animation (5s)"):
                    anim_placeholder = st.empty()
                    for _ in range(25): 
                        anim_placeholder.image(results['final_rot'], caption="Original", width="stretch")
                        time.sleep(0.1)
                        anim_placeholder.image(results['final_flipped'], caption="Flipped", width="stretch")
                        time.sleep(0.1)
                    anim_placeholder.image(results['final_rot'], caption="Original", width="stretch")

if __name__ == "__main__":
    main()
