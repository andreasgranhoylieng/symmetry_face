import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from scipy.optimize import minimize
from PIL import Image
import time
from symmetry_scores import FacialSymmetryAnalyzer, FacialSymmetryReport

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
    
    ADVANCED SYMMETRY AXIS ESTIMATION:
    Uses a multi-stage approach for finding the perfect symmetry line:
    1. PCA-based initial angle from symmetric pair midpoints
    2. RANSAC for robust midline fitting with outlier rejection
    3. 3D depth compensation using landmark z-coordinates
    4. Weighted ensemble from multiple estimation methods
    5. Cross-validation between methods for confidence scoring
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
    
    # ============================================================
    # ADVANCED SYMMETRY AXIS ESTIMATION
    # ============================================================
    
    # Extended midline landmarks for better coverage
    MIDLINE_LANDMARKS = {
        'forehead_top': 10,      # Top of forehead
        'forehead_mid': 151,     # Mid forehead
        'glabella': 9,           # Between eyebrows (glabella)
        'nose_bridge_top': 168,  # Top of nose bridge
        'nose_bridge_upper': 197, # Upper nose bridge
        'nose_bridge_mid': 6,    # Middle of nose bridge
        'nose_tip': 1,           # Tip of nose
        'nose_bottom': 2,        # Bottom of nose
        'philtrum_top': 164,     # Philtrum top
        'philtrum_bottom': 167,  # Philtrum bottom
        'upper_lip': 0,          # Center of upper lip
        'lower_lip': 17,         # Center of lower lip
        'chin_upper': 18,        # Upper chin
        'chin_tip': 152,         # Chin tip
        'chin_bottom': 175,      # Lower chin
    }
    
    # Extended symmetric pairs with anatomical importance weights
    SYMMETRIC_PAIRS = [
        # Most reliable - Eyes
        (468, 473, 5.0, 'iris'),           # Irises (highest reliability)
        (33, 263, 4.0, 'inner_eye'),       # Inner eye corners
        (133, 362, 3.5, 'outer_eye'),      # Outer eye corners
        (159, 386, 3.0, 'upper_lid'),      # Upper eyelid
        (145, 374, 3.0, 'lower_lid'),      # Lower eyelid
        
        # Very reliable - Eyebrows
        (107, 336, 2.5, 'inner_brow'),     # Inner eyebrow
        (105, 334, 2.5, 'brow_peak'),      # Eyebrow peak
        (70, 300, 2.0, 'outer_brow'),      # Outer eyebrow
        
        # Reliable - Nose
        (129, 358, 2.5, 'nostril_outer'),  # Outer nostril
        (98, 327, 2.0, 'nostril_bottom'),  # Bottom nostril
        (49, 279, 2.0, 'alar'),            # Alar crease
        
        # Reliable - Mouth
        (61, 291, 3.0, 'mouth_corner'),    # Mouth corners
        (37, 267, 2.0, 'cupid_bow'),       # Cupid's bow peaks
        (78, 308, 1.5, 'upper_lip'),       # Upper lip sides
        (95, 324, 1.5, 'lower_lip'),       # Lower lip sides
        
        # Moderate reliability - Cheeks/Jaw
        (116, 345, 1.5, 'cheekbone'),      # Cheekbone
        (123, 352, 1.5, 'cheek'),          # Cheek
        (172, 397, 2.0, 'jaw_angle'),      # Jaw angle
        (136, 365, 1.5, 'jaw_mid'),        # Mid jaw
        (234, 454, 1.0, 'temple'),         # Temple
        (127, 356, 1.5, 'face_side'),      # Face side
    ]
    
    try:
        # ============================================================
        # STAGE 1: Collect all landmark data with 3D coordinates
        # ============================================================
        
        midline_points_3d = []
        for name, idx in MIDLINE_LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                midline_points_3d.append({
                    'name': name,
                    'x': lm.x * w,
                    'y': lm.y * h,
                    'z': lm.z * w,  # Z is relative depth, scale by width
                    'visibility': getattr(lm, 'visibility', 1.0) if hasattr(lm, 'visibility') else 1.0
                })
        
        # Collect symmetric pair data
        pair_data = []
        for pair in SYMMETRIC_PAIRS:
            left_idx, right_idx = pair[0], pair[1]
            weight = pair[2] if len(pair) > 2 else 1.0
            name = pair[3] if len(pair) > 3 else 'unknown'
            
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_lm = landmarks[left_idx]
                right_lm = landmarks[right_idx]
                
                pair_data.append({
                    'name': name,
                    'left': (left_lm.x * w, left_lm.y * h, left_lm.z * w),
                    'right': (right_lm.x * w, right_lm.y * h, right_lm.z * w),
                    'weight': weight,
                    'midpoint': ((left_lm.x + right_lm.x) / 2 * w, 
                                (left_lm.y + right_lm.y) / 2 * h),
                })
        
        # ============================================================
        # STAGE 2: PCA-based angle estimation from midpoints
        # ============================================================
        
        # Collect all points that should lie on the symmetry axis
        axis_candidate_points = []
        axis_weights = []
        
        # Add midline points
        for pt in midline_points_3d:
            axis_candidate_points.append([pt['x'], pt['y']])
            # Weight by position (nose area most reliable)
            name_weights = {
                'nose_bridge_top': 3.0, 'nose_bridge_upper': 2.5,
                'nose_bridge_mid': 3.0, 'nose_tip': 2.5,
                'philtrum_top': 2.0, 'philtrum_bottom': 2.0,
                'glabella': 1.5, 'forehead_top': 0.8, 'forehead_mid': 0.8,
                'upper_lip': 1.5, 'lower_lip': 1.2,
                'chin_upper': 1.0, 'chin_tip': 1.0, 'chin_bottom': 0.8,
            }
            axis_weights.append(name_weights.get(pt['name'], 1.0))
        
        # Add symmetric pair midpoints
        for pair in pair_data:
            axis_candidate_points.append(list(pair['midpoint']))
            axis_weights.append(pair['weight'])
        
        axis_candidate_points = np.array(axis_candidate_points)
        axis_weights = np.array(axis_weights)
        
        # Weighted PCA for initial angle estimation
        weighted_center = np.average(axis_candidate_points, axis=0, weights=axis_weights)
        centered_points = axis_candidate_points - weighted_center
        
        # Weighted covariance matrix
        weighted_cov = np.zeros((2, 2))
        total_weight = np.sum(axis_weights)
        for i, (pt, wt) in enumerate(zip(centered_points, axis_weights)):
            weighted_cov += wt * np.outer(pt, pt)
        weighted_cov /= total_weight
        
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)
        # Principal component is the SECOND eigenvector (we want axis direction)
        # For a vertical line, we want the eigenvector with smaller variance
        principal_axis = eigenvectors[:, 0]  # Smaller eigenvalue = tighter fit
        
        # Calculate angle from principal axis
        pca_angle = np.degrees(np.arctan2(principal_axis[0], principal_axis[1]))
        # Adjust to be close to 0 (vertical line should have ~0 angle)
        if abs(pca_angle) > 45:
            pca_angle = pca_angle - 90 if pca_angle > 0 else pca_angle + 90
        
        # ============================================================
        # STAGE 3: Angle estimation from symmetric pairs with RANSAC
        # ============================================================
        
        angles_from_pairs = []
        angle_pair_weights = []
        
        for pair in pair_data:
            lx, ly, lz = pair['left']
            rx, ry, rz = pair['right']
            
            dx = rx - lx
            dy = ry - ly
            dz = rz - lz
            
            if abs(dx) > 5:  # Avoid nearly coincident points
                # 3D depth compensation: if one eye is closer, it appears larger
                # Adjust for perspective distortion
                depth_ratio = 1.0
                avg_z = (lz + rz) / 2
                if abs(dz) > 0.5 and abs(avg_z) > 0.1:
                    # Points at different depths - compensate for perspective
                    depth_ratio = 1.0 + 0.1 * (dz / w)  # Subtle adjustment
                
                pair_angle = np.degrees(np.arctan2(dy * depth_ratio, dx))
                pair_distance = np.sqrt(dx*dx + dy*dy)
                
                angles_from_pairs.append(pair_angle)
                angle_pair_weights.append(pair['weight'] * pair_distance)
        
        angles_from_pairs = np.array(angles_from_pairs)
        angle_pair_weights = np.array(angle_pair_weights)
        
        # RANSAC-style robust angle estimation
        if len(angles_from_pairs) >= 5:
            best_inliers = 0
            best_angle = np.average(angles_from_pairs, weights=angle_pair_weights)
            
            # Multiple RANSAC iterations
            for _ in range(50):
                # Randomly sample 3 pairs
                sample_idx = np.random.choice(len(angles_from_pairs), size=min(3, len(angles_from_pairs)), replace=False)
                sample_angles = angles_from_pairs[sample_idx]
                sample_weights = angle_pair_weights[sample_idx]
                
                # Estimate angle from sample
                candidate_angle = np.average(sample_angles, weights=sample_weights)
                
                # Count inliers (within 2 degrees)
                residuals = np.abs(angles_from_pairs - candidate_angle)
                inlier_mask = residuals < 2.0
                num_inliers = np.sum(inlier_mask * angle_pair_weights)
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    # Refit using all inliers
                    if np.sum(inlier_mask) >= 3:
                        best_angle = np.average(
                            angles_from_pairs[inlier_mask], 
                            weights=angle_pair_weights[inlier_mask]
                        )
            
            ransac_angle = best_angle
        else:
            ransac_angle = np.average(angles_from_pairs, weights=angle_pair_weights) if len(angles_from_pairs) > 0 else 0.0
        
        # ============================================================
        # STAGE 4: Combine angle estimates with confidence weighting
        # ============================================================
        
        # Check agreement between PCA and RANSAC
        angle_agreement = abs(pca_angle - ransac_angle)
        
        if angle_agreement < 1.0:
            # High agreement - use weighted average
            initial_angle = 0.6 * ransac_angle + 0.4 * pca_angle
        elif angle_agreement < 3.0:
            # Moderate agreement - favor RANSAC (more robust to outliers)
            initial_angle = 0.75 * ransac_angle + 0.25 * pca_angle
        else:
            # Disagreement - use RANSAC as it's more robust
            initial_angle = ransac_angle
        
        # ============================================================
        # STAGE 5: Precise axis position estimation with RANSAC
        # ============================================================
        
        # Rotate all candidate points by the estimated angle
        M_rot = cv2.getRotationMatrix2D((center_x, center_y), initial_angle, 1.0)
        
        rotated_axis_x = []
        rotated_axis_weights = []
        
        # Rotate and collect midline points
        for pt in midline_points_3d:
            original_pt = np.array([pt['x'], pt['y'], 1.0])
            rotated_pt = M_rot @ original_pt
            rotated_axis_x.append(rotated_pt[0])
            
            name_weights = {
                'nose_bridge_top': 3.0, 'nose_bridge_upper': 2.5,
                'nose_bridge_mid': 3.0, 'nose_tip': 2.5,
                'philtrum_top': 2.0, 'philtrum_bottom': 2.0,
                'glabella': 1.5, 'upper_lip': 1.5,
            }
            rotated_axis_weights.append(name_weights.get(pt['name'], 1.0))
        
        # Rotate and collect pair midpoints
        for pair in pair_data:
            mid_x, mid_y = pair['midpoint']
            original_pt = np.array([mid_x, mid_y, 1.0])
            rotated_pt = M_rot @ original_pt
            rotated_axis_x.append(rotated_pt[0])
            rotated_axis_weights.append(pair['weight'])
        
        rotated_axis_x = np.array(rotated_axis_x)
        rotated_axis_weights = np.array(rotated_axis_weights)
        
        # RANSAC for axis position
        if len(rotated_axis_x) >= 5:
            best_inliers = 0
            best_axis_x = np.average(rotated_axis_x, weights=rotated_axis_weights)
            
            for _ in range(50):
                # Random sample
                sample_idx = np.random.choice(len(rotated_axis_x), size=min(3, len(rotated_axis_x)), replace=False)
                candidate_x = np.average(rotated_axis_x[sample_idx], weights=rotated_axis_weights[sample_idx])
                
                # Count inliers (within 3 pixels)
                residuals = np.abs(rotated_axis_x - candidate_x)
                inlier_mask = residuals < 3.0
                num_inliers = np.sum(inlier_mask * rotated_axis_weights)
                
                if num_inliers > best_inliers:
                    best_inliers = num_inliers
                    if np.sum(inlier_mask) >= 3:
                        best_axis_x = np.average(
                            rotated_axis_x[inlier_mask], 
                            weights=rotated_axis_weights[inlier_mask]
                        )
            
            final_axis_x = best_axis_x
        else:
            final_axis_x = np.average(rotated_axis_x, weights=rotated_axis_weights)
        
        initial_shift = final_axis_x - center_x
        
        # ============================================================
        # STAGE 6: Fine-tune using geometric constraints
        # ============================================================
        
        # Use iris centers as the most reliable reference if available
        try:
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            
            iris_mid_x = (left_iris.x + right_iris.x) / 2 * w
            iris_mid_y = (left_iris.y + right_iris.y) / 2 * h
            
            # Rotate iris midpoint
            iris_pt = np.array([iris_mid_x, iris_mid_y, 1.0])
            rotated_iris = M_rot @ iris_pt
            
            # Iris midpoint should be VERY close to the axis
            iris_axis_diff = rotated_iris[0] - final_axis_x
            
            # If iris suggests different position, blend slightly towards it
            if abs(iris_axis_diff) < 10:  # Sanity check
                initial_shift += iris_axis_diff * 0.3  # 30% correction towards iris
            
            # Similarly refine angle using iris pair
            iris_dx = right_iris.x * w - left_iris.x * w
            iris_dy = right_iris.y * h - left_iris.y * h
            iris_angle = np.degrees(np.arctan2(iris_dy, iris_dx))
            
            angle_diff = iris_angle - initial_angle
            if abs(angle_diff) < 3:  # Sanity check
                initial_angle += angle_diff * 0.2  # 20% correction towards iris angle
                
        except (IndexError, AttributeError):
            pass
        
        # Store enhanced debug info
        try:
            left_iris = landmarks[468]
            right_iris = landmarks[473]
            nose_tip = landmarks[1]
            
            landmarks_debug['left_iris'] = (int(left_iris.x * w), int(left_iris.y * h))
            landmarks_debug['right_iris'] = (int(right_iris.x * w), int(right_iris.y * h))
            landmarks_debug['nose_tip'] = (int(nose_tip.x * w), int(nose_tip.y * h))
            
            # Add midline debug visualization
            landmarks_debug['midline_points'] = [
                (int(pt['x']), int(pt['y'])) for pt in midline_points_3d
            ]
            
            # Add pair midpoints for debugging
            landmarks_debug['pair_midpoints'] = [
                (int(pair['midpoint'][0]), int(pair['midpoint'][1])) for pair in pair_data
            ]
            
            # Store estimation confidence info
            landmarks_debug['angle_agreement'] = angle_agreement
            landmarks_debug['pca_angle'] = pca_angle
            landmarks_debug['ransac_angle'] = ransac_angle
            
        except (IndexError, AttributeError):
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
    
    # ============================================================
    # ENHANCED 3-STAGE MULTI-RESOLUTION OPTIMIZATION
    # For sub-pixel accuracy in symmetry line detection
    # ============================================================
    
    # Stage 1: Very coarse search at 0.25x resolution (fast global search)
    scale_1 = 0.25
    tiny_gray = cv2.resize(image_gray, None, fx=scale_1, fy=scale_1)
    tiny_mask = cv2.resize(mask, None, fx=scale_1, fy=scale_1, interpolation=cv2.INTER_NEAREST)
    tiny_edges = cv2.resize(edge_weights, None, fx=scale_1, fy=scale_1)
    tiny_center = (center_guess[0] * scale_1, center_guess[1] * scale_1)
    
    def loss_scale_1(params):
        scaled_params = [params[0], params[1] * scale_1]
        return transform_and_compare(
            scaled_params, tiny_gray, tiny_mask, tiny_center, tiny_edges
        )
    
    # Stage 2: Coarse search at 0.5x resolution
    scale_2 = 0.5
    small_gray = cv2.resize(image_gray, None, fx=scale_2, fy=scale_2)
    small_mask = cv2.resize(mask, None, fx=scale_2, fy=scale_2, interpolation=cv2.INTER_NEAREST)
    small_edges = cv2.resize(edge_weights, None, fx=scale_2, fy=scale_2)
    small_center = (center_guess[0] * scale_2, center_guess[1] * scale_2)
    
    def loss_scale_2(params):
        scaled_params = [params[0], params[1] * scale_2]
        return transform_and_compare(
            scaled_params, small_gray, small_mask, small_center, small_edges
        )
    
    # Stage 3: Fine search at full resolution
    def loss_full(params):
        return transform_and_compare(
            params, image_gray, mask, center_guess, edge_weights
        )
    
    from scipy.optimize import differential_evolution, minimize
    
    # ============================================================
    # STAGE 1: Global search at 0.25x (wide exploration)
    # ============================================================
    stage1_bounds = [
        (initial_guess[0] - 20, initial_guess[0] + 20),  # ±20° angle
        (initial_guess[1] - 60, initial_guess[1] + 60)   # ±60px shift
    ]
    
    try:
        # Use differential evolution for global search
        result_1 = differential_evolution(
            loss_scale_1, 
            stage1_bounds, 
            maxiter=50,      # More iterations for thorough search
            popsize=20,      # Larger population for better coverage
            tol=0.05,
            seed=42,
            workers=1,
            polish=False,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        stage1_angle, stage1_shift = result_1.x
    except Exception:
        stage1_angle, stage1_shift = initial_guess
    
    # ============================================================
    # STAGE 2: Refined search at 0.5x (narrower bounds)
    # ============================================================
    stage2_bounds = [
        (stage1_angle - 8, stage1_angle + 8),   # ±8° around stage 1
        (stage1_shift - 25, stage1_shift + 25)  # ±25px around stage 1
    ]
    
    try:
        result_2 = differential_evolution(
            loss_scale_2, 
            stage2_bounds, 
            maxiter=40,
            popsize=15,
            tol=0.02,
            seed=42,
            workers=1,
            polish=True,  # Enable polishing for better local minimum
            mutation=(0.4, 0.9),
            recombination=0.8
        )
        stage2_angle, stage2_shift = result_2.x
    except Exception:
        stage2_angle, stage2_shift = stage1_angle, stage1_shift
    
    # ============================================================
    # STAGE 3: Fine optimization at full resolution
    # ============================================================
    
    # 3a: Nelder-Mead for robust local search
    fine_result = minimize(
        loss_full, 
        [stage2_angle, stage2_shift], 
        method='Nelder-Mead',
        options={
            'xatol': 0.005,   # High angle precision (0.005 degrees)
            'fatol': 0.05,    # Low function tolerance
            'maxiter': 200,   # More iterations
            'adaptive': True  # Adaptive algorithm for better convergence
        }
    )
    stage3_angle, stage3_shift = fine_result.x
    
    # 3b: BFGS refinement for gradient-based polishing
    try:
        # Use BFGS for final sub-pixel refinement
        bfgs_result = minimize(
            loss_full, 
            [stage3_angle, stage3_shift], 
            method='L-BFGS-B',
            bounds=[
                (stage3_angle - 0.5, stage3_angle + 0.5),  # ±0.5° very tight
                (stage3_shift - 3, stage3_shift + 3)       # ±3px very tight
            ],
            options={
                'ftol': 1e-8,    # Very high precision
                'gtol': 1e-7,
                'maxiter': 100
            }
        )
        best_angle, best_shift = bfgs_result.x
    except Exception:
        best_angle, best_shift = stage3_angle, stage3_shift
    
    # ============================================================
    # FINAL VALIDATION: Ensure solution is better than initial
    # ============================================================
    initial_loss = loss_full(initial_guess)
    final_loss = loss_full([best_angle, best_shift])
    
    # If optimization made things worse, fall back to initial guess
    if final_loss > initial_loss * 1.1:  # Allow 10% tolerance
        best_angle, best_shift = initial_guess

    
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
    
    return {
        "mask": mask,
        "landmarks_debug": landmarks_debug,
        "final_rot": final_rot,
        "final_rot_with_line": final_rot_with_line,
        "final_flipped": final_flipped,
        "symmetrical_avg": symmetrical_avg,
        "diff_heatmap": diff_heatmap,
        "heatmap_overlay": heatmap_overlay,
        "diff_masked": diff_masked,
        "intersection": intersection,
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
                    # Generate hot spots visualization with adjustable threshold
                    threshold_pct = st.slider(
                        "Threshold (% of differences to highlight)",
                        min_value=0,
                        max_value=100,
                        value=25,
                        help="Lower values highlight more areas, higher values show only the most prominent asymmetries"
                    )
                    
                    # Calculate threshold from percentile (inverting: 25% means top 25%, so 75th percentile)
                    diff_masked = results['diff_masked']
                    intersection = results['intersection']
                    masked_values = diff_masked[intersection > 0]
                    
                    percentile = 100 - threshold_pct  # Convert to percentile (25% -> 75th percentile)
                    threshold_val = np.percentile(masked_values, percentile) if len(masked_values) > 0 else 128
                    hot_spots_mask = (diff_masked > threshold_val).astype(np.uint8) * 255
                    # Dilate to make spots more visible
                    hot_spots_mask = cv2.dilate(hot_spots_mask, np.ones((5, 5), np.uint8), iterations=1)
                    
                    # Create hot spots visualization: original face with red overlay on problem areas
                    final_rot = results['final_rot']
                    hot_spots_vis = final_rot.copy()
                    red_overlay = np.zeros_like(final_rot)
                    red_overlay[:, :] = [255, 0, 0]  # Red color
                    hot_spots_vis = np.where(hot_spots_mask[:, :, np.newaxis] > 0, 
                                              cv2.addWeighted(final_rot, 0.4, red_overlay, 0.6, 0),
                                              final_rot)
                    
                    st.image(hot_spots_vis, caption=f"🚨 Problem Areas (top {threshold_pct}% differences)", width="stretch")
                    
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

                # Detailed Symmetry Scores Section
                st.markdown("---")
                st.markdown("### 📊 Detailed Symmetry Scores")
                st.markdown("""
                This analysis breaks down facial symmetry into specific anatomical categories.
                Each score is from 0-100 where **100 = perfect symmetry**.
                """)
                
                # Run the detailed symmetry analysis
                analyzer = FacialSymmetryAnalyzer()
                symmetry_report = analyzer.analyze(image)
                
                if symmetry_report:
                    # Overall score display with grade
                    grade_colors = {
                        "A+": "#00C853", "A": "#00E676", "A-": "#69F0AE",
                        "B+": "#76FF03", "B": "#C6FF00", "B-": "#EEFF41",
                        "C+": "#FFEA00", "C": "#FFC400", "C-": "#FF9100",
                        "D": "#FF6D00", "F": "#FF3D00"
                    }
                    grade_color = grade_colors.get(symmetry_report.grade, "#FFFFFF")
                    
                    overall_col1, overall_col2, overall_col3 = st.columns([1, 2, 1])
                    with overall_col2:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; border: 2px solid {grade_color};">
                            <h2 style="margin: 0; color: #ffffff;">Overall Symmetry Score</h2>
                            <h1 style="font-size: 4em; margin: 10px 0; color: {grade_color};">{symmetry_report.overall_score:.1f}</h1>
                            <h2 style="margin: 0; padding: 10px 20px; background: {grade_color}; border-radius: 10px; display: inline-block; color: #000000;">Grade: {symmetry_report.grade}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    # Category breakdown with progress bars
                    st.markdown("#### Category Breakdown")
                    
                    # Sort categories by score (highest first)
                    sorted_categories = sorted(
                        symmetry_report.category_scores.items(),
                        key=lambda x: x[1].score,
                        reverse=True
                    )
                    
                    for category, score_data in sorted_categories:
                        # Determine color based on score
                        if score_data.score >= 85:
                            bar_color = "#00C853"  # Green
                            emoji = "✅"
                        elif score_data.score >= 70:
                            bar_color = "#76FF03"  # Light green
                            emoji = "👍"
                        elif score_data.score >= 55:
                            bar_color = "#FFEA00"  # Yellow
                            emoji = "⚠️"
                        else:
                            bar_color = "#FF6D00"  # Orange
                            emoji = "📍"
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            # Make the category an expander button for interactive visualization
                            with st.expander(f"{emoji} {category} — Score: {score_data.score:.0f}/100", expanded=False):
                                # Show the asymmetry details
                                st.markdown(f"**Analysis:** {score_data.asymmetry_details}")
                                st.markdown(f"*{score_data.description}*")
                                
                                # Show visualization if available
                                if score_data.visualization is not None:
                                    from symmetry_scores import draw_category_visualization
                                    
                                    # Draw the annotated image
                                    viz_image = draw_category_visualization(
                                        image, 
                                        score_data, 
                                        symmetry_report.midline_x
                                    )
                                    
                                    st.image(viz_image, caption=f"{category} Visualization", use_container_width=True)
                                    
                                    # Show key insight
                                    if score_data.visualization.key_insight:
                                        if score_data.score >= 85:
                                            st.success(f"💡 {score_data.visualization.key_insight}")
                                        elif score_data.score >= 70:
                                            st.info(f"� {score_data.visualization.key_insight}")
                                        else:
                                            st.warning(f"💡 {score_data.visualization.key_insight}")
                                    
                                    # Show legend
                                    st.markdown("""
                                    **Legend:**
                                    - 🟠 Orange dots = Left side landmarks
                                    - 🔵 Cyan dots = Right side landmarks  
                                    - 🟢 Green circles = Ideal (symmetric) positions
                                    - 🔴 Red arrows = Suggested corrections
                                    - ⚪ White line = Symmetry axis (midline)
                                    """)
                                else:
                                    st.info("Visualization data not available for this category.")
                                
                                # Show detailed measurements
                                st.markdown("---")
                                st.markdown("**Measurements:**")
                                meas_col1, meas_col2 = st.columns(2)
                                
                                with meas_col1:
                                    st.markdown("*Left Side:*")
                                    for key, value in score_data.left_measurements.items():
                                        formatted_key = key.replace("_", " ").title()
                                        st.text(f"  {formatted_key}: {value:.1f}")
                                
                                with meas_col2:
                                    st.markdown("*Right Side:*")
                                    for key, value in score_data.right_measurements.items():
                                        formatted_key = key.replace("_", " ").title()
                                        st.text(f"  {formatted_key}: {value:.1f}")
                        
                        with col2:
                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background: {bar_color}20; border-radius: 10px; border: 1px solid {bar_color};">
                                <span style="font-size: 1.5em; font-weight: bold; color: {bar_color};">{score_data.score:.0f}</span>
                                <span style="color: #888;">/100</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("")
                    
                    # Strengths and Areas for Improvement
                    st.markdown("---")
                    str_col, imp_col = st.columns(2)
                    
                    with str_col:
                        st.markdown("#### 💪 Strengths")
                        for strength in symmetry_report.strengths:
                            st.markdown(f"- {strength}")
                    
                    with imp_col:
                        st.markdown("#### 🎯 Areas of Note")
                        for area in symmetry_report.areas_for_improvement:
                            st.markdown(f"- {area}")
                else:
                    st.warning("Could not generate detailed symmetry scores. Face detection may have failed.")

if __name__ == "__main__":
    main()
