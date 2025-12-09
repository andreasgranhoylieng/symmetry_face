"""
Detailed Facial Symmetry Scoring System

This module provides granular symmetry analysis broken down into 
anatomical categories using MediaPipe Face Mesh landmarks.

Categories analyzed:
- Eye Placement & Shape
- Eyebrow Position & Arc
- Nose Alignment
- Mouth/Lip Symmetry
- Cheek Contours
- Jaw/Chin Alignment
- Overall Face Shape

Each category returns a score from 0-100 where:
- 100 = Perfect symmetry
- 0 = Maximum asymmetry observed
"""

import numpy as np
import cv2
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


@dataclass
class VisualizationData:
    """Data for visualizing asymmetries on an image"""
    # Landmark points on left side: {name: (x, y)}
    left_landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Landmark points on right side: {name: (x, y)}
    right_landmarks: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Ideal/corrected positions: {name: (x, y)}
    ideal_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Correction arrows: [(from_point, to_point, label)]
    correction_arrows: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = field(default_factory=list)
    # Comparison lines between left/right pairs: [(left_pt, right_pt, label)]
    comparison_lines: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = field(default_factory=list)
    # The midline x coordinate
    midline_x: float = 0.0
    # Highlight color for this category (BGR)
    highlight_color: Tuple[int, int, int] = (0, 255, 255)
    # Key insight text
    key_insight: str = ""


@dataclass
class SymmetryScore:
    """Individual category symmetry score with details"""
    category: str
    score: float  # 0-100
    description: str
    left_measurements: Dict[str, float]
    right_measurements: Dict[str, float]
    asymmetry_details: str
    # New: visualization data for interactive display
    visualization: Optional[VisualizationData] = None


@dataclass
class FacialSymmetryReport:
    """Complete facial symmetry analysis report"""
    overall_score: float  # Weighted average 0-100
    category_scores: Dict[str, SymmetryScore]
    grade: str  # A, B, C, D, F
    strengths: List[str]
    areas_for_improvement: List[str]
    # New: store the original landmarks for visualization
    landmarks: Optional[List] = None
    image_dimensions: Tuple[int, int] = (0, 0)  # (width, height)
    midline_x: float = 0.0


# MediaPipe Face Mesh landmark definitions for symmetry analysis
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

LANDMARK_PAIRS = {
    # Eyes - comprehensive measurements
    "eyes": {
        "inner_corner": (133, 362),      # Inner eye corners (left, right)
        "outer_corner": (33, 263),       # Outer eye corners
        "upper_lid_center": (159, 386),  # Upper eyelid peak
        "lower_lid_center": (145, 374),  # Lower eyelid lowest
        "iris_center": (468, 473),       # Iris centers (if refine_landmarks=True)
        "upper_lid_crease": (158, 385),  # Upper eyelid crease
        "tear_duct": (173, 398),         # Near tear duct
    },
    
    # Eyebrows
    "eyebrows": {
        "inner": (107, 336),             # Inner eyebrow point
        "peak": (105, 334),              # Eyebrow peak/arch
        "outer": (70, 300),              # Outer eyebrow point
        "mid_upper": (66, 296),          # Mid upper brow
    },
    
    # Nose 
    "nose": {
        "nostril_outer": (129, 358),     # Outer nostril edges
        "nostril_bottom": (98, 327),     # Bottom of nostrils
        "alar_crease": (49, 279),        # Alar crease (nose-cheek junction)
        "bridge_side": (193, 417),       # Side of nose bridge
    },
    
    # Mouth/Lips
    "mouth": {
        "corner": (61, 291),             # Mouth corners
        "upper_lip_peak": (37, 267),     # Cupid's bow peaks
        "lower_lip_dip": (84, 314),      # Lower lip dips
        "upper_lip_outer": (40, 270),    # Upper lip outer
        "lower_lip_outer": (88, 318),    # Lower lip outer
    },
    
    # Cheeks
    "cheeks": {
        "cheekbone_high": (116, 345),    # High cheekbone
        "cheek_center": (123, 352),      # Cheek center
        "cheek_lower": (215, 435),       # Lower cheek
        "nasolabial": (92, 322),         # Nasolabial fold area
    },
    
    # Jaw/Chin
    "jaw": {
        "jaw_angle": (172, 397),         # Jaw angle
        "jaw_mid": (136, 365),           # Mid jawline
        "jaw_lower": (150, 379),         # Lower jaw
        "chin_side": (204, 424),         # Side of chin
    },
    
    # Face outline
    "face_shape": {
        "temple": (54, 284),             # Temple area
        "forehead_side": (21, 251),      # Forehead sides
        "cheek_outer": (227, 447),       # Outer cheek
        "jaw_outer": (234, 454),         # Outer jaw
    },
}

# Midline landmarks (should be centered)
MIDLINE_LANDMARKS = {
    "forehead_top": 10,
    "glabella": 9,           # Between eyebrows
    "nose_bridge_top": 168,
    "nose_bridge_mid": 6,
    "nose_tip": 1,
    "philtrum": 164,         # Above upper lip
    "upper_lip_center": 0,
    "lower_lip_center": 17,
    "chin_tip": 152,
}


class FacialSymmetryAnalyzer:
    """Analyzes facial symmetry using MediaPipe landmarks"""
    
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
    def _get_landmark_coords(self, landmarks, idx: int, w: int, h: int) -> Tuple[float, float]:
        """Extract x, y coordinates for a landmark index"""
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)
    
    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _vertical_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Vertical (Y) distance between two points"""
        return abs(p1[1] - p2[1])
    
    def _horizontal_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Horizontal (X) distance between two points"""
        return abs(p1[0] - p2[0])
    
    def _calculate_asymmetry_ratio(self, left_val: float, right_val: float) -> float:
        """
        Calculate asymmetry ratio. Returns 0 for perfect symmetry, 
        higher values for more asymmetry.
        """
        if left_val == 0 and right_val == 0:
            return 0
        avg = (left_val + right_val) / 2
        if avg == 0:
            return 0
        return abs(left_val - right_val) / avg
    
    def _asymmetry_to_score(self, asymmetry_ratio: float, 
                            sensitivity: float = 1.0) -> float:
        """
        Convert asymmetry ratio to 0-100 score.
        sensitivity adjusts how strict the scoring is.
        """
        # Using exponential decay: score = 100 * exp(-k * asymmetry)
        # k controls sensitivity (higher = more strict)
        k = 5 * sensitivity
        score = 100 * np.exp(-k * asymmetry_ratio)
        return max(0, min(100, score))
    
    def _get_face_width(self, landmarks, w: int, h: int) -> float:
        """Get interpupillary distance for normalization"""
        try:
            left_eye = self._get_landmark_coords(landmarks, 468, w, h)
            right_eye = self._get_landmark_coords(landmarks, 473, w, h)
            return self._distance(left_eye, right_eye)
        except:
            # Fallback to eye corners
            left_outer = self._get_landmark_coords(landmarks, 33, w, h)
            right_outer = self._get_landmark_coords(landmarks, 263, w, h)
            return self._distance(left_outer, right_outer)
    
    def analyze_eyes(self, landmarks, w: int, h: int, 
                     midline_x: float) -> SymmetryScore:
        """
        Analyze eye symmetry including:
        - Eye size (width and height)
        - Eye shape (aspect ratio)
        - Eye position relative to midline
        - Eyelid appearance
        """
        pairs = LANDMARK_PAIRS["eyes"]
        
        # Get coordinates
        l_inner = self._get_landmark_coords(landmarks, pairs["inner_corner"][0], w, h)
        r_inner = self._get_landmark_coords(landmarks, pairs["inner_corner"][1], w, h)
        l_outer = self._get_landmark_coords(landmarks, pairs["outer_corner"][0], w, h)
        r_outer = self._get_landmark_coords(landmarks, pairs["outer_corner"][1], w, h)
        l_upper = self._get_landmark_coords(landmarks, pairs["upper_lid_center"][0], w, h)
        r_upper = self._get_landmark_coords(landmarks, pairs["upper_lid_center"][1], w, h)
        l_lower = self._get_landmark_coords(landmarks, pairs["lower_lid_center"][0], w, h)
        r_lower = self._get_landmark_coords(landmarks, pairs["lower_lid_center"][1], w, h)
        
        # Get iris centers if available
        try:
            l_iris = self._get_landmark_coords(landmarks, pairs["iris_center"][0], w, h)
            r_iris = self._get_landmark_coords(landmarks, pairs["iris_center"][1], w, h)
        except:
            l_iris = ((l_inner[0] + l_outer[0]) / 2, (l_upper[1] + l_lower[1]) / 2)
            r_iris = ((r_inner[0] + r_outer[0]) / 2, (r_upper[1] + r_lower[1]) / 2)
        
        # Calculate measurements
        left_measurements = {
            "width": self._distance(l_inner, l_outer),
            "height": self._distance(l_upper, l_lower),
            "distance_from_midline": abs(l_inner[0] - midline_x),
            "vertical_position": (l_upper[1] + l_lower[1]) / 2,
            "outer_corner_y": l_outer[1],
            "inner_corner_y": l_inner[1],
        }
        left_measurements["aspect_ratio"] = (
            left_measurements["width"] / left_measurements["height"] 
            if left_measurements["height"] > 0 else 0
        )
        
        right_measurements = {
            "width": self._distance(r_inner, r_outer),
            "height": self._distance(r_upper, r_lower),
            "distance_from_midline": abs(r_inner[0] - midline_x),
            "vertical_position": (r_upper[1] + r_lower[1]) / 2,
            "outer_corner_y": r_outer[1],
            "inner_corner_y": r_inner[1],
        }
        right_measurements["aspect_ratio"] = (
            right_measurements["width"] / right_measurements["height"] 
            if right_measurements["height"] > 0 else 0
        )
        
        # Calculate asymmetry scores for each metric
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["width"], right_measurements["width"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["height"], right_measurements["height"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["distance_from_midline"], 
            right_measurements["distance_from_midline"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["aspect_ratio"], right_measurements["aspect_ratio"]))
        
        # Vertical position asymmetry (normalized by face width)
        face_width = self._get_face_width(landmarks, w, h)
        vert_diff = abs(left_measurements["vertical_position"] - 
                       right_measurements["vertical_position"]) / face_width
        asymmetries.append(vert_diff)
        
        # Outer corner tilt
        outer_tilt_diff = abs(left_measurements["outer_corner_y"] - 
                             right_measurements["outer_corner_y"]) / face_width
        asymmetries.append(outer_tilt_diff)
        
        # Weighted average asymmetry
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=1.2)
        
        # Generate description
        details = []
        key_insight = ""
        
        if asymmetries[0] > 0.05:
            if left_measurements["width"] > right_measurements["width"]:
                details.append("Left eye slightly wider")
                key_insight = f"Left eye is {(asymmetries[0]*100):.1f}% wider than right"
            else:
                details.append("Right eye slightly wider")
                key_insight = f"Right eye is {(asymmetries[0]*100):.1f}% wider than left"
        if asymmetries[1] > 0.05:
            if left_measurements["height"] > right_measurements["height"]:
                details.append("Left eye more open")
            else:
                details.append("Right eye more open")
        if outer_tilt_diff > 0.02:
            details.append("Slight difference in eye tilt")
            if not key_insight:
                key_insight = f"Eye tilt differs by {(outer_tilt_diff * face_width):.1f}px"
            
        asymmetry_details = "; ".join(details) if details else "Eyes are well balanced"
        if not key_insight:
            key_insight = "Eyes show excellent symmetry"
        
        # Create visualization data
        avg_y = (left_measurements["vertical_position"] + right_measurements["vertical_position"]) / 2
        avg_dist = (left_measurements["distance_from_midline"] + right_measurements["distance_from_midline"]) / 2
        
        # Calculate ideal (symmetric) positions
        ideal_l_inner = (midline_x - avg_dist, avg_y)
        ideal_r_inner = (midline_x + avg_dist, avg_y)
        
        # Create correction arrows for significant asymmetries
        correction_arrows = []
        if abs(left_measurements["vertical_position"] - right_measurements["vertical_position"]) > 2:
            # Show vertical adjustment needed
            if left_measurements["vertical_position"] > avg_y:
                correction_arrows.append((l_iris, (l_iris[0], avg_y), "Move up"))
            else:
                correction_arrows.append((r_iris, (r_iris[0], avg_y), "Move up"))
        
        if abs(left_measurements["distance_from_midline"] - right_measurements["distance_from_midline"]) > 2:
            # Show horizontal adjustment needed
            if left_measurements["distance_from_midline"] > avg_dist:
                correction_arrows.append((l_inner, ideal_l_inner, "Closer"))
            else:
                correction_arrows.append((r_inner, ideal_r_inner, "Closer"))
        
        viz = VisualizationData(
            left_landmarks={
                "inner_corner": l_inner,
                "outer_corner": l_outer,
                "upper_lid": l_upper,
                "lower_lid": l_lower,
                "iris": l_iris,
            },
            right_landmarks={
                "inner_corner": r_inner,
                "outer_corner": r_outer,
                "upper_lid": r_upper,
                "lower_lid": r_lower,
                "iris": r_iris,
            },
            ideal_positions={
                "left_inner_corner": ideal_l_inner,
                "right_inner_corner": ideal_r_inner,
            },
            correction_arrows=correction_arrows,
            comparison_lines=[
                (l_iris, r_iris, "Eye level"),
                (l_outer, r_outer, "Outer corners"),
            ],
            midline_x=midline_x,
            highlight_color=(255, 200, 0),  # Cyan-ish
            key_insight=key_insight,
        )
        
        return SymmetryScore(
            category="Eye Placement & Shape",
            score=round(score, 1),
            description="Measures eye size, shape, position, and tilt symmetry",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details,
            visualization=viz,
        )
    
    def analyze_eyebrows(self, landmarks, w: int, h: int, 
                         midline_x: float) -> SymmetryScore:
        """
        Analyze eyebrow symmetry including:
        - Brow height
        - Brow arch position  
        - Brow length
        - Brow shape
        """
        pairs = LANDMARK_PAIRS["eyebrows"]
        
        l_inner = self._get_landmark_coords(landmarks, pairs["inner"][0], w, h)
        r_inner = self._get_landmark_coords(landmarks, pairs["inner"][1], w, h)
        l_peak = self._get_landmark_coords(landmarks, pairs["peak"][0], w, h)
        r_peak = self._get_landmark_coords(landmarks, pairs["peak"][1], w, h)
        l_outer = self._get_landmark_coords(landmarks, pairs["outer"][0], w, h)
        r_outer = self._get_landmark_coords(landmarks, pairs["outer"][1], w, h)
        
        # Use eye position as reference for brow height
        l_eye_upper = self._get_landmark_coords(landmarks, 159, w, h)
        r_eye_upper = self._get_landmark_coords(landmarks, 386, w, h)
        
        left_measurements = {
            "length": self._distance(l_inner, l_outer),
            "arch_height": l_inner[1] - l_peak[1],  # Higher arch = more positive
            "inner_height_from_eye": l_eye_upper[1] - l_inner[1],
            "peak_height_from_eye": l_eye_upper[1] - l_peak[1],
            "peak_position_x": abs(l_peak[0] - l_inner[0]) / self._distance(l_inner, l_outer) if self._distance(l_inner, l_outer) > 0 else 0,
        }
        
        right_measurements = {
            "length": self._distance(r_inner, r_outer),
            "arch_height": r_inner[1] - r_peak[1],
            "inner_height_from_eye": r_eye_upper[1] - r_inner[1],
            "peak_height_from_eye": r_eye_upper[1] - r_peak[1],
            "peak_position_x": abs(r_peak[0] - r_inner[0]) / self._distance(r_inner, r_outer) if self._distance(r_inner, r_outer) > 0 else 0,
        }
        
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["length"], right_measurements["length"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            max(left_measurements["arch_height"], 1), 
            max(right_measurements["arch_height"], 1)))
        asymmetries.append(self._calculate_asymmetry_ratio(
            max(left_measurements["inner_height_from_eye"], 1),
            max(right_measurements["inner_height_from_eye"], 1)))
        asymmetries.append(self._calculate_asymmetry_ratio(
            max(left_measurements["peak_height_from_eye"], 1),
            max(right_measurements["peak_height_from_eye"], 1)))
        
        weights = [0.2, 0.3, 0.25, 0.25]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=1.0)
        
        details = []
        if asymmetries[2] > 0.08:
            if left_measurements["inner_height_from_eye"] > right_measurements["inner_height_from_eye"]:
                details.append("Left brow sits higher")
            else:
                details.append("Right brow sits higher")
        if asymmetries[1] > 0.1:
            details.append("Arch height differs between brows")
            
        asymmetry_details = "; ".join(details) if details else "Eyebrows are well matched"
        
        return SymmetryScore(
            category="Eyebrow Position & Arc",
            score=round(score, 1),
            description="Measures brow height, arch, length, and shape symmetry",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def analyze_nose(self, landmarks, w: int, h: int, 
                     midline_x: float) -> SymmetryScore:
        """
        Analyze nose symmetry including:
        - Nostril size and shape
        - Nose tip deviation from midline
        - Bridge alignment
        - Overall nose straightness
        """
        pairs = LANDMARK_PAIRS["nose"]
        
        l_nostril_out = self._get_landmark_coords(landmarks, pairs["nostril_outer"][0], w, h)
        r_nostril_out = self._get_landmark_coords(landmarks, pairs["nostril_outer"][1], w, h)
        l_nostril_bot = self._get_landmark_coords(landmarks, pairs["nostril_bottom"][0], w, h)
        r_nostril_bot = self._get_landmark_coords(landmarks, pairs["nostril_bottom"][1], w, h)
        l_alar = self._get_landmark_coords(landmarks, pairs["alar_crease"][0], w, h)
        r_alar = self._get_landmark_coords(landmarks, pairs["alar_crease"][1], w, h)
        
        nose_tip = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["nose_tip"], w, h)
        nose_bridge = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["nose_bridge_mid"], w, h)
        nose_top = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["nose_bridge_top"], w, h)
        
        face_width = self._get_face_width(landmarks, w, h)
        
        left_measurements = {
            "nostril_width": abs(l_nostril_out[0] - l_nostril_bot[0]),
            "nostril_height": abs(l_nostril_out[1] - l_nostril_bot[1]),
            "alar_distance_from_midline": abs(l_alar[0] - midline_x),
            "nostril_outer_distance": abs(l_nostril_out[0] - midline_x),
        }
        
        right_measurements = {
            "nostril_width": abs(r_nostril_out[0] - r_nostril_bot[0]),
            "nostril_height": abs(r_nostril_out[1] - r_nostril_bot[1]),
            "alar_distance_from_midline": abs(r_alar[0] - midline_x),
            "nostril_outer_distance": abs(r_nostril_out[0] - midline_x),
        }
        
        # Midline deviation measurements
        tip_deviation = abs(nose_tip[0] - midline_x) / face_width
        bridge_deviation = abs(nose_bridge[0] - midline_x) / face_width
        
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["nostril_width"], right_measurements["nostril_width"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["alar_distance_from_midline"], 
            right_measurements["alar_distance_from_midline"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["nostril_outer_distance"], 
            right_measurements["nostril_outer_distance"]))
        asymmetries.append(tip_deviation * 2)  # Scale up for sensitivity
        asymmetries.append(bridge_deviation * 2)
        
        weights = [0.2, 0.2, 0.2, 0.25, 0.15]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=1.5)
        
        details = []
        if tip_deviation > 0.02:
            if nose_tip[0] > midline_x:
                details.append("Nose tip slightly right of center")
            else:
                details.append("Nose tip slightly left of center")
        if asymmetries[0] > 0.08:
            details.append("Nostrils differ slightly in size")
            
        asymmetry_details = "; ".join(details) if details else "Nose is well centered"
        
        return SymmetryScore(
            category="Nose Alignment",
            score=round(score, 1),
            description="Measures nostril symmetry, tip centering, and bridge straightness",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def analyze_mouth(self, landmarks, w: int, h: int, 
                      midline_x: float) -> SymmetryScore:
        """
        Analyze mouth/lip symmetry including:
        - Lip fullness on each side
        - Mouth corner positions
        - Cupid's bow symmetry
        - Overall mouth centering
        """
        pairs = LANDMARK_PAIRS["mouth"]
        
        l_corner = self._get_landmark_coords(landmarks, pairs["corner"][0], w, h)
        r_corner = self._get_landmark_coords(landmarks, pairs["corner"][1], w, h)
        l_upper = self._get_landmark_coords(landmarks, pairs["upper_lip_peak"][0], w, h)
        r_upper = self._get_landmark_coords(landmarks, pairs["upper_lip_peak"][1], w, h)
        l_lower = self._get_landmark_coords(landmarks, pairs["lower_lip_dip"][0], w, h)
        r_lower = self._get_landmark_coords(landmarks, pairs["lower_lip_dip"][1], w, h)
        
        upper_lip_center = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["upper_lip_center"], w, h)
        lower_lip_center = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["lower_lip_center"], w, h)
        
        face_width = self._get_face_width(landmarks, w, h)
        
        left_measurements = {
            "corner_distance_from_midline": abs(l_corner[0] - midline_x),
            "corner_height": l_corner[1],
            "upper_lip_peak_height": upper_lip_center[1] - l_upper[1],
            "corner_to_center": self._distance(l_corner, upper_lip_center),
        }
        
        right_measurements = {
            "corner_distance_from_midline": abs(r_corner[0] - midline_x),
            "corner_height": r_corner[1],
            "upper_lip_peak_height": upper_lip_center[1] - r_upper[1],
            "corner_to_center": self._distance(r_corner, upper_lip_center),
        }
        
        # Mouth center deviation
        mouth_center_x = (l_corner[0] + r_corner[0]) / 2
        mouth_deviation = abs(mouth_center_x - midline_x) / face_width
        
        # Corner height difference (smile asymmetry)
        corner_height_diff = abs(l_corner[1] - r_corner[1]) / face_width
        
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["corner_distance_from_midline"],
            right_measurements["corner_distance_from_midline"]))
        asymmetries.append(corner_height_diff * 3)
        asymmetries.append(self._calculate_asymmetry_ratio(
            max(left_measurements["upper_lip_peak_height"], 1),
            max(right_measurements["upper_lip_peak_height"], 1)))
        asymmetries.append(mouth_deviation * 2)
        
        weights = [0.25, 0.3, 0.2, 0.25]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=1.0)
        
        details = []
        if corner_height_diff > 0.015:
            if l_corner[1] < r_corner[1]:
                details.append("Left mouth corner slightly higher")
            else:
                details.append("Right mouth corner slightly higher")
        if asymmetries[0] > 0.06:
            details.append("Mouth slightly off-center")
            
        asymmetry_details = "; ".join(details) if details else "Mouth is well balanced"
        
        return SymmetryScore(
            category="Mouth/Lip Symmetry",
            score=round(score, 1),
            description="Measures lip fullness, corner positions, and overall centering",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def analyze_cheeks(self, landmarks, w: int, h: int, 
                       midline_x: float) -> SymmetryScore:
        """
        Analyze cheek symmetry including:
        - Cheekbone prominence
        - Cheek volume distribution
        - Nasolabial fold depth/position
        """
        pairs = LANDMARK_PAIRS["cheeks"]
        
        l_high = self._get_landmark_coords(landmarks, pairs["cheekbone_high"][0], w, h)
        r_high = self._get_landmark_coords(landmarks, pairs["cheekbone_high"][1], w, h)
        l_center = self._get_landmark_coords(landmarks, pairs["cheek_center"][0], w, h)
        r_center = self._get_landmark_coords(landmarks, pairs["cheek_center"][1], w, h)
        l_lower = self._get_landmark_coords(landmarks, pairs["cheek_lower"][0], w, h)
        r_lower = self._get_landmark_coords(landmarks, pairs["cheek_lower"][1], w, h)
        l_nasolabial = self._get_landmark_coords(landmarks, pairs["nasolabial"][0], w, h)
        r_nasolabial = self._get_landmark_coords(landmarks, pairs["nasolabial"][1], w, h)
        
        nose_tip = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["nose_tip"], w, h)
        
        left_measurements = {
            "cheekbone_distance": abs(l_high[0] - midline_x),
            "cheekbone_height": l_high[1],
            "cheek_width": abs(l_center[0] - midline_x),
            "nasolabial_distance": self._distance(l_nasolabial, nose_tip),
        }
        
        right_measurements = {
            "cheekbone_distance": abs(r_high[0] - midline_x),
            "cheekbone_height": r_high[1],
            "cheek_width": abs(r_center[0] - midline_x),
            "nasolabial_distance": self._distance(r_nasolabial, nose_tip),
        }
        
        face_width = self._get_face_width(landmarks, w, h)
        cheekbone_height_diff = abs(l_high[1] - r_high[1]) / face_width
        
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["cheekbone_distance"],
            right_measurements["cheekbone_distance"]))
        asymmetries.append(cheekbone_height_diff * 2)
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["cheek_width"],
            right_measurements["cheek_width"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["nasolabial_distance"],
            right_measurements["nasolabial_distance"]))
        
        weights = [0.3, 0.25, 0.25, 0.2]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=0.8)
        
        details = []
        if asymmetries[0] > 0.05:
            if left_measurements["cheekbone_distance"] > right_measurements["cheekbone_distance"]:
                details.append("Left cheek slightly wider")
            else:
                details.append("Right cheek slightly wider")
        if cheekbone_height_diff > 0.02:
            details.append("Cheekbones at slightly different heights")
            
        asymmetry_details = "; ".join(details) if details else "Cheeks are well balanced"
        
        return SymmetryScore(
            category="Cheek Contours",
            score=round(score, 1),
            description="Measures cheekbone position, cheek volume, and contour symmetry",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def analyze_jaw(self, landmarks, w: int, h: int, 
                    midline_x: float) -> SymmetryScore:
        """
        Analyze jaw/chin symmetry including:
        - Jaw angle prominence
        - Jawline contour
        - Chin centering and shape
        """
        pairs = LANDMARK_PAIRS["jaw"]
        
        l_angle = self._get_landmark_coords(landmarks, pairs["jaw_angle"][0], w, h)
        r_angle = self._get_landmark_coords(landmarks, pairs["jaw_angle"][1], w, h)
        l_mid = self._get_landmark_coords(landmarks, pairs["jaw_mid"][0], w, h)
        r_mid = self._get_landmark_coords(landmarks, pairs["jaw_mid"][1], w, h)
        l_lower = self._get_landmark_coords(landmarks, pairs["jaw_lower"][0], w, h)
        r_lower = self._get_landmark_coords(landmarks, pairs["jaw_lower"][1], w, h)
        l_chin_side = self._get_landmark_coords(landmarks, pairs["chin_side"][0], w, h)
        r_chin_side = self._get_landmark_coords(landmarks, pairs["chin_side"][1], w, h)
        
        chin_tip = self._get_landmark_coords(landmarks, MIDLINE_LANDMARKS["chin_tip"], w, h)
        
        face_width = self._get_face_width(landmarks, w, h)
        
        left_measurements = {
            "jaw_angle_distance": abs(l_angle[0] - midline_x),
            "jaw_angle_height": l_angle[1],
            "jawline_length": self._distance(l_angle, l_mid),
            "chin_side_distance": abs(l_chin_side[0] - midline_x),
        }
        
        right_measurements = {
            "jaw_angle_distance": abs(r_angle[0] - midline_x),
            "jaw_angle_height": r_angle[1],
            "jawline_length": self._distance(r_angle, r_mid),
            "chin_side_distance": abs(r_chin_side[0] - midline_x),
        }
        
        # Chin deviation from midline
        chin_deviation = abs(chin_tip[0] - midline_x) / face_width
        
        # Jaw angle height difference
        jaw_angle_height_diff = abs(l_angle[1] - r_angle[1]) / face_width
        
        asymmetries = []
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["jaw_angle_distance"],
            right_measurements["jaw_angle_distance"]))
        asymmetries.append(jaw_angle_height_diff * 2)
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["jawline_length"],
            right_measurements["jawline_length"]))
        asymmetries.append(self._calculate_asymmetry_ratio(
            left_measurements["chin_side_distance"],
            right_measurements["chin_side_distance"]))
        asymmetries.append(chin_deviation * 3)
        
        weights = [0.25, 0.2, 0.2, 0.15, 0.2]
        avg_asymmetry = np.average(asymmetries, weights=weights)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=0.9)
        
        details = []
        if chin_deviation > 0.02:
            if chin_tip[0] > midline_x:
                details.append("Chin slightly right of center")
            else:
                details.append("Chin slightly left of center")
        if asymmetries[0] > 0.06:
            if left_measurements["jaw_angle_distance"] > right_measurements["jaw_angle_distance"]:
                details.append("Left jaw appears wider")
            else:
                details.append("Right jaw appears wider")
                
        asymmetry_details = "; ".join(details) if details else "Jaw and chin are well aligned"
        
        return SymmetryScore(
            category="Jaw/Chin Alignment",
            score=round(score, 1),
            description="Measures jaw angle, jawline contour, and chin centering",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def analyze_face_shape(self, landmarks, w: int, h: int, 
                           midline_x: float) -> SymmetryScore:
        """
        Analyze overall face shape symmetry including:
        - Face width at different levels
        - Face outline symmetry
        - Temple to jaw proportions  
        """
        pairs = LANDMARK_PAIRS["face_shape"]
        
        l_temple = self._get_landmark_coords(landmarks, pairs["temple"][0], w, h)
        r_temple = self._get_landmark_coords(landmarks, pairs["temple"][1], w, h)
        l_forehead = self._get_landmark_coords(landmarks, pairs["forehead_side"][0], w, h)
        r_forehead = self._get_landmark_coords(landmarks, pairs["forehead_side"][1], w, h)
        l_cheek_outer = self._get_landmark_coords(landmarks, pairs["cheek_outer"][0], w, h)
        r_cheek_outer = self._get_landmark_coords(landmarks, pairs["cheek_outer"][1], w, h)
        l_jaw_outer = self._get_landmark_coords(landmarks, pairs["jaw_outer"][0], w, h)
        r_jaw_outer = self._get_landmark_coords(landmarks, pairs["jaw_outer"][1], w, h)
        
        left_measurements = {
            "temple_distance": abs(l_temple[0] - midline_x),
            "forehead_distance": abs(l_forehead[0] - midline_x),
            "cheek_distance": abs(l_cheek_outer[0] - midline_x),
            "jaw_distance": abs(l_jaw_outer[0] - midline_x),
        }
        
        right_measurements = {
            "temple_distance": abs(r_temple[0] - midline_x),
            "forehead_distance": abs(r_forehead[0] - midline_x),
            "cheek_distance": abs(r_cheek_outer[0] - midline_x),
            "jaw_distance": abs(r_jaw_outer[0] - midline_x),
        }
        
        asymmetries = []
        for key in left_measurements:
            asymmetries.append(self._calculate_asymmetry_ratio(
                left_measurements[key], right_measurements[key]))
        
        avg_asymmetry = np.mean(asymmetries)
        score = self._asymmetry_to_score(avg_asymmetry, sensitivity=0.8)
        
        # Find which level has most asymmetry
        levels = ["temple", "forehead", "cheek", "jaw"]
        max_asym_idx = np.argmax(asymmetries)
        
        details = []
        if asymmetries[max_asym_idx] > 0.05:
            level = levels[max_asym_idx]
            left_val = list(left_measurements.values())[max_asym_idx]
            right_val = list(right_measurements.values())[max_asym_idx]
            if left_val > right_val:
                details.append(f"Face slightly wider on left at {level} level")
            else:
                details.append(f"Face slightly wider on right at {level} level")
                
        asymmetry_details = "; ".join(details) if details else "Face shape is well balanced"
        
        return SymmetryScore(
            category="Overall Face Shape",
            score=round(score, 1),
            description="Measures face width balance at multiple levels",
            left_measurements=left_measurements,
            right_measurements=right_measurements,
            asymmetry_details=asymmetry_details
        )
    
    def _calculate_midline_x(self, landmarks, w: int, h: int) -> float:
        """Calculate the face midline X coordinate from midline landmarks"""
        midline_x_values = []
        for name, idx in MIDLINE_LANDMARKS.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                midline_x_values.append(lm.x * w)
        return np.median(midline_x_values)
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 45:
            return "D"
        else:
            return "F"
    
    def analyze(self, image_rgb: np.ndarray) -> Optional[FacialSymmetryReport]:
        """
        Perform complete facial symmetry analysis on an image.
        
        Args:
            image_rgb: RGB image as numpy array
            
        Returns:
            FacialSymmetryReport with all category scores and overall assessment,
            or None if no face detected
        """
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = image_rgb.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate midline
        midline_x = self._calculate_midline_x(landmarks, w, h)
        
        # Analyze each category
        category_scores = {}
        
        eyes_score = self.analyze_eyes(landmarks, w, h, midline_x)
        category_scores[eyes_score.category] = eyes_score
        
        eyebrows_score = self.analyze_eyebrows(landmarks, w, h, midline_x)
        category_scores[eyebrows_score.category] = eyebrows_score
        
        nose_score = self.analyze_nose(landmarks, w, h, midline_x)
        category_scores[nose_score.category] = nose_score
        
        mouth_score = self.analyze_mouth(landmarks, w, h, midline_x)
        category_scores[mouth_score.category] = mouth_score
        
        cheeks_score = self.analyze_cheeks(landmarks, w, h, midline_x)
        category_scores[cheeks_score.category] = cheeks_score
        
        jaw_score = self.analyze_jaw(landmarks, w, h, midline_x)
        category_scores[jaw_score.category] = jaw_score
        
        face_shape_score = self.analyze_face_shape(landmarks, w, h, midline_x)
        category_scores[face_shape_score.category] = face_shape_score
        
        # Calculate weighted overall score
        # Weight more visible features higher
        weights = {
            "Eye Placement & Shape": 0.20,
            "Eyebrow Position & Arc": 0.10,
            "Nose Alignment": 0.18,
            "Mouth/Lip Symmetry": 0.17,
            "Cheek Contours": 0.10,
            "Jaw/Chin Alignment": 0.15,
            "Overall Face Shape": 0.10,
        }
        
        weighted_scores = [
            category_scores[cat].score * weights[cat] 
            for cat in weights
        ]
        overall_score = sum(weighted_scores)
        
        # Find strengths and areas for improvement
        sorted_categories = sorted(
            category_scores.items(), 
            key=lambda x: x[1].score, 
            reverse=True
        )
        
        strengths = [
            f"{cat}: {score.score:.0f}/100" 
            for cat, score in sorted_categories[:2]
            if score.score >= 70
        ]
        
        areas_for_improvement = [
            f"{cat}: {score.asymmetry_details}"
            for cat, score in sorted_categories[-2:]
            if score.score < 80 and score.asymmetry_details != cat.split()[0] + "s are well" and "well" not in score.asymmetry_details.lower()
        ]
        
        return FacialSymmetryReport(
            overall_score=round(overall_score, 1),
            category_scores=category_scores,
            grade=self._score_to_grade(overall_score),
            strengths=strengths if strengths else ["Face shows good overall balance"],
            areas_for_improvement=areas_for_improvement if areas_for_improvement else ["No significant asymmetries detected"],
            landmarks=landmarks,
            image_dimensions=(w, h),
            midline_x=midline_x,
        )


def draw_category_visualization(image: np.ndarray, 
                                 score: SymmetryScore, 
                                 midline_x: float) -> np.ndarray:
    """
    Draw interactive visualization for a specific symmetry category.
    Shows landmarks, ideal positions, and correction arrows.
    
    Args:
        image: RGB image as numpy array
        score: SymmetryScore with visualization data
        midline_x: The calculated midline x coordinate
        
    Returns:
        Annotated image as numpy array
    """
    if score.visualization is None:
        return image
    
    viz = score.visualization
    img = image.copy()
    h, w = img.shape[:2]
    
    # Define colors (RGB for display)
    LEFT_COLOR = (0, 200, 255)      # Orange for left side
    RIGHT_COLOR = (255, 200, 0)     # Cyan for right side  
    IDEAL_COLOR = (0, 255, 0)       # Green for ideal positions
    ARROW_COLOR = (255, 50, 50)     # Red for corrections
    MIDLINE_COLOR = (255, 255, 255) # White for midline
    LINE_COLOR = (200, 200, 200)    # Gray for comparison lines
    
    # Draw midline
    cv2.line(img, (int(midline_x), 0), (int(midline_x), h), MIDLINE_COLOR, 1, cv2.LINE_AA)
    
    # Draw left landmarks
    for name, pt in viz.left_landmarks.items():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, LEFT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 7, LEFT_COLOR, 1, cv2.LINE_AA)
    
    # Draw right landmarks
    for name, pt in viz.right_landmarks.items():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 5, RIGHT_COLOR, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 7, RIGHT_COLOR, 1, cv2.LINE_AA)
    
    # Draw comparison lines between pairs
    for left_pt, right_pt, label in viz.comparison_lines:
        lx, ly = int(left_pt[0]), int(left_pt[1])
        rx, ry = int(right_pt[0]), int(right_pt[1])
        
        # Draw dashed line
        cv2.line(img, (lx, ly), (rx, ry), LINE_COLOR, 1, cv2.LINE_AA)
        
        # Check if line is horizontal (symmetric)
        if abs(ly - ry) > 2:
            # Draw a horizontal reference line at average y
            avg_y = (ly + ry) // 2
            cv2.line(img, (lx, avg_y), (rx, avg_y), IDEAL_COLOR, 1, cv2.LINE_AA)
            
            # Mark the deviation
            if ly > avg_y:
                cv2.arrowedLine(img, (lx, ly), (lx, avg_y), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.3)
            else:
                cv2.arrowedLine(img, (rx, ry), (rx, avg_y), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.3)
    
    # Draw ideal position markers (hollow green circles)
    for name, pt in viz.ideal_positions.items():
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 8, IDEAL_COLOR, 2, cv2.LINE_AA)
    
    # Draw correction arrows
    for from_pt, to_pt, label in viz.correction_arrows:
        fx, fy = int(from_pt[0]), int(from_pt[1])
        tx, ty = int(to_pt[0]), int(to_pt[1])
        
        # Only draw if there's meaningful movement
        if abs(fx - tx) > 2 or abs(fy - ty) > 2:
            cv2.arrowedLine(img, (fx, fy), (tx, ty), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.3)
            
            # Add label
            mid_x = (fx + tx) // 2
            mid_y = (fy + ty) // 2
            cv2.putText(img, label, (mid_x + 5, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ARROW_COLOR, 1, cv2.LINE_AA)
    
    # Add key insight text at bottom
    if viz.key_insight:
        # Create text background
        text = viz.key_insight
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at bottom center
        text_x = (w - text_w) // 2
        text_y = h - 20
        
        # Draw background rectangle
        padding = 5
        cv2.rectangle(img, 
                     (text_x - padding, text_y - text_h - padding),
                     (text_x + text_w + padding, text_y + padding),
                     (40, 40, 40), -1)
        
        # Draw text
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    # Add legend in top left
    legend_y = 25
    cv2.circle(img, (15, legend_y), 5, LEFT_COLOR, -1)
    cv2.putText(img, "Left", (25, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, LEFT_COLOR, 1)
    
    cv2.circle(img, (65, legend_y), 5, RIGHT_COLOR, -1)
    cv2.putText(img, "Right", (75, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RIGHT_COLOR, 1)
    
    cv2.circle(img, (120, legend_y), 5, IDEAL_COLOR, 2)
    cv2.putText(img, "Ideal", (130, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, IDEAL_COLOR, 1)
    
    return img


def create_symmetry_visualization(report: FacialSymmetryReport) -> Dict:
    """
    Create data structure for visualizing the symmetry report.
    Can be used with Streamlit to create nice visualizations.
    """
    return {
        "overall": {
            "score": report.overall_score,
            "grade": report.grade,
        },
        "categories": {
            cat: {
                "score": score.score,
                "description": score.description,
                "details": score.asymmetry_details,
            }
            for cat, score in report.category_scores.items()
        },
        "strengths": report.strengths,
        "improvements": report.areas_for_improvement,
    }

