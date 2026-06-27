import math

def compute_posture_metrics(landmarks):
    
   # Computes posture metrics: neck inclination, shoulder slope, and back curvature.
  
   
    # NECK INCLINATION

    l_ear = landmarks.get('left_ear', {'x': 0, 'y': 0, 'visibility': 0})
    r_ear = landmarks.get('right_ear', {'x': 0, 'y': 0, 'visibility': 0})
    l_sh = landmarks.get('left_shoulder', {'x': 0, 'y': 0, 'visibility': 0})
    r_sh = landmarks.get('right_shoulder', {'x': 0, 'y': 0, 'visibility': 0})
    
    # Determine which sides are reliably visible (threshold visibility > 0.5)
    left_visible = l_ear.get('visibility', 1.0) > 0.5 and l_sh.get('visibility', 1.0) > 0.5
    right_visible = r_ear.get('visibility', 1.0) > 0.5 and r_sh.get('visibility', 1.0) > 0.5
    
    if left_visible and right_visible:
        ear_x = (l_ear['x'] + r_ear['x']) / 2.0
        ear_y = (l_ear['y'] + r_ear['y']) / 2.0
        sh_x = (l_sh['x'] + r_sh['x']) / 2.0
        sh_y = (l_sh['y'] + r_sh['y']) / 2.0
    elif left_visible:
        ear_x, ear_y = l_ear['x'], l_ear['y']
        sh_x, sh_y = l_sh['x'], l_sh['y']
    else:
        # Fallback to right side or default safely to right side
        ear_x, ear_y = r_ear['x'], r_ear['y']
        sh_x, sh_y = r_sh['x'], r_sh['y']

    # Compute angle using atan2(|dx|, |dy|) as per sprint requirement
    dx_neck = abs(sh_x - ear_x)
    dy_neck = abs(sh_y - ear_y)
    
    # Avoid division by zero edge case
    if dy_neck == 0:
        neck_angle = 90.0
    else:
        neck_angle = math.degrees(math.atan2(dx_neck, dy_neck))

    
    # SHOULDER SLOPE PERCENTAGE
    dy_shoulder = abs(l_sh['y'] - r_sh['y'])
    dx_shoulder = abs(l_sh['x'] - r_sh['x'])
    
    if dx_shoulder == 0:
        shoulder_slope = 0.0
    else:
        shoulder_slope = (dy_shoulder / dx_shoulder) * 100

   
    # BACK CURVATURE RATIO
    nose = landmarks.get('nose', {'x': 0, 'y': 0})
    l_hip = landmarks.get('left_hip', {'x': 0, 'y': 0})
    r_hip = landmarks.get('right_hip', {'x': 0, 'y': 0})
    
    mid_shoulder_x = (l_sh['x'] + r_sh['x']) / 2.0
    mid_hip_x = (l_hip['x'] + r_hip['x']) / 2.0
    
    # Define the spine line: from shoulder_mid to nose
    # Calculate perpendicular distance from hip_mid to this line
    nose_x = nose.get('x', 0)
    nose_y = nose.get('y', 0)
    mid_shoulder_y = (l_sh['y'] + r_sh['y']) / 2.0
    mid_hip_y = (l_hip['y'] + r_hip['y']) / 2.0

    # Perpendicular distance from point (mid_hip_x, mid_hip_y) to line (mid_shoulder_x, mid_shoulder_y)→(nose_x, nose_y)
    dx_spine = nose_x - mid_shoulder_x
    dy_spine = nose_y - mid_shoulder_y
    spine_length = math.sqrt(dx_spine**2 + dy_spine**2)

    if spine_length > 0:
        back_curvature = abs((dy_spine * mid_hip_x - dx_spine * mid_hip_y + nose_x * mid_shoulder_y - nose_y * mid_shoulder_x) / spine_length)
    else:
        back_curvature = 0

    # -------------------------------------------------------------------------
    # 4. PER-FRAME VERDICT ENGINE
    # -------------------------------------------------------------------------
    reasons = []
    
    if neck_angle > 15.0:
        reasons.append(f"neck_angle={neck_angle:.1f}° (threshold: 15°)")
    if shoulder_slope > 5.0:
        reasons.append(f"shoulder_slope={shoulder_slope:.1f}% (threshold: 5%)")
    if back_curvature > 0.15:
        reasons.append(f"back_curvature={back_curvature:.2f} (threshold: 0.15)")
        
    if len(reasons) == 0:
        verdict = "GOOD"
        reason_str = "All metrics within thresholds"
    else:
        verdict = "POOR"
        reason_str = f"POOR — {', '.join(reasons)}"
        print(reason_str)  # Print bad frames explicitly as required
        
    return {
        "neck_angle": round(neck_angle, 2),
        "shoulder_slope": round(shoulder_slope, 2),
        "back_curvature": round(back_curvature, 3),
        "verdict": verdict,
        "reason": reason_str
    }