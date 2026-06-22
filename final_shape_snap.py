"""
shape_snap.py - Flagship build (Repaired)
- Fixed the Hollow Ribbon bug (uses fillPoly for solid geometric analysis)
- Fixed the Diagonal bug (uses minAreaRect for true aspect ratios)
- Scorer-based classification
- Heart/Star detection using geometry + symmetry
"""
import cv2
import numpy as np
import math
import time

# ════════════════════════════════════════════════════════════
# STROKE PREPROCESSING
# ════════════════════════════════════════════════════════════

def resample_stroke(points, spacing=6):
    if len(points) < 2: return points
    resampled = [points[0]]
    carry = 0.0
    for i in range(1, len(points)):
        p0 = np.array(points[i-1], dtype=float)
        p1 = np.array(points[i],   dtype=float)
        seg_len = np.linalg.norm(p1 - p0)
        if seg_len < 1e-6: continue
        d = carry
        while d < seg_len:
            t = d / seg_len
            pt = p0 + t * (p1 - p0)
            resampled.append(pt.astype(int).tolist())
            d += spacing
        carry = d - seg_len
    if len(resampled) < 2: return points
    return resampled

def smooth_stroke(points, window=5):
    if len(points) < window: return points
    pts = np.array(points, dtype=float)
    smoothed = []
    half = window // 2
    for i in range(len(pts)):
        lo = max(0, i - half)
        hi = min(len(pts), i + half + 1)
        smoothed.append(pts[lo:hi].mean(axis=0).astype(int).tolist())
    return smoothed

def stroke_to_contour(points, canvas_h, canvas_w):
    if len(points) < 4: return None, None

    pts = resample_stroke(points, spacing=6)
    pts = smooth_stroke(pts, window=5)
    pts_arr = np.array(pts, dtype=np.int32)

    mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    thickness = max(14, int(min(canvas_w, canvas_h) * 0.015))
    
    # CRITICAL FIX 1: Solid fill the interior so unclosed hand loops don't trick the math
    cv2.fillPoly(mask, [pts_arr], 255)
    cv2.polylines(mask, [pts_arr], False, 255, thickness)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,  3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 300: return None, None

    peri   = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    return cnt, approx

# ════════════════════════════════════════════════════════════
# GEOMETRY FEATURES
# ════════════════════════════════════════════════════════════

def _circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    return 4 * math.pi * area / (peri * peri + 1e-6)

def _solidity(cnt):
    area      = cv2.contourArea(cnt)
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    return area / (hull_area + 1e-6)

def _ellipse_ratio(cnt):
    if len(cnt) < 5: return 1.0
    try:
        _, (ma, mi), _ = cv2.fitEllipse(cnt)
        return max(ma, mi) / max(min(ma, mi), 1e-6)
    except: return 1.0

def _defect_count(cnt, min_depth_ratio=0.10):
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3: return 0
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is None: return 0
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        threshold   = min(rw, rh) * min_depth_ratio * 256
        return int(np.sum(defects[:, 0, 3] > threshold))
    except: return 0

def _bilateral_symmetry(cnt):
    try:
        pts    = cnt.reshape(-1, 2).astype(float)
        cx     = pts[:, 0].mean()
        left   = pts[pts[:, 0] < cx]
        right  = pts[pts[:, 0] >= cx]
        if len(left) < 3 or len(right) < 3: return 0.5
        left_y_range  = left[:, 1].max()  - left[:, 1].min()
        right_y_range = right[:, 1].max() - right[:, 1].min()
        range_diff = abs(left_y_range - right_y_range) / (max(left_y_range, right_y_range) + 1e-6)
        return max(0.0, 1.0 - range_diff)
    except: return 0.5

def _bottom_point_sharpness(cnt):
    try:
        pts   = cnt.reshape(-1, 2).astype(float)
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        bottom_y  = pts[:, 1].max()
        bottom_pts = pts[pts[:, 1] > bottom_y - rh * 0.1]
        x_spread  = bottom_pts[:, 0].max() - bottom_pts[:, 0].min()
        return 1.0 - min(x_spread / (rw + 1e-6), 1.0)
    except: return 0.0

def _top_notch_depth(cnt):
    try:
        pts    = cnt.reshape(-1, 2).astype(float)
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        cx     = rx + rw / 2
        top_band = pts[pts[:, 1] < ry + rh * 0.25]
        if len(top_band) < 3: return 0.0
        near_center = top_band[np.abs(top_band[:, 0] - cx) < rw * 0.2]
        if len(near_center) < 1: return 0.0
        notch_y    = near_center[:, 1].max()
        top_y      = pts[:, 1].min()
        notch_depth = (notch_y - top_y) / (rh + 1e-6)
        return min(notch_depth * 4, 1.0)
    except: return 0.0

# ════════════════════════════════════════════════════════════
# SCORER-BASED CLASSIFIER
# ════════════════════════════════════════════════════════════

def _score_circle(cnt, approx, circ, ellipse_r, defects, solidity, corners):
    s  = circ * 45
    s += max(0, (1.0 - (ellipse_r - 1.0) / 0.5)) * 20
    s += max(0, 1.0 - defects * 0.2) * 20
    s += solidity * 15
    if corners > 7: s += 10
    return min(s, 100)

def _score_ellipse(cnt, approx, circ, ellipse_r, defects, solidity, corners):
    s  = circ * 35
    s += min((ellipse_r - 1.0) / 0.5, 1.0) * 35
    s += max(0, 1.0 - defects * 0.2) * 20
    s += solidity * 10
    return min(s, 100)

def _score_heart(cnt, approx, circ, ellipse_r, defects, solidity, corners, symmetry, bottom_sharp, top_notch):
    s  = top_notch * 35
    s += bottom_sharp * 25
    s += symmetry * 20
    if defects == 1: s += 15
    elif defects == 0 or defects > 3: s -= 20
    s += (solidity - 0.6) * 20 if solidity > 0.6 else -10
    if circ > 0.85: s -= 20
    return max(min(s, 100), 0)

def _score_triangle(cnt, approx, circ, ellipse_r, defects, solidity, corners):
    s = 0
    if corners == 3: s += 60
    elif corners == 4: s += 10
    s += (1.0 - circ) * 25
    s += max(0, 1.0 - defects * 0.3) * 15
    return min(s, 100)

def _score_rectangle(cnt, approx, circ, ellipse_r, defects, solidity, corners, min_aspect):
    s = 0
    if corners == 4: s += 55
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    angles = []
    for i in range(n):
        v1 = pts[i-1] - pts[i]
        v2 = pts[(i+1)%n] - pts[i]
        cos_a = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
        angles.append(math.degrees(math.acos(np.clip(cos_a,-1,1))))
    if angles:
        right_angle_err = np.mean([abs(a-90) for a in angles])
        s += max(0, (1.0 - right_angle_err/90)) * 30
    s += (1.0 - circ) * 15
    return min(s, 100)

def _score_square(cnt, approx, circ, ellipse_r, defects, solidity, corners, min_aspect):
    base = _score_rectangle(cnt, approx, circ, ellipse_r, defects, solidity, corners, min_aspect)
    # CRITICAL FIX 2: Check aspect ratio via oriented bounding box instead of upright bounding box
    square_bonus = max(0, 1.0 - abs(min_aspect - 1.0) / 0.25) * 20
    return min(base + square_bonus, 100)

def _score_pentagon(corners, circ, solidity):
    s = max(0, (1.0 - abs(corners-5)*0.3)) * 60
    return min(s + circ * 20 + solidity * 20, 100)

def _score_hexagon(corners, circ, solidity):
    s = max(0, (1.0 - abs(corners-6)*0.3)) * 60
    return min(s + circ * 20 + solidity * 20, 100)

def _score_star(cnt, approx, defects, solidity, corners):
    s = 0
    if defects == 5: s += 50
    elif 4 <= defects <= 6: s += 25
    if solidity < 0.60: s += 30
    elif solidity < 0.70: s += 15
    if corners >= 8: s += 20
    return min(s, 100)

def _score_line(cnt, min_aspect):
    s = 0
    # CRITICAL FIX 2: Diagonal lines now resolve with accurate high aspect scores
    if min_aspect > 4:
        s += 50 + min((min_aspect-4)*5, 40)
    return min(s, 100)

CONFIDENCE_THRESHOLD = 62

def classify_shape(cnt, approx):
    try:
        # True orientation independent sizing
        rect = cv2.minAreaRect(cnt)
        min_w, min_h = rect[1]
        min_aspect = max(min_w, min_h) / max(min(min_w, min_h), 1e-6)

        corners  = len(approx)
        circ     = _circularity(cnt)
        solidity = _solidity(cnt)
        ellipse_r= _ellipse_ratio(cnt)
        defects  = _defect_count(cnt)
        symmetry = _bilateral_symmetry(cnt)
        bot_sharp= _bottom_point_sharpness(cnt)
        top_notch= _top_notch_depth(cnt)

        scores = {
            "Circle":    _score_circle(cnt,approx,circ,ellipse_r,defects,solidity,corners),
            "Ellipse":   _score_ellipse(cnt,approx,circ,ellipse_r,defects,solidity,corners),
            "Heart":     _score_heart(cnt,approx,circ,ellipse_r,defects,solidity,corners, symmetry,bot_sharp,top_notch),
            "Square":    _score_square(cnt,approx,circ,ellipse_r,defects,solidity,corners,min_aspect),
            "Rectangle": _score_rectangle(cnt,approx,circ,ellipse_r,defects,solidity,corners,min_aspect),
            "Triangle":  _score_triangle(cnt,approx,circ,ellipse_r,defects,solidity,corners),
            "Pentagon":  _score_pentagon(corners,circ,solidity),
            "Hexagon":   _score_hexagon(corners,circ,solidity),
            "Star":      _score_star(cnt,approx,defects,solidity,corners),
            "Line":      _score_line(cnt,min_aspect),
        }

        winner     = max(scores, key=scores.get)
        top_score  = scores[winner]
        confidence = top_score / 100.0

        if top_score < CONFIDENCE_THRESHOLD:
            return "Freeform", 0.0

        return winner, round(confidence, 2)
    except Exception as e:
        print(f"classify_shape error: {e}")
        return "Freeform", 0.0

# ════════════════════════════════════════════════════════════
# CLEAN SHAPE DRAWING 
# ════════════════════════════════════════════════════════════

def _draw_heart(canvas, cnt, color, thickness):
    rx,ry,rw,rh = cv2.boundingRect(cnt)
    cx = rx + rw // 2
    pts = []
    for deg in range(0, 361, 4):
        t = math.radians(deg)
        x = 16 * (math.sin(t)**3)
        y = -(13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t))
        sx = int(cx  + x * rw / 36)
        sy = int(ry + rh//2 + y * rh / 28)
        pts.append([sx, sy])
    pts_arr = np.array(pts, np.int32).reshape((-1,1,2))
    cv2.polylines(canvas, [pts_arr], True, color, thickness, cv2.LINE_AA)

def _draw_star(canvas, cnt, color, thickness):
    rx,ry,rw,rh = cv2.boundingRect(cnt)
    cx, cy = rx+rw//2, ry+rh//2
    out_r  = max(rw,rh)//2
    in_r   = int(out_r / 2.5)
    pts    = []
    for i in range(10):
        angle = i * math.pi/5 - math.pi/2
        r     = out_r if i%2==0 else in_r
        pts.append([int(cx+math.cos(angle)*r), int(cy+math.sin(angle)*r)])
    cv2.drawContours(canvas,[np.array(pts,np.int32)],0,color,thickness,cv2.LINE_AA)

def draw_clean_shape(canvas, shape_name, cnt, approx, color, thickness):
    try:
        rx,ry,rw,rh = cv2.boundingRect(cnt)
        cx,cy = rx+rw//2, ry+rh//2

        if shape_name == "Heart": _draw_heart(canvas, cnt, color, thickness)
        elif shape_name == "Star": _draw_star(canvas, cnt, color, thickness)
        elif shape_name == "Line":
            pts_f = cnt.reshape(-1,2).astype(np.float32)
            vx,vy,lx,ly = cv2.fitLine(pts_f,cv2.DIST_L2,0,0.01,0.01)
            length = max(rw,rh)//2+10
            p1=(int(lx-vx*length),int(ly-vy*length))
            p2=(int(lx+vx*length),int(ly+vy*length))
            cv2.line(canvas,p1,p2,color,thickness,cv2.LINE_AA)
        elif shape_name == "Circle":
            res    = cv2.minEnclosingCircle(cnt)
            center = (int(res[0][0]),int(res[0][1]))
            radius = int(res[1])
            cv2.circle(canvas,center,radius,color,thickness,cv2.LINE_AA)
        elif shape_name == "Ellipse":
            if len(cnt)>=5: cv2.ellipse(canvas,cv2.fitEllipse(cnt),color,thickness,cv2.LINE_AA)
            else: cv2.ellipse(canvas,(cx,cy),(rw//2,rh//2),0,0,360,color,thickness,cv2.LINE_AA)
        elif shape_name == "Square":
            side=max(rw,rh)
            sx1=cx-side//2; sy1=cy-side//2
            cv2.rectangle(canvas,(sx1,sy1),(sx1+side,sy1+side),color,thickness,cv2.LINE_AA)
        elif shape_name == "Rectangle":
            box=np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(canvas,[box],0,color,thickness,cv2.LINE_AA)
        elif shape_name == "Triangle":
            hull=cv2.convexHull(approx)
            cv2.drawContours(canvas,[hull],0,color,thickness,cv2.LINE_AA)
        else:
            hull=cv2.convexHull(cnt)
            cv2.drawContours(canvas,[hull],0,color,thickness,cv2.LINE_AA)
    except:
        cv2.drawContours(canvas,[cv2.convexHull(cnt)],0,color,thickness,cv2.LINE_AA)

# ════════════════════════════════════════════════════════════
# LAYER MANAGER & ANIMATION
# ════════════════════════════════════════════════════════════

class LayerManager:
    def __init__(self, h, w):
        self.h, self.w = h, w
        self.layer0 = np.zeros((h,w,3),np.uint8)
        self.layer1 = None

    def composite(self):
        base=self.layer0.copy()
        if self.layer1 is not None:
            gray=cv2.cvtColor(self.layer1,cv2.COLOR_BGR2GRAY)
            _,mask=cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
            mask3=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
            base=np.where(mask3>0,self.layer1,base)
        return base

    def start_snap(self,cnt,approx,shape_name,color,thickness):
        self.layer1=np.zeros((self.h,self.w,3),np.uint8)
        draw_clean_shape(self.layer1,shape_name,cnt,approx,color,thickness)

    def confirm_snap(self,stroke_points,stroke_thickness):
        if self.layer1 is None: return
        pts=np.array(stroke_points,dtype=np.int32)
        for i in range(len(pts)-1):
            cv2.line(self.layer0,tuple(pts[i]),tuple(pts[i+1]),(0,0,0),stroke_thickness+16)
        gray=cv2.cvtColor(self.layer1,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,10,255,cv2.THRESH_BINARY)
        mask3=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        self.layer0=np.where(mask3>0,self.layer1,self.layer0)
        self.layer1=None

    def cancel_snap(self): self.layer1=None
    def draw_stroke(self,p1,p2,color,thickness):
        steps=max(1,int(math.hypot(p2[0]-p1[0],p2[1]-p1[1])/3))
        for i in range(steps+1):
            t=i/steps
            cv2.circle(self.layer0,(int(p1[0]+(p2[0]-p1[0])*t),int(p1[1]+(p2[1]-p1[1])*t)),thickness//2,color,cv2.FILLED)
    def erase_stroke(self,p1,p2,thickness): self.draw_stroke(p1,p2,(0,0,0),thickness)
    def clear(self): self.layer0=np.zeros((self.h,self.w,3),np.uint8); self.layer1=None
    def copy_layer0(self): return self.layer0.copy()
    def restore_layer0(self,saved): self.layer0=saved; self.layer1=None

class SnapAnimation:
    def __init__(self,duration_ms=220):
        self.duration=duration_ms/1000.0; self.active=False
    def start(self,before,after):
        self.before=before.copy(); self.after=after.copy()
        self.start_t=time.time(); self.active=True
    def update(self,current):
        if not self.active: return current
        alpha=min((time.time()-self.start_t)/self.duration,1.0)
        if alpha>=1.0: self.active=False; return self.after.copy()
        alpha=alpha*alpha*(3-2*alpha)
        return cv2.addWeighted(self.before,1-alpha,self.after,alpha,0)
    @property
    def done(self): return not self.active

def snap_from_stroke(layers, stroke_points, color, thickness):
    try:
        if len(stroke_points) < 4: return False, "Too few points", 0.0
        cnt, approx = stroke_to_contour(stroke_points, layers.h, layers.w)
        if cnt is None: return False, "Could not build contour", 0.0
        shape_name, conf = classify_shape(cnt, approx)
        if shape_name == "Freeform": return False, "No supported shape detected", 0.0
        layers.start_snap(cnt, approx, shape_name, color, thickness)
        return True, shape_name, conf
    except Exception as e:
        print(f"snap_from_stroke error: {e}")
        return False, "Internal error", 0.0