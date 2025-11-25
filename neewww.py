import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.transform import resize 

# --- تشخيص المكتبات ---
try:
    from skimage.transform import resize
except ImportError:
    print("FATAL ERROR: Library 'scikit-image' is not installed. Please install it using: pip install scikit-image")
    exit()
# ----------------------

# --- الثوابت والنموذج ---
EMPTY = True
NOT_EMPTY = False

model_path = "/Users/janamac/Documents/project1/parking/model/model.p"
try:
    MODEL = pickle.load(open(model_path, "rb"))
    print(f"✅ Model loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at: {model_path}. Please verify the path.")
    exit()

# --- الدوال (بدون تغيير) ---
def empty_or_not(spot_bgr):
    """يحدد ما إذا كانت بقعة وقوف السيارات فارغة أم مشغولة."""
    flat_data = []
    img_resized = resize(spot_bgr, (15, 15, 3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_output = MODEL.predict(flat_data)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    """يستخرج الإحداثيات (x1, y1, w, h) لمناطق وقوف السيارات من القناع."""
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots


def calc_diff(im1, im2):
    """يحسب الفرق المطلق بين متوسط سطوع صورتين."""
    return np.abs(np.mean(im1) - np.mean(im2))



# --- مسارات الملفات ---
mask_path = "/Users/janamac/Documents/project1/parking/mask_1920_1080.png" 
video_path = "/Users/janamac/Documents/project1/parking/parking_1920_1080.mp4" 

# --- تهيئة OpenCV ---
mask = cv2.imread(mask_path, 0)
if mask is None:
    print(f"FATAL ERROR: Could not load mask file from: {mask_path}. Please verify the path.")
    exit()
print(f"✅ Mask loaded successfully from: {mask_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"FATAL ERROR: Could not open video file at: {video_path}. Please verify the path.")
    exit()
print(f"✅ Video opened successfully from: {video_path}")

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

print(f"✅ Found {len(spots)} parking spots.")
if len(spots) == 0:
    print("WARNING: No parking spots found. Check your mask file.")
# ----------------------


spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None
frame_nmr = 0
ret = True
step = 30 

while ret:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            
            # فحص حدود الإطار
            if previous_frame is not None and previous_frame.shape[0] > y1 + h and previous_frame.shape[1] > x1 + w:
                diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        temp_diffs = [d if d is not None else 0.0 for d in diffs]
        print(f"Frame {frame_nmr}: Top 5 Diffs = {[round(d, 2) for d in sorted(temp_diffs, reverse=True)[:5]]}")

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            valid_diffs = [d for d in diffs if d is not None]
            if valid_diffs:
                max_diff = np.amax(valid_diffs)
                temp_diffs = [d if d is not None else 0.0 for d in diffs]
                
                if max_diff > 0:
                    arr_ = [j for j in np.argsort(temp_diffs) if temp_diffs[j] / max_diff > 0.4]
                else: 
                    arr_ = range(len(spots)) 
            else: 
                 arr_ = range(len(spots)) 

        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # --- رسم أماكن الوقوف ---
    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status is not None:
            color = (0, 255, 0) if spot_status else (0, 0, 255) 
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
    
  

    # عرض عدد الأماكن المتاحة
    available_spots_count = sum(s for s in spots_status if s is not None)
    total_spots_count = len(spots_status)
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(available_spots_count), str(total_spots_count)), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()

print("Video playback ended. Press any key to close the OpenCV window (to enable quick re-run).")
cv2.waitKey(0)