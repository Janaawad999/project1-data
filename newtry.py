import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ============= Paths - Edit according to your files =============
MODEL_PATH = "/Users/janamac/Documents/proj1/parking/parking_availiablity.h5"
MASK_PATH = "/Users/janamac/Documents/proj1/parking/mask_1920_1080.png"
VIDEO_PATH = "/Users/janamac/Documents/proj1/parking/parking_1920_1080.mp4"
OUTPUT_PATH = "/Users/janamac/Documents/proj1/parking/output_video.mp4"  # Optional


# ============= Settings =============
FRAME_SKIP = 30  # Process every 30 frames (adjust for faster/slower)
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for prediction
SAVE_VIDEO = True  # True if you want to save the output video


# ============= Load Model =============
print("🔄 Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()


# ============= Load Mask =============
print("🔄 Loading mask...")
mask = cv2.imread(MASK_PATH, 0)
if mask is None:
    print(f"❌ Could not load mask from: {MASK_PATH}")
    exit()
print("✅ Mask loaded successfully!")


# ============= Extract parking spots from Mask =============
def extract_parking_spots(mask):
    """Extract parking spot coordinates from mask"""
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    (totalLabels, label_ids, values, centroid) = connected_components
    
    spots = []
    for i in range(1, totalLabels):
        x = int(values[i, cv2.CC_STAT_LEFT])
        y = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        
        # Filter very small areas
        if w > 20 and h > 20:
            spots.append({'x': x, 'y': y, 'w': w, 'h': h, 'status': None})
    
    return spots


spots = extract_parking_spots(mask)
print(f"✅ Found {len(spots)} parking spots")


if len(spots) == 0:
    print("⚠️ No parking spots found! Check your mask file.")
    exit()


# ============= Prediction Function =============
def predict_spot(frame, spot, model):
    """Predict parking spot status"""
    x, y, w, h = spot['x'], spot['y'], spot['w'], spot['h']
    
    # Crop the area
    spot_img = frame[y:y+h, x:x+w]
    
    if spot_img.size == 0:
        return None, 0.0
    
    # Prepare image for model
    spot_img = cv2.resize(spot_img, (299, 299))
    spot_img = cv2.cvtColor(spot_img, cv2.COLOR_BGR2RGB)
    spot_img = spot_img.astype('float32') / 255.0
    spot_img = np.expand_dims(spot_img, axis=0)
    
    # Prediction
    prediction = model.predict(spot_img, verbose=0)
    probability = prediction[0][0]
    
    # Determine status
    if probability > CONFIDENCE_THRESHOLD:
        status = 'not_empty'
    else:
        status = 'empty'
    
    return status, probability


# ============= Difference calculation function (for performance) =============
def calc_diff(img1, img2):
    """Calculate difference between two images"""
    return np.abs(np.mean(img1) - np.mean(img2))


# ============= Process Video =============
print("🔄 Opening video...")
cap = cv2.VideoCapture(VIDEO_PATH)


if not cap.isOpened():
    print(f"❌ Could not open video: {VIDEO_PATH}")
    exit()


# Video information
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print(f"✅ Video: {width}x{height} @ {fps} FPS, {total_frames} frames")


# Setup video writer
writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    print(f"💾 Output will be saved to: {OUTPUT_PATH}")


# Processing variables
previous_frame = None
frame_count = 0
diffs = [None] * len(spots)


print("🚀 Processing video... Press 'q' to quit")
print("=" * 50)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every FRAME_SKIP frame
        if frame_count % FRAME_SKIP == 0:
            
            # Calculate differences if there's a previous frame
            if previous_frame is not None:
                for i, spot in enumerate(spots):
                    x, y, w, h = spot['x'], spot['y'], spot['w'], spot['h']
                    
                    if y + h <= frame.shape[0] and x + w <= frame.shape[1]:
                        current_crop = frame[y:y+h, x:x+w]
                        previous_crop = previous_frame[y:y+h, x:x+w]
                        diffs[i] = calc_diff(current_crop, previous_crop)
            
            # Determine spots to update
            if previous_frame is None:
                spots_to_update = range(len(spots))
            else:
                valid_diffs = [d for d in diffs if d is not None]
                if valid_diffs and max(valid_diffs) > 0:
                    max_diff = max(valid_diffs)
                    threshold = 0.4
                    spots_to_update = [i for i, d in enumerate(diffs) 
                                     if d is not None and d / max_diff > threshold]
                else:
                    spots_to_update = range(len(spots))
            
            # Predict for selected spots
            for i in spots_to_update:
                status, prob = predict_spot(frame, spots[i], model)
                if status:
                    spots[i]['status'] = status
                    spots[i]['probability'] = prob
            
            previous_frame = frame.copy()
            
            progress = (frame_count / total_frames) * 100
            empty_count = sum(1 for s in spots if s['status'] == 'empty')
            occupied_count = sum(1 for s in spots if s['status'] == 'not_empty')
            print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                  f"Available: {empty_count}, Occupied: {occupied_count}")
        
        # Draw results on all frames
        not_empty_count = 0
        occupied_count = 0
        
        for spot in spots:
            x, y, w, h = spot['x'], spot['y'], spot['w'], spot['h']
            status = spot['status']
            if status == 'empty':
                color = (0, 0, 255)  # Red for empty
                empty_count += 1
            elif status == 'not_empty':
                color = (0, 255, 0)  # Green for occupied
                occupied_count += 1
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add statistics on video (without parentheses)
        cv2.rectangle(frame, (10, 10), (400, 70), (0, 0, 0), -1)
        cv2.putText(frame, f'Available Spots {not_empty_count}/396',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Parking Detection', frame)
        
        if writer:
            writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Stopped by user")
            break
        
        frame_count += 1


except KeyboardInterrupt:
    print("\n⚠️ Interrupted by user")


finally:
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print("=" * 50)
    print(f"✅ Processing complete!")
    print(f"📊 Processed {frame_count} frames")
    if SAVE_VIDEO:
        print(f"💾 Video saved to: {OUTPUT_PATH}")
    print("Done! 🎉")
