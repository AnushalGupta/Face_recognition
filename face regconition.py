import cv2
from deepface import DeepFace
import concurrent.futures
import time

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream from webcam")
    exit()

# Set video frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Path to the reference image for face verification
ref_image_path = "PP.jpg"
ref_image = cv2.imread(ref_image_path)
if ref_image is None:
    print(f"Error: Could not load reference image from {ref_image_path}")
    exit()
else:
    print("Reference image loaded successfully")

# Function to check face match
def check_face(frame):
    global face_match
    try:
        # Verify the face in the frame against the reference image
        if DeepFace.verify(frame, ref_image.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        pass

# Create a ThreadPoolExecutor for managing threads efficiently
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Perform face check every 60 frames to reduce computational load
        if counter % 60 == 0:
            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            try:
                # Submit the face check task to the thread pool
                executor.submit(check_face, small_frame)
            except ValueError:
                face_match = False
        counter += 1

        # Display the result on the frame
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Show the frame in a window
        cv2.imshow("video", frame)

    # Break the loop if 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
executor.shutdown()
