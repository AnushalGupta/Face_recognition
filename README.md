# Face_recognition
A basic simple python program for face recognition using image as a reference 

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

## Installation

1. Install OpenCV:
    ```
    pip install opencv-python
    ```

2. Install DeepFace:
    ```
    pip install deepface
    ```

3. Install tf-keras:
    ```
    pip install tf-keras
    ```

4. (Optional) If you encounter issues with TensorFlow, install a compatible version:
    ```
    pip install tensorflow==2.17.1


### Explanation of Key Parts

1. **Initialization and Setup:**
   - `cv2.VideoCapture(0)`: Initializes video capture from the webcam.
   - `cap.set(...)`: Sets the resolution of the video frames.
   - `ref_image_path` and `cv2.imread(...)`: Loads the reference image for face verification.

2. **Face Verification Function:**
   - `check_face(frame)`: Verifies if the face in the given frame matches the reference image using DeepFace.

3. **Main Loop:**
   - `while True`: Continuously captures frames from the webcam.
   - `if counter % 60 == 0`: Checks face match every 60 frames.
   - `cv2.resize(...)`: Resizes the frame for faster processing.
   - `executor.submit(...)`: Submits the face verification task to the thread pool.
   - `cv2.putText(...)`: Displays the result ("MATCH!" or "NO MATCH!") on the video frame.
   - `cv2.imshow("video", frame)`: Shows the video frame in a window.
   - `if key == ord("q")`: Breaks the loop if the 'q' key is pressed.

4. **Cleanup:**
   - `cap.release()`: Releases the video capture object.
   - `cv2.destroyAllWindows()`: Closes all OpenCV windows.
   - `executor.shutdown()`: Shuts down the thread pool executor.

These comments should help you understand the code more easily. If you have any further questions, feel free to ask!
##Fixes:
1> Installing Deepface : try installing tensorflow first before installing Deepface 
2> Make sure your reference image is captured in good lighting and proper face is shown , does not recogize if face is shown sideways 
3> lagging or low performance try increasing the value of "counter % 60" ,here every 60th frame of your webcame is used to match and check with the refernce image 
