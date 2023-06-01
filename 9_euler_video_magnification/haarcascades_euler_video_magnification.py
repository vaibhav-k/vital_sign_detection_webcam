import cv2
import numpy as np
from scipy.signal import find_peaks, filtfilt, butter

# Butterworth filter for signal smoothing


def butterworth_filter(signal, cutoff, fs, order=5):
    """
    Apply a Butterworth filter to smooth the signal.

    Args:
        signal (array-like): Input signal to be filtered.
        cutoff (float): Cutoff frequency of the filter.
        fs (float): Sampling frequency of the signal.
        order (int, optional): Filter order. Defaults to 5.

    Returns:
        array-like: Filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    smoothed_signal = filtfilt(b, a, signal)
    return smoothed_signal

# Function to estimate heart rate from the amplified signal


def estimate_heart_rate(signal, fps):
    """
    Estimate heart rate from the amplified signal.

    Args:
        signal (array-like): Amplified signal.
        fps (float): Frames per second of the signal.

    Returns:
        float: Estimated heart rate in beats per minute.
    """
    # Convert signal to 1-D array
    signal = np.asarray(signal).ravel()

    # Perform peak detection on the signal (you can use your preferred method)
    # Here, we use the find_peaks function from scipy.signal
    peaks, _ = find_peaks(signal, height=0)

    # Calculate the time difference between consecutive peaks
    time_diff = np.diff(peaks) / fps

    # Calculate heart rate as beats per minute based on the mean time difference
    heart_rate = 60 / np.mean(time_diff)
    return heart_rate


# Webcam video capture settings
video_width = 640
video_height = 480
fps = 30

# Signal amplification parameters
alpha = 50  # Adjust the amplification factor - does not matter much
low_cutoff = 1.92  # Adjust the low-frequency cutoff - makes a HUGE difference

# Initialize the video capture from webcam
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, video_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height)
video_capture.set(cv2.CAP_PROP_FPS, fps)

# Initialize variables for signal processing
previous_frame = None
signal_buffer = []
heart_rate_buffer = []
heart_rate_window = 10  # Time window for heart rate calculation (in seconds)

# Initialize face detection cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous_frame is not None:
        # Calculate the temporal difference between frames
        diff = gray.astype(np.float32) - previous_frame.astype(np.float32)

        # Amplify the temporal difference signal
        amplified_signal = alpha * diff

        # Apply Butterworth filter for signal smoothing
        filtered_signal = butterworth_filter(amplified_signal, low_cutoff, fps)

        # Add the filtered signal to the buffer
        signal_buffer.extend(filtered_signal)

        # Calculate heart rate based on the signal buffer
        if len(signal_buffer) >= heart_rate_window * fps:
            # Detect faces in the current frame
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                # Assume the first face detected as the region of interest (ROI)
                x, y, w, h = faces[0]

                # Extract the ROI (forehead) from the grayscale frame
                forehead_roi = gray[y:y+h//3, x:x+w]

                # Calculate heart rate based on the ROI signal buffer
                if len(signal_buffer) >= heart_rate_window * fps:
                    roi_heart_rate = estimate_heart_rate(
                        np.array(signal_buffer[-heart_rate_window * fps:]), fps
                    )
                    heart_rate_buffer.append(roi_heart_rate)
                    signal_buffer = signal_buffer[-heart_rate_window * fps:]

                    # Print the current heart rate
                    print("Forehead Heart Rate:", roi_heart_rate)

                    # Draw a rectangle around the detected face and ROI (forehead)
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h//3), (0, 255, 0), 2
                    )

            else:
                # No face detected, clear the signal buffer
                signal_buffer = []

    # Store the current frame for the next iteration
    previous_frame = gray

    # Display the resulting frame
    cv2.imshow("Heart Rate Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
