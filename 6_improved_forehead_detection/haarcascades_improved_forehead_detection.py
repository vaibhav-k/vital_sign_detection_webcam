import cv2
import numpy as np
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks


def smooth_signal(signal_buffer, padlen, cutoff_frequency):
    signal = np.array(signal_buffer[-padlen:])
    frequencies = fftfreq(padlen)
    fft_values = fft(signal)

    b, a = butter(4, cutoff_frequency, "low")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def calculate_heart_rate(signal_buffer, video_fps, heart_rate_buffer, heart_rate_window):
    peaks, _ = find_peaks(signal_buffer, height=0)

    if len(peaks) >= 2:
        peak_times = np.array(peaks) / video_fps
        time_diff = np.diff(peak_times)
        heart_rate = 60 / np.mean(time_diff)
        heart_rate_buffer.append(heart_rate)

        if len(heart_rate_buffer) > heart_rate_window:
            heart_rate_buffer = heart_rate_buffer[-heart_rate_window:]
            average_heart_rate = np.mean(heart_rate_buffer)
            print(f"Heart Rate: {average_heart_rate:.2f} beats per minute")
    return heart_rate_buffer


# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize the video_capture
video_capture = None
if len(sys.argv) == 2:
    video_capture = cv2.VideoCapture(sys.argv[1])
else:
    video_capture = cv2.VideoCapture(0)

# Define variables for signal processing
previous_average_color = None
signal_buffer = []
padlen = 100  # Length of the signal buffer

# Variables for heart rate estimation
heart_rate_buffer = []
heart_rate_frequency = None
heart_rate_window = 5  # Time window for heart rate calculation (in seconds)

# Variables for respiration rate estimation
respiration_rate_buffer = []
respiration_rate_frequency = None
# Time window for respiration rate calculation (in seconds)
respiration_rate_window = 10

frame_counter = 0
face_detection_interval = 5  # Face detection interval in frames

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    # Increment frame counter
    frame_counter += 1

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection at a specified interval
    if frame_counter % face_detection_interval == 0:
        # Reset frame counter for next face detection
        frame_counter = 0

        # Perform face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw rectangles around the detected faces and extract ROIs
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x+w, y+h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            face_roi = frame[y:y+h, x:x+w]

            # Extract the region of interest from the face ROI
            # Adjust the region based on the specific location for heart rate and respiration rate estimation
            forehead = frame[y:y+int(h/4), x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+int(h/4)), (0, 255, 0), 2)
            cv2.putText(frame, "Forehead", (x+w, y+int(h/4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Calculate the average color in the vital ROI
            average_color = np.mean(forehead)

            # Signal processing
            if previous_average_color is None:
                previous_average_color = average_color
            else:
                # Calculate the color variation (signal)
                color_variation = average_color - previous_average_color

                # Store the color variation in the signal buffer
                signal_buffer.append(color_variation)

                # Apply signal processing techniques to the signal buffer
                if len(signal_buffer) >= padlen:
                    # Smooth the signal using a Butterworth filter
                    cutoff_frequency = 0.1
                    filtered_signal = smooth_signal(
                        signal_buffer, padlen, cutoff_frequency)

                    # Estimate heart rate based on detected peaks
                    heart_rate_buffer = calculate_heart_rate(filtered_signal, video_capture.get(
                        cv2.CAP_PROP_FPS), heart_rate_buffer, heart_rate_window)

                # Update the previous average color with the current average color
                previous_average_color = average_color

    # Display the resulting frame
    cv2.imshow("Face Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
