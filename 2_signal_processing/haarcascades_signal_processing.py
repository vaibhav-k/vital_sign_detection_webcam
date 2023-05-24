import cv2
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt


# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Define variables for signal processing
previous_average_color = None
signal_buffer = []
padlen = 100  # Length of the signal buffer

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        face_roi = frame[y:y+h, x:x+w]

        # Extract the smaller region of interest from the face ROI
        roi_x = int(face_roi.shape[1] * 0.2)
        roi_y = int(face_roi.shape[0] * 0.1)
        roi_width = int(face_roi.shape[1] * 0.6)
        roi_height = int(face_roi.shape[0] * 0.3)
        vital_roi = face_roi[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Calculate the average color in the vital ROI
        average_color = np.mean(vital_roi)

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
                # For example, perform Fast Fourier Transform (FFT)
                signal = np.array(signal_buffer[-padlen:])
                frequencies = fftfreq(padlen)
                fft_values = fft(signal)

                # Filter the signal using a Butterworth filter
                # Modify the filter parameters based on your requirements
                cutoff_frequency = 0.1
                b, a = butter(4, cutoff_frequency, 'low')
                filtered_signal = filtfilt(b, a, signal)
                print(f"filtered_signal = {filtered_signal}")

                # Do further processing or analysis with the filtered signal

            # Update the previous average color with the current average color
            previous_average_color = average_color

        # Display the extracted region of interest
        cv2.imshow("Vital ROI", vital_roi)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
