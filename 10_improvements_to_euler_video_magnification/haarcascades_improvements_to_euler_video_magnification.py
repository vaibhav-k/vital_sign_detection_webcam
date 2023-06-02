# import the necessary packages
import cv2
import numpy as np
import time
from scipy.signal import find_peaks, filtfilt, butter


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


def capture_video(video_capture, video_width, video_height, fps):
    """
    Capture video frames from the webcam.

    Args:
        video_capture: Video capture object.
        video_width (int): Width of the video frame.
        video_height (int): Height of the video frame.
        fps (float): Frames per second of the video.

    Yields:
        frame: Captured video frame.
    """
    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (video_width, video_height))

        yield frame


def process_frame(
    frame,
    previous_frame,
    signal_buffer,
    heart_rate_buffer,
    face_cascade,
    fps,
    alpha,
    low_cutoff,
    heart_rate_window
):
    """
    Process a single frame to detect faces, calculate heart rate, and annotate the frame.

    Args:
        frame: Input frame to process.
        previous_frame: Previous frame for temporal difference calculation.
        signal_buffer: Buffer to store the filtered signal.
        heart_rate_buffer: Buffer to store heart rate values.
        face_cascade: Cascade classifier for face detection.
        fps: Frames per second of the video.
        alpha: Amplification factor for the temporal difference signal.
        low_cutoff: Low-frequency cutoff for the Butterworth filter.
        heart_rate_window: Time window for heart rate calculation (in seconds).

    Returns:
        tuple: Processed frame, estimated heart rate, and face detection status.
    """
    face_detected = False
    roi_heart_rate = None

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
                forehead_roi = gray[y:y + h // 3, x:x + w]

                # Calculate heart rate based on the ROI signal buffer
                roi_heart_rate = estimate_heart_rate(
                    np.array(signal_buffer), fps
                )
                heart_rate_buffer.append(roi_heart_rate)

                # Set face detected flag to True
                face_detected = True

                # Draw a rectangle around the detected face and ROI (forehead)
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h // 3), (0, 255, 0), 2
                )

            # Clear the signal buffer
            signal_buffer.clear()

    return frame, roi_heart_rate, face_detected


def display_frame(frame, face_detected, roi_heart_rate):
    """
    Display the processed frame with heart rate information.

    Args:
        frame: Processed frame.
        face_detected: Boolean flag indicating if a face was detected.
        roi_heart_rate: Estimated heart rate from the ROI (forehead) signal.
    """
    # Define the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2

    # Calculate the width and height of the text
    text = f"Heart rate = {roi_heart_rate:.2f} BPM" if face_detected else "N/A"
    text_width, text_height = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )[0]
    color = (0, 0, 0) if face_detected else (0, 0, 255)

    # Calculate the position to center align the text
    x = int((frame.shape[1] - text_width) / 2)
    y = int(frame.shape[0] * 0.1)

    # Print the heart rate or no face detected message in the center of the frame
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        color,
        font_thickness
    )

    # Display the resulting frame
    cv2.imshow("Heart Rate Detection", frame)


def main():
    try:
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

        # Check if the camera is opened successfully
        if not video_capture.isOpened():
            raise RuntimeError("Failed to open camera. Exiting...")

        # Initialize variables for signal processing
        previous_frame = None
        signal_buffer = []
        heart_rate_buffer = []

        # Increasing this window can provide a more accurate estimation of the heart rate.
        # Time window for heart rate calculation (in seconds)
        heart_rate_window = 10

        # Initialize variable face detection
        face_detected = False

        # Initialize face detection cascade classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        for frame in capture_video(video_capture, video_width, video_height, fps):
            processed_frame, roi_heart_rate, face_detected = process_frame(
                frame,
                previous_frame,
                signal_buffer,
                heart_rate_buffer,
                face_cascade,
                fps,
                alpha,
                low_cutoff,
                heart_rate_window
            )

            display_frame(processed_frame, face_detected, roi_heart_rate)

            # Store the current frame for the next iteration
            previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video capture and close all windows
        video_capture.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
