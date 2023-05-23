import cv2

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

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
        # You can perform further processing on the extracted face_roi
        # Define the region of interest (ROI) within the face ROI
        # You can adjust the coordinates and size based on your specific requirements
        roi_x = int(face_roi.shape[1] * 0.2)  # X-coordinate of the top-left corner
        roi_y = int(face_roi.shape[0] * 0.1)  # Y-coordinate of the top-left corner
        roi_width = int(face_roi.shape[1] * 0.6)  # Width of the ROI
        roi_height = int(face_roi.shape[0] * 0.3)  # Height of the ROI

        # Extract the smaller region of interest from the face ROI
        vital_roi = face_roi[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Display the extracted region of interest
        cv2.imshow("Vital ROI", vital_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
