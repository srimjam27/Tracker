import cv2
import mediapipe as mp

# Initialize MediaPipe Hands, Face Detection, and Face Mesh
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands()
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize VideoCapture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Detection
    face_results = face_detection.process(frame_rgb)

    # Process the image with MediaPipe Face Mesh
    face_mesh_results = face_mesh.process(frame_rgb)

    # Process the image with MediaPipe Hands
    hands_results = hands.process(frame_rgb)

    # Check if face is detected
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bbox, (128, 0, 128), 2)  # Purple color for face detection

    # Check if face mesh landmarks are detected
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                # Extract landmark positions
                height, width, _ = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

    # Check if hands are detected
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            for lm in hand_landmarks.landmark:
                # Extract landmark positions
                height, width, _ = frame.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # Draw connection lines between the landmarks
            connections = mp_hands.HAND_CONNECTIONS
            for connection in connections:
                # Extract the indices of the connected landmarks
                idx1, idx2 = connection
                # Extract the positions of the connected landmarks
                lm1 = hand_landmarks.landmark[idx1]
                lm2 = hand_landmarks.landmark[idx2]
                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(lm1.x * width), int(lm1.y * height)
                x2, y2 = int(lm2.x * width), int(lm2.y * height)
                # Draw the connection line
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Purple color for hand connections

    # Display the frame
    cv2.imshow('Face, Hand and Landmarks Detection', frame)

    # Check for key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
