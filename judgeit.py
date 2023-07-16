import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

def get_label_from_class(class_index):
    out = chr(ord("A")+class_index)
    return out

def get_hand_bounding_box(hand_landmarks, image_width, image_height):
    landmark_list = []
    for landmark in hand_landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append((x, y))

    x_min = min(landmark_list, key=lambda x: x[0])[0]
    y_min = min(landmark_list, key=lambda x: x[1])[1]
    x_max = max(landmark_list, key=lambda x: x[0])[0]
    y_max = max(landmark_list, key=lambda x: x[1])[1]

    return x_min, y_min, x_max, y_max


# Initialize hand detection module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the CNN model for American Sign Language
model = tf.keras.models.load_model('americanSignLanguage.h5')

# Function to preprocess the cropped hand image for the model
def preprocess_image(image):
    # Convert image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to match the model's input shape
    image_resized = cv2.resize(image_gray, (28, 28))
    # Reshape the image to have a single channel
    image_reshaped = image_resized.reshape((28, 28, 1))
    # Normalize the pixel values to the range [0, 1]
    #image_normalized = image_reshaped / 255.0
    # Expand the dimensions to match the model's input shape
    image_expanded = np.expand_dims(image_reshaped, axis=0)
    return image_expanded

cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB format for processing by MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Extract the hand landmarks from the detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Calculate the bounding box coordinates for the hand
        x_min, y_min, x_max, y_max = get_hand_bounding_box(hand_landmarks, image.shape[1], image.shape[0])

        # Crop the hand region from the image
        hand_image = image[y_min:y_max, x_min:x_max]

        # Preprocess the hand image for the model
        preprocessed_image = preprocess_image(hand_image)

        # Perform prediction using the CNN model
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)
        predicted_label = get_label_from_class(predicted_class)  # Function to map class index to label

        # Display the predicted label
        cv2.putText(image, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw hand landmarks on the original image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the original image with hand landmarks and predicted label
    cv2.imshow("Image", image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
