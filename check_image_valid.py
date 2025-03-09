import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def test_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Invalid image path")
        return
    
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        print("Hand detected!")
        # Draw landmarks
        mp_drawing = mp.solutions.drawing_utils
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Detection', annotated_image)
        cv2.waitKey(0)
    else:
        print("No hand detected")
    
if __name__ == "__main__":
    test_image("dataset/A/0.png")  # Replace with your image
    hands.close()