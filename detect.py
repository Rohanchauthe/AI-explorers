import cv2
import csv
import mediapipe as mp
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class SignLanguageRecognizer:
    def __init__(self, model_path):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.load_training_data(model_path)
        
    def load_training_data(self, model_path):
        data = np.loadtxt(model_path, delimiter=',', skiprows=1, dtype=str)
        X = data[:, 1:].astype(float)
        y = data[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.knn.fit(X_train, y_train)
        print(f"Model loaded with {len(X)} samples")

    def recognize(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        hand_detected = False
        
        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            prediction = self.knn.predict([landmarks])
            return prediction[0], hand_detected
        
        return None, hand_detected

def main():
    recognizer = SignLanguageRecognizer("sign_language_model.csv")
    cap = cv2.VideoCapture(0)
    
    # Timing controls
    last_letter_time = 0
    last_detection_time = time.time()
    letter_cooldown = 1  # 1 second between letters
    word_cooldown = 5    # 5 seconds for new word
    
    current_word = []
    words_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        prediction, hand_detected = recognizer.recognize(frame)
        
        # Handle letter input with cooldown
        if hand_detected:
            last_detection_time = current_time
            if prediction and (current_time - last_letter_time) >= letter_cooldown:
                current_word.append(prediction)
                last_letter_time = current_time
                print(f"Added letter: {prediction}")
        else:
            # Handle word completion after timeout
            if (current_time - last_detection_time) > word_cooldown:
                if current_word:
                    completed_word = ''.join(current_word)
                    words_list.append(completed_word)
                    print("\n--- New Word Completed ---")
                    print(f"Word: {completed_word}")
                    print("All Words:", ' | '.join(words_list))
                    current_word = []
                last_detection_time = current_time

        # Display current status
        status_text = f"Current Word: {''.join(current_word)}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display timing info
        time_since_last_letter = current_time - last_letter_time
        cooldown_text = f"Next letter in: {max(0, letter_cooldown - time_since_last_letter):.1f}s"
        cv2.putText(frame, cooldown_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Sign Language Detection', frame)

        key = cv2.waitKey(10)
        if key == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()