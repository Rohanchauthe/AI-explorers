import os
import cv2
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_images(dataset_path, output_file):
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,  # Lower confidence threshold
        min_tracking_confidence=0.3
    )
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['label'] + [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

        for label in sorted(os.listdir(dataset_path)):
            label_path = os.path.join(dataset_path, label)
            if not os.path.isdir(label_path):
                continue

            print(f"\nProcessing {label}:")
            valid_samples = 0
            
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  Skipped {image_file} (invalid image)")
                    continue

                # Preprocess image
                image = cv2.resize(image, (640, 480))
                debug_image = image.copy()
                
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    valid_samples += 1
                    landmarks = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    writer.writerow([label.upper()] + landmarks)
                    
                    # Draw landmarks for verification
                    mp_drawing.draw_landmarks(
                        debug_image,
                        results.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS)
                    
                    cv2.imshow('Verification', debug_image)
                    cv2.waitKey(200)  # Briefly show detection
                else:
                    print(f"  No hand in {image_file} - REJECTED")
                    # Show failed image for debugging
                    cv2.putText(image, "REJECTED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Verification', image)
                    cv2.waitKey(500)

            print(f"  Accepted {valid_samples} samples for {label}")
    
    hands.close()
    cv2.destroyAllWindows()
    print("\nTraining data collection complete!")

if __name__ == "__main__":
    dataset_path = "dataset"
    output_file = "sign_language_model.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Create an 'images' directory with A-Z subfolders first!")
    else:
        process_images(dataset_path, output_file)