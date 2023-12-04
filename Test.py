import numpy as np
import cv2
import pickle

# Set parameters
frame_width = 640
frame_height = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup the video camera
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, brightness)

# Import the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

# Get class name based on class number
def get_class_name(class_no):
    # Map class numbers to corresponding names
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vechiles',
        'Vechiles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vechiles over 3.5 metric tons'
    ]
    return class_names[class_no]

while True:
    # Read image
    success, img_original = cap.read()

    # Process image
    img = np.asarray(img_original)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)

    img = img.reshape(1, 32, 32, 1)
    cv2.putText(img_original, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_original, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # Predict image
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    probability_value = np.max(predictions)

    if probability_value > threshold:
        cv2.putText(img_original, str(class_index) + " " + str(get_class_name(class_index)),
                    (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img_original, str(round(probability_value * 100, 2)) + "%",
                    (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", img_original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
