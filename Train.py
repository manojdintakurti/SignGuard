import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator

# Set parameters
path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
steps_per_epoch_val = 2000
epochs_val = 10
image_dimensions = (32, 32, 3)
test_ratio = 0.2
validation_ratio = 0.2

# Import images
count = 0
images = []
class_no = []
class_folders = os.listdir(path)
print("Total Classes Detected:", len(class_folders))
no_of_classes = len(class_folders)
print("Importing Classes.....")
for class_idx in range(0, len(class_folders)):
    class_images = os.listdir(os.path.join(path, str(count)))
    for img_name in class_images:
        img_path = os.path.join(path, str(count), img_name)
        cur_img = cv2.imread(img_path)
        images.append(cur_img)
        class_no.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
class_no = np.array(class_no)

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, class_no, test_size=test_ratio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_ratio)

# Check data shapes
print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

# Check image dimensions
assert X_train.shape[1:] == image_dimensions, "The dimensions of the training images are incorrect"
assert X_validation.shape[1:] == image_dimensions, "The dimensions of the validation images are incorrect"
assert X_test.shape[1:] == image_dimensions, "The dimensions of the test images are incorrect"

# Read CSV file
data = pd.read_csv(labelFile)
print("Data shape", data.shape, type(data))

# Display some sample images of all the classes
num_of_samples = []
cols = 5
fig, axs = plt.subplots(nrows=no_of_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :])
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(f"{j}-{row['Name']}")
            num_of_samples.append(len(x_selected))

# Display a bar chart showing the number of samples for each category
print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, no_of_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

# Preprocess the images
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

X_train = np.array([preprocess(img) for img in X_train])
X_validation = np.array([preprocess(img) for img in X_validation])
X_test = np.array([preprocess(img) for img in X_test])

# Add a depth of 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Data augmentation
data_gen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
data_gen.fit(X_train)
batches = data_gen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

# Show augmented image samples
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(image_dimensions[0], image_dimensions[1]), cmap=plt.get_cmap("gray"))
    axs[i].axis('off')
plt.show()

# Convert labels to categorical
y_train = to_categorical(y_train, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)

# Convolutional Neural Network model
def create_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(image_dimensions[0], image_dimensions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
cnn_model = create_model()
print(cnn_model.summary())
history = cnn_model.fit_generator(
    data_gen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=steps_per_epoch_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation),
    shuffle=True
)

# Plot training history
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model on the test set
test_score = cnn_model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', test_score)