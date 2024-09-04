import cv2
import os
import numpy as np  # Array handling jazz


# Goes through all images of a given person and collects data of their images to give to the face recognizer
def collect_training_data(file_directory):
    dirs = os.listdir(file_directory)

    faces = []  # List of faces
    labels = []  # List of corresponding labels for faces
    map_label = {}  # Dictionary to map label IDs to people's names

    for label_id, dir_name in enumerate(dirs):  # Go through each directory (person)
        if not os.path.isdir(os.path.join(file_directory, dir_name)):
            continue

        map_label[label_id] = dir_name  # Maps label ID to the directory (person)
        subject_dir_path = os.path.join(file_directory, dir_name)  # Construct path to person's directory
        subject_images_names = os.listdir(subject_dir_path)  # All images of that person

        for image_name in subject_images_names:  # Enumerate through files and only read images
            if image_name.startswith("."):
                continue

            image_path = os.path.join(subject_dir_path, image_name)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces.append(gray)
            labels.append(label_id)

    return faces, labels, map_label


# Uses data collection from a person's file to train a model that can try to predict a person's face
def train_face_recognizer(file_directory):
    faces, labels, map_label = collect_training_data(file_directory)

    # Creates an instance of the Local Binary Patterns Histograms (LBPH) face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('trained_model.yml')
    return map_label


if __name__ == "__main__":
    data_folder_path = '../training_data'
    label_map = train_face_recognizer(data_folder_path)
    print("Training complete. Model saved as 'trained_model.yml'.")
    print("Label map:", label_map)
