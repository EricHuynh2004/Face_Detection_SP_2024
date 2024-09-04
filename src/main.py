###
# @author: Eric Huynh
# Summer Project (file re-uploaded)
#
# SYNOPSIS:
# Using my Nexigo camera and an AI trained face detection model
# open my door without physically interacting with anything
###

import camera


def main():
    cam = camera.initialize_camera()
    face_recognizer = camera.load_face_recognizer('trained_model.yml')  # Quick note all photos I used were jpgs

    # Update this label_map according to the output from train_faces.py
    label_map = {0: "person1", 1: "person2"}  # In my case person1 is me and person 2 is empty

    target_label_id = 0
    open_door_threshold = 60  # Adjust this threshold based on your model's performance

    camera.display_video_feed(cam, face_recognizer, label_map, target_label_id, open_door_threshold)


if __name__ == "__main__":
    main()
