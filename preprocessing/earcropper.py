import cv2
import numpy as np
import os


class Detector:

    cascade_left = cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')
    cascade_right = cv2.CascadeClassifier('haarcascade_mcs_rightear.xml')

    def detect(self, img_):
        det_list_left = self.cascade_left.detectMultiScale(img_, 1.05, 1)
        det_list_right = self.cascade_right.detectMultiScale(img_, 1.05, 1)

        ear_list = []
        for x, y, w, h in det_list_left:
            ear_list.append((x, y, w, h))
        for x, y, w, h in det_list_right:
            ear_list.append((x, y, w, h))

        return ear_list


def crop_ears_folder(folder_path):
    # Load the ear detector
    detector = Detector()

    # Traverse all images in the folder
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.jpg'):  # Skip non-JPEG files
            continue
        input_file_path = os.path.join(folder_path, file_name)

        # Load the input image
        img = cv2.imread(input_file_path)

        # Detect ears
        detected_loc = detector.detect(img)

        # Crop the ears
        for x, y, w, h in detected_loc:
            ear_img = img[y-100:y+h+100, x-100:x+w+100]
            # Save the cropped ear image
            ear_file_path = os.path.splitext(input_file_path)[0] + '_ear.jpg'
            cv2.imwrite(ear_file_path, ear_img)

    return


folder_path = '../Dataset/person1/'
crop_ears_folder(folder_path)
