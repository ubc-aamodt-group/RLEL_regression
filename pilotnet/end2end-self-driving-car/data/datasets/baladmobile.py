import torch
import random

import os
import torch
import csv
import numpy as np
import cv2


class BaladMobileDataset(object):
    def __init__(self, cfg, data_dir, ann_path=None):
        self.cfg = cfg
        self.augment_data = self.cfg.IMAGE.DO_AUGMENTATION
        self.data_dir = data_dir    # ...6-16/car1/
        ann_path = os.path.join(data_dir, './data.csv')
        ann_file = open(ann_path, 'r')
        ann_reader = csv.reader(ann_file, delimiter=',')
        self.annotations = [r for r in ann_reader]
        self.annotations = self.annotations[1:]     # ignore headers
        # ANNOTATION -> [front_id, left_id, right_id, steer_angle]

    def __getitem__(self, idx):
        while True:
            steering_command = float(self.annotations[idx][-1])
            if abs(steering_command) > 100.:
                idx = (idx + 1) % len(self.annotations)
            else:
                break

        camera_id = random.choice([0, 1, 2])
        camera = ['camera_front', 'camera_left', 'camera_right'][camera_id]
        filepath = os.path.join(self.data_dir, camera, '{}.jpg'.format(self.annotations[idx][camera_id]))

        if camera == 'camera_front':
            steering_command = steering_command
        if camera == 'camera_left':
            steering_command = steering_command + self.cfg.IMAGE.AUGMENTATION_DELTA_CORRECTION
        if camera == 'camera_right':
            steering_command = steering_command - self.cfg.IMAGE.AUGMENTATION_DELTA_CORRECTION

        image = self._preprocess_img(cv2.imread(filepath))

        if self.augment_data:
            # mirror images with chance=0.5
            if random.choice([True, False]):
                image = image[:, ::-1, :]
                steering_command *= -1.

            # perturb slightly steering direction
            steering_command += np.random.normal(loc=0, scale=self.cfg.STEER.AUGMENTATION_SIGMA)

            # if color images, randomly change brightness
            if self.cfg.MODEL.CNN.INPUT_CHANNELS == 3:
                image = cv2.cvtColor(image, code=cv2.COLOR_BGR2HSV)
                image[:, :, 2] *= random.uniform(self.cfg.IMAGE.AUGMENTATION_BRIGHTNESS_MIN,
                                                 self.cfg.IMAGE.AUGMENTATION_BRIGHTNESS_MAX)
                image[:, :, 2] = np.clip(image[:, :, 2], a_min=0, a_max=255)
                image = cv2.cvtColor(image, code=cv2.COLOR_HSV2BGR)

        image = torch.from_numpy(image)
        steering_command = torch.tensor([steering_command])

        path_to_image = self.id_to_filename[idx].split(".")[0]

        return image, steering_command, path_to_image

    def __len__(self):
        return len(self.annotations)

    def _preprocess_img(self, img):
        # set training images resized shape
        h, w = self.cfg.IMAGE.TARGET_HEIGHT, self.cfg.IMAGE.TARGET_WIDTH

        # crop image (remove useless information)
        img_cropped = img[range(*self.cfg.IMAGE.CROP_HEIGHT), :, :]

        # resize image
        img_resized = cv2.resize(img_cropped, dsize=(w, h))

        # eventually change color space
        if self.cfg.MODEL.CNN.INPUT_CHANNELS == 1:
            img_resized = np.expand_dims(cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)[:, :, 0], 2)

        return img_resized.astype('float32')
