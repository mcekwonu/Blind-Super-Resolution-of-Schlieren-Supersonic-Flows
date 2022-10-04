import os
import cv2
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def download_images_from_event(input_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    event_acc = event_accumulator.EventAccumulator(input_dir, size_guidance={"image": 0})
    event_acc.Reload()

    for tag in event_acc.Tags()["images"]:
        events = event_acc.Images(tag)
        tag_name = tag.replace("/", "_")
        dirpath = os.path.join(save_dir, tag_name)
        os.makedirs(dirpath, exist_ok=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            img = cv2.imdecode(s, cv2.IMREAD_UNCHANGED)
            output_path = os.path.join(dirpath, f"{index:03d}.png")
            cv2.imwrite(output_path, img)


if __name__ == "__main__":

    for root, subdirs, files in os.walk("../training_logs/images"):
        for subdir in subdirs:
            input_path = os.path.join(root, subdir)
            output_dir = "../training_logs/images"
            download_images_from_event(input_dir=input_path, save_dir=output_dir)
