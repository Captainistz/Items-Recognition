import torch
import cv2

from pathlib import Path
from ultralytics import YOLO


model = YOLO('./runs/detect/lost_objects_25epochs_nofreeze/weights/best.pt')

dataset_path = Path("./data/yolo_dataset")
data_to_predict_path = dataset_path / "images" / "val"
val_image_list = list(data_to_predict_path.glob("*.png"))


# for i in range(1, 5):
img = cv2.imread(f'../python/img/detected/first-plane_0_1558.png')
# img = img[:, :350]
# img = cv2.resize(img, (400, 300))
res = model.predict(img)
for result in res:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    # result.save(filename=f"result{i}.jpg")