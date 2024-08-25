import cv2
import math
import torch
import string
import cvzone
import random
import imutils
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from cv2.typing import MatLike
from shutil import copy, rmtree
from sklearn.model_selection import train_test_split


def transparents(image_bgr: MatLike) -> MatLike:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    _, alpha = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    b, g, r = cv2.split(image_bgr)

    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    dst = cv2.bitwise_not(dst)
    return dst


class YoloWrapper:
    def __init__(self, model_weights: str) -> None:
        if model_weights == "nano":
            model_weights = "yolov8n.pt"
        elif model_weights == "small":
            model_weights = "yolov8s.pt"
        elif model_weights == "medium":
            model_weights = "yolov8m.pt"
        elif (not Path(model_weights).exists()) or (
            Path(model_weights).suffix != ".pt"
        ):
            raise ValueError(
                'The parameter model_weight should be "nano", "small" or a'
                "path to a .pt file with saved weights"
            )
        random.seed(datetime.now().timestamp())

        self.model = YOLO(model_weights).to("cuda")

    def train(self, config: str, epochs: int = 100, name: str = None) -> None:
        if Path(config).suffix != ".yaml":
            raise ValueError("Config file should be a yaml file")
        self.model.train(
            data=config, epochs=epochs, name=name, batch=-1, mosaic=0, close_mosaic=10,
            perspective=0.0001, scale=0.3, shear=10, bgr=1, erasing=0.5
        )

    def predict(
        self,
        image: str | Path | np.ndarray | list[str] | list[Path] | list[np.ndarray],
        threshold: float = 0.25,
    ) -> list[np.ndarray]:

        yolo_results = self.model(image, conf=threshold)
        bounding_boxes = [
            torch.concatenate(
                [x.boxes.xyxy[:, :2], x.boxes.xyxy[:, 2:] - x.boxes.xyxy[:, :2]], dim=1
            )
            .cpu()
            .numpy()
            for x in yolo_results
        ]
        return bounding_boxes

    def save(self) -> None:
        self.model.export(format="onnx")

    @staticmethod
    def create_config_file(
        parent_data_path: str | Path, class_names: list[str], path_to_save: str = None
    ) -> None:
        parent_data_path = Path(parent_data_path)
        if not parent_data_path.exists():
            raise FileNotFoundError(f"Folder {parent_data_path} is not found")
        if not (parent_data_path / "images" / "train").exists():
            raise FileNotFoundError(
                f'There is not folder {parent_data_path / "images" / "train"}'
            )
        if not (parent_data_path / "labels" / "train").exists():
            raise FileNotFoundError(
                f'There is not folder {parent_data_path / "labels" / "train"}'
            )

        config = {
            "path": str(parent_data_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(class_names),
            "names": class_names,
        }

        if not (parent_data_path / "images" / "val").exists():
            config.pop("val")

        if path_to_save is None:
            path_to_save = "config.yaml"
        path_to_save = Path(path_to_save)

        if not path_to_save.suffix:  # is a folder
            path_to_save.mkdir(parents=True, exist_ok=True)
            path_to_save = path_to_save / "config.yaml"

        if path_to_save.suffix != ".yaml":
            raise ValueError(
                f"The path to save the configuration file should be a folder, a yaml file or None."
                f"Got a {path_to_save.suffix} file instead"
            )

        with open(path_to_save, "w") as file:
            for key, value in config.items():
                file.write(f"{key}: {value}\n")

    @staticmethod
    def generate_data(n_images) -> None:
        images_path = Path("./data/raw_data")

        result_path = Path("./data/yolo_dataset")

        if result_path is None:
            parent_dir = images_path.parent
            result_path = parent_dir / "data"
        else:
            result_path = Path(result_path)

        if result_path.exists():
            rmtree(result_path)
            
        result_path_image_training = result_path / "images" / "train"
        result_path_image_training.mkdir(parents=True, exist_ok=False)
        result_path_label_training = result_path / "labels" / "train"
        result_path_label_training.mkdir(parents=True, exist_ok=False)
        
        result_path_image_validation = result_path / "images" / "val"
        result_path_image_validation.mkdir(parents=True, exist_ok=False)
        result_path_label_validation = result_path / "labels" / "val"
        result_path_label_validation.mkdir(parents=True, exist_ok=False)

        images_list = sorted(list(images_path.glob("*")))

        c = 0
        n_images = 1000

        val_select = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

        for _ in range(min(n_images // 2, 100)):
            white_canvas = np.zeros([900, 1200, 4], dtype=np.uint8)
            white_canvas[:] = 255
            name = "".join(
                random.sample((string.ascii_uppercase + string.digits) * 6, 6)
            )
            folder = "val" if random.choice(val_select) else "train"
            cv2.imwrite(f"./data/yolo_dataset/images/{folder}/{name}.png", white_canvas)
            f = open(f"./data/yolo_dataset/labels/{folder}/{name}.txt", "w")
            f.close()


        for image_path in images_list:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            for _ in range(n_images):
                white_canvas = np.zeros([900, 1200, 4], dtype=np.uint8)
                white_canvas[:] = 255
                res = white_canvas.copy()
                count = random.randint(1, 4)  # debug
                name = "".join(
                    random.sample((string.ascii_uppercase + string.digits) * 6, 6)
                )
                folder = "val" if random.choice(val_select) else "train"
                f = open(f"./data/yolo_dataset/labels/{folder}/{name}.txt", "w")
                for _ in range(count):
                    deg = random.randint(0, 3600) / 10.0
                    scale = random.randint(750, 1300) / 1000.0
                    resized = cv2.resize(
                        image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                    rotated = imutils.rotate_bound(resized, deg)

                    bg = np.array([255, 255, 255])
                    alpha = (rotated[:, :, 3] / 255).reshape(rotated.shape[:2] + (1,))
                    rotated_8U = (
                        (bg * (1 - alpha)) + (rotated[:, :, :3] * alpha)
                    ).astype(np.uint8)
                    rotated_8U = cv2.bitwise_not(rotated)
                    rotated_8U = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

                    _, thresh = cv2.threshold(rotated_8U, 0, 127, cv2.THRESH_BINARY)
                    cont, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    x, y, w, h = cv2.boundingRect(cont[0])
                    rotated = rotated[y : y + h, x : x + w]
                    h, w, _ = rotated.shape
                    offset = 20
                    x = random.randint(offset, 1200 - w - offset)
                    y = random.randint(offset, 900 - h - offset)
                    res = cvzone.overlayPNG(white_canvas, rotated, [x, y])
                    text = (
                        f"{c} "
                        f"{((2 * x + w) / 2) / res.shape[1]} "
                        f"{((2 * y + h) / 2) / res.shape[0]} "
                        f"{w / res.shape[1]} "
                        f"{h / res.shape[0]}"
                        f"\n"
                    )
                    f.write(text)
                f.close()
                cv2.imwrite(f"./data/yolo_dataset/images/{folder}/{name}.png", res)
            c += 1

    @staticmethod
    def create_dataset(
        images_path: str | Path,
        labels_path: str | Path = None,
        result_path: str | Path = None,
        train_size: float = 0.9,
    ) -> None:

        if train_size <= 0 or 1 < train_size:
            raise ValueError(
                f"Train size should be between 0 to 1, but got {train_size}"
            )

        images_path = Path(images_path)
        labels_path = Path(labels_path)

        if result_path is None:
            parent_dir = images_path.parent
            result_path = parent_dir / "data"
        else:
            result_path = Path(result_path)

        if result_path.exists():
            rmtree(result_path)

        all_images = sorted(list(images_path.glob("*")))
        all_labels = sorted(list(labels_path.glob("*")))

        training_dataset, val_dataset, train_labels, val_labels = train_test_split(
            all_images, all_labels, train_size=train_size
        )

        result_path_image_training = result_path / "images" / "train"
        result_path_image_training.mkdir(parents=True, exist_ok=False)
        result_path_label_training = result_path / "labels" / "train"
        result_path_label_training.mkdir(parents=True, exist_ok=False)

        for image, label in zip(training_dataset, train_labels):
            copy(image, result_path_image_training / image.name)
            copy(label, result_path_label_training / label.name)

        if val_dataset:
            result_path_image_validation = result_path / "images" / "val"
            result_path_image_validation.mkdir(parents=True, exist_ok=False)
            result_path_label_validation = result_path / "labels" / "val"
            result_path_label_validation.mkdir(parents=True, exist_ok=False)

            for image, label in zip(val_dataset, val_labels):
                copy(image, result_path_image_validation / image.name)
                copy(label, result_path_label_validation / label.name)
