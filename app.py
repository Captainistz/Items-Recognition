from pathlib import Path
from wrapper import YoloWrapper

classes_name = [
    "beaker",
    "goggle",
    "hammer",
    "kapton_tape",
    "pipette",
    "screwdriver",
    "thermometer",
    "top",
    "watch",
    "wrench",
]

dataset_path = Path("./data/yolo_dataset")
images_path = Path("./data/raw_data")
labels_path = Path("./data/labels")
generated_path = Path("./data/generated")
config_path = "lost_objects.yaml"


if __name__ == '__main__':
    
    mode = 'nano'
    epochs = 100
    n_images = 1000
    
    print("----- START -----")
    print("> Generating data")
    YoloWrapper.generate_data(n_images)
    print("  ^ (Done)")
    print("> Create config")
    YoloWrapper.create_config_file(dataset_path, classes_name, config_path)
    print("  ^ (Done)")
    model = YoloWrapper(mode)
    model.train(config_path, epochs=epochs, name=f"[{mode}] lost_objects_{epochs}epochs_{n_images}samples")
    model.save()
    print("-----  END  -----")

