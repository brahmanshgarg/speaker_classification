from pathlib import Path

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def split_data(data_dir, train_dir, test_dir, test_size):
    import os
    import shutil
    from sklearn.model_selection import train_test_split

    # Start fresh each run.
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Read from raw class folders and copy files into train/test.
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        if not images:
            continue
        # print(images)
        
        # train_images, test_images = train_test_split(
        #     images, test_size=test_size, random_state=42
        # )
        # first 80% of the images go to train, last 20% go to test, alphabetically sorted
        images.sort()
        split_index = int(len(images) * (1 - test_size))
        train_images = images[:split_index]
        test_images = images[split_index:]

        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(train_class_dir, img))
        for img in test_images:
            shutil.copy2(os.path.join(class_path, img), os.path.join(test_class_dir, img))




def create_datasets(data_dir):

    data_dir = Path(data_dir)
    train_dataset = ImageFolder(root=data_dir / "train", transform=transform)
    test_dataset = ImageFolder(root=data_dir / "test", transform=transform)
    return train_dataset, test_dataset


def create_dataloaders(data_dir, batch_size=128):
    train_dataset, test_dataset = create_datasets(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader, train_dataset, test_dataset