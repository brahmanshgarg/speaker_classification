from speaker_classifier_modules.model import VoiceClassifier
from speaker_classifier_modules.data import create_dataloaders, split_data
from speaker_classifier_modules.config import DATA_DIR, BEST_MODEL
from speaker_classifier_modules.train import train_model
import torch

print("runs!!!")

def calc_accuracy(dl, model, device):
    correct = 0
    total = 0
    misclassified_images = []
    model = model.to(device)
    with torch.no_grad():
        for images, labels in dl:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # list down the predicted and actual labels for each image in the batch
            for i in range(len(labels)):
                # print(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
                if predicted[i].item() != labels[i].item():
                    #print(f"Misclassified image index in batch: {i}")
                    # name of the image in the dataset
                    #print(f"Image path: {test_dataset.samples[i][0]}")
                    # raw logits of misclassified image
                    #print(f"Raw logits: {outputs[i].data}")
                    misclassified_images.append((images[i], predicted[i].item(), labels[i].item(), outputs[i].data))

    accuracy = correct / total
    # print(f"Accuracy: {accuracy:.4f}")
    return accuracy, misclassified_images

if __name__ == "__main__":
    split_data = False
    if split_data:
        split_data(DATA_DIR / "raw", DATA_DIR / "train", DATA_DIR / "test", test_size=0.2)

    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(DATA_DIR, batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_losses, test_losses = train_model(train_loader, test_loader, device, 
                                                   epochs=20, learning_rate=0.003, 
                                                   weight_decay=3e-3, patience=5, 
                                                   warmup=8, save_path=BEST_MODEL)

    model.load_state_dict(torch.load(BEST_MODEL))
    # model.load_state_dict(torch.load("models\\best_model_20260812_192422.pth"))

    model.eval()
 
    # Accuracy calculation on the test set
    accuracy, misclassified_images = calc_accuracy(test_loader, model, device)
    accuracy_train, misclassified_images_train = calc_accuracy(train_loader, model, device)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Train Accuracy: {accuracy_train:.4f}")