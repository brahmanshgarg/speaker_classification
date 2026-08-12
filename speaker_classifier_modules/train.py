import torch
from torch import nn
from tqdm import tqdm
from speaker_classifier_modules.model import VoiceClassifier

def train_model(train_loader, test_loader, device, epochs = 20, learning_rate = 0.003, weight_decay = 3e-3, patience = 5, warmup = 8, save_path = None):
    model = VoiceClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # L2 regularization
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    patience_counter = 0
    for epoch in tqdm(range(epochs), desc="Training Progress (epochs)"):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # labels = labels.float().unsqueeze(1)  # convert to float and add dimension for BCEWithLogitsLoss
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))
        # get last learning rate from scheduler
        # print(scheduler.get_last_lr())
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # labels = labels.float().unsqueeze(1)  # convert to float and add dimension for BCEWithLogitsLoss
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        test_losses.append(test_loss / len(test_loader))

        # early stopping logic
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0

            if save_path is not None:
                torch.save(model.state_dict(), save_path)

        elif epoch >= warmup:
            patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping at epoch", epoch + 1)
                break
        
        
        print(
            f"Epoch {epoch+1:2d} | "
            # f"LR={scheduler.get_last_lr()[0]:.6f} | "
            f"Train={train_losses[-1]:.4f} | "
            f"Val={test_losses[-1]:.4f}"
        )

    return model, train_losses, test_losses