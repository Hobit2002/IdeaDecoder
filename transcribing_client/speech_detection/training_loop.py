import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from cp_resnet import get_model  # Ensure cp_resnet.py is in the path
from dataset import AudioSegmentDataset, mixstyle  # Replace with actual file name
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
epochs = 10
learning_rate = 1e-3
weight_decay = 0.001
val_split_ratio = 0.1
num_workers=0
best_val_acc = 0.0
mixstyle_p = 0.8
mixstyle_alpha = 0.4 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Full dataset
full_dataset = AudioSegmentDataset('data/speech', 'data/non_speech', augment = False)

# Split into train and val
val_size = int(len(full_dataset) * val_split_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model
model = get_model(n_classes=1, in_channels=1)  # Binary classification
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training Loop
#full_dataset.augment = True
for epoch in range(epochs):
    model.train()
    train_losses = []
    all_preds, all_labels = [], []

    for mel_specs, labels in train_loader:
        mel_specs = mel_specs.to(device)

        if full_dataset.augment:
            mel_specs = mixstyle(mel_specs, mixstyle_p, mixstyle_alpha)

        labels = labels.float().unsqueeze(1).to(device)  # Shape [batch_size, 1]

        if len(all_labels) and not len(all_labels) % 100:
            plt.imsave(f"mels/{len(all_labels)}.png", mel_specs[0,0])

        outputs = model(mel_specs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if len(all_labels) and not len(all_labels) % 10:
            if len(all_labels) and not len(all_labels) % 200: print(torch.sigmoid(outputs))
            print(f"{len(all_labels)//batch_size + 1}/{len(train_loader.dataset)//batch_size + 1} Intermediate accuracy:", accuracy_score(all_labels, all_preds))

    train_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {np.mean(train_losses):.4f} Acc: {train_acc:.4f}")

    # Validation
    model.eval()
    val_losses = []
    val_preds, val_labels = [], []
    #full_dataset.augment = False

    with torch.no_grad():
        for mel_specs, labels in val_loader:
            mel_specs = mel_specs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            if full_dataset.augment:
                mel_specs = mixstyle(mel_specs, mixstyle_p, mixstyle_alpha)

            outputs = model(mel_specs)
            loss = criterion(outputs, labels)

            val_losses.append(loss.item())

            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Loss: {np.mean(val_losses):.4f} Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), f"speech_detection_model_{epoch+1}.pth")
    
    """# Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_speech_detection_model.pth")
        print("Saved best model.")"""

