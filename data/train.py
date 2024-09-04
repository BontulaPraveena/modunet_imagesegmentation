import torch
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    binary_cross_entropy_dice_loss,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
# PATIENCE = 10  # Patience for early stopping
# EARLY_STOPPING_DELTA = 0.001  # Minimum change in validation loss to consider as improvement

def train_fn(loader, model, optimizer, loss_fn, scaler):
    Loop = tqdm(loader)

    # for batch_idx, (data, targets) in enumerate(Loop):
    #     data = data.to(device=DEVICE)
    #     targets = targets.float().unsqueeze(1).to(device=DEVICE)

    for batch_idx, (data, targets) in enumerate(Loop):
        if data is None or targets is None:
            continue  # Skip None samples
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)


        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        Loop.set_postfix(loss=loss.item())

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    os.makedirs(folder, exist_ok=True)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")  # Save predicted masks
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")  # Save ground truth masks

    model.train()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = binary_cross_entropy_dice_loss  # Using the custom loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4, betas=(0.9, 0.999))

    train_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        try:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
            print("Checkpoint loaded successfully!")
        except FileNotFoundError:
            print("Checkpoint file not found. Training from scratch.")

    best_val_loss = float('inf')
    patience_counter = 0 

    # check_accuracy(train_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # val_loss = evaluate(model, val_loader, loss_fn, DEVICE)

        # if val_loss < best_val_loss - EARLY_STOPPING_DELTA:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     checkpoint = {
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     }
        #     save_checkpoint(checkpoint)
        # else:
        #     patience_counter += 1
        #     if patience_counter >= PATIENCE:
        #         print("Early stopping triggered!")
        #         break

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        # check_accuracy(train_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     train_loader, model, folder="saved_images/training", device=DEVICE,
        # )

if __name__ == "__main__":
    main()