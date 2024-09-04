import torch
import torchvision
import torch.nn.functional as F
from dataset import CustomDataset  # Assuming you have Covid19CTDataset defined
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def dice_coefficient(y_pred, y_true, epsilon=1e-6):
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice_score = (2. * intersection + epsilon) / (union + epsilon)
    return dice_score

def binary_cross_entropy_dice_loss(y_pred, y_true, alpha=0.5):
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
    dice = dice_coefficient(torch.sigmoid(y_pred), y_true)
    combined_loss = alpha * bce_loss + (1 - alpha) * (1 - dice)
    return combined_loss

def get_loaders(
    train_dir,
    train_maskdir,
    batch_size,
    train_transform,
    num_workers=4,
    pin_memory=True,
):
    
    train_ds = CustomDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/mask_{idx}.png")

    model.train()