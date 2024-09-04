import os
import torch
import torchvision  
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import UNET
from utils import load_checkpoint, get_loaders, save_predictions_as_imgs

# Define the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(preds, masks):
    intersection = torch.logical_and(masks, preds).sum()
    union = torch.logical_or(masks, preds).sum()
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou

def calculate_mae(preds, masks):
    mae = torch.abs(preds - masks).mean()
    return mae

def calculate_dice(preds, masks):
    intersection = torch.logical_and(masks, preds).sum()
    union = torch.logical_or(masks, preds).sum()
    dice = (2 * intersection) / (union + 1e-8)
    return dice

def test(model, test_loader, device):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables for accuracy, IoU, MAE, and Dice score calculation
    num_correct = 0
    num_pixels = 0
    iou_total = 0
    mae_total = 0
    dice_score = 0
    
    # Check if the directory exists, if not, create it
    output_folder = "saved_images/testing"
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over the test dataset and perform inference
    for idx, (images, masks) in enumerate(tqdm(test_loader)):
        # Move images and masks to the device
        images = images.to(device)
        masks = masks.to(device)
        
        # Perform inference
        with torch.no_grad():
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()

        # Calculate accuracy
        num_correct += (preds == masks).sum()
        num_pixels += torch.numel(preds)

        # Calculate IoU
        iou_total += calculate_iou(preds, masks)

        # Calculate MAE
        mae_total += calculate_mae(preds, masks)
        
        # Calculate Dice score
        dice_score += calculate_dice(preds, masks)
    
    # Calculate accuracy, IoU, MAE, and Dice score
    accuracy = num_correct.item() / num_pixels * 100
    iou_avg = iou_total / len(test_loader)  # Use test_loader instead of loader
    mae_avg = mae_total / len(test_loader)  # Use test_loader instead of loader
    dice_avg = dice_score / len(test_loader)  # Use test_loader instead of loader
    
    # Print accuracy, IoU, MAE, and Dice score
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Average IoU: {iou_avg:.4f}")
    print(f"Average MAE: {mae_avg:.4f}")
    print(f"Average Dice Score: {dice_avg:.4f}")

    # print some examples to a folder
    save_predictions_as_imgs(
        test_loader, model, folder=output_folder, device=device,
    )
    
    # Set the model back to training mode
    model.train()

if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = UNET(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load("my_checkpoint.pth.tar")
    load_checkpoint(checkpoint, model)

    # Define transformations for test dataset
    test_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0], std=[1.0]),
        ToTensorV2()
    ])

    # Create the test dataset and data loader
    test_dataset = CustomDataset(image_dir="data/test_images/", mask_dir="data/test_masks/", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Perform testing and calculate IoU and MAE
    test(model, test_loader, device)