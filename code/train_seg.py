import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import u_net
import utils

# to ensure reproducible training/validation split
random.seed(42)

# directorys with data and to store training checkpoints and logs
DATA_DIR = Path.cwd() / "TrainingData"
CHECKPOINTS_DIR = Path.cwd() / "segmentation_model_weights"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_LOGDIR = "segmentation_runs"

# training settings and hyperparameters
NO_VALIDATION_PATIENTS = 2
IMAGE_SIZE = [64, 64]
BATCH_SIZE = 32
N_EPOCHS = 100
LEARNING_RATE = 1e-4
TOLERANCE = 0.01  # for early stopping

# find patient folders in training directory
# excluding hidden folders (start with .)
patients = [
    path
    for path in DATA_DIR.glob("*")
    if not any(part.startswith(".") for part in path.parts)
]
random.shuffle(patients)

# split in training/validation after shuffling
partition = {
    "train": patients[:-NO_VALIDATION_PATIENTS],
    "validation": patients[-NO_VALIDATION_PATIENTS:],
}

# load training data and create DataLoader with batching and shuffling
dataset = utils.ProstateMRDataset(partition["train"], IMAGE_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# load validation data
valid_dataset = utils.ProstateMRDataset(partition["validation"], IMAGE_SIZE)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
)

# initialise model, optimiser, and loss function
loss_function = torch.nn.MSELoss
unet_model = u_net.UNet
learning_rate = 0.01
optimizer = torch.optim.Adam(unet_model.parameters(), lr=learning_rate)

minimum_valid_loss = 10  # initial validation loss
writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR)  # tensorboard summary

# training loop
for epoch in range(N_EPOCHS):
    current_train_loss = 0.0
    current_valid_loss = 0.0
    
    # TODO 
    # training iterations
    for inputs, labels in tqdm(dataloader,position=0):
        
        #zero gradients in each iterations
        optimizer.zero_grad()
        outputs = unet_model(inputs)
        loss = loss_function(outputs, labels.float())
        loss.backward()
        current_train_loss += loss.item()
        optimizer.step()


    # evaluate validation loss
    with torch.no_grad():
        unet_model.eval()
        # TODO 
        for inputs, labels in tqdm(valid_dataloader, position=0):
            outputs = unet_model(inputs)
            loss = loss_function(outputs, labels.float())
            current_valid_loss += loss.item()

        unet_model.train()

    # write to tensorboard log
    writer.add_scalar("Loss/train", current_train_loss / len(dataloader), epoch)
    writer.add_scalar(
        "Loss/validation", current_valid_loss / len(valid_dataloader), epoch
    )

    # if validation loss is improving, save model checkpoint
    # only start saving after 10 epochs
    if (current_valid_loss / len(valid_dataloader)) < minimum_valid_loss + TOLERANCE:
        minimum_valid_loss = current_valid_loss / len(valid_dataloader)
        if epoch > 9:
            torch.save(
                unet_model.cpu().state_dict(),
                CHECKPOINTS_DIR / f"u_net_{epoch}.pth",
            )
