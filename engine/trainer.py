import os
from engine.loss import compute_loss
from tqdm import tqdm
import torch


def train_one_epoch(model, dataloader, optimizer, device):
    running_loss = 0.0
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets, device)
        print(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # print(running_loss)
    print(f"Epoch loss: {running_loss / len(dataloader)}")

def train(num_epochs, model, data_loader, optimizer, device):
    checkpoint_path = 'saved_model.pth'

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint) # ['model_state_dict']
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # start_epoch = checkpoint['epoch']
    # print(f"Resuming training from epoch {start_epoch}")
    for epoch in range(num_epochs):
        print(f"✨epoch {epoch}")
        train_one_epoch(model, data_loader, optimizer, device)
        torch.save(model.state_dict(), "saved_model.pth")
        print(f"epoch {epoch} ✅ Model saved to 'saved_model.pth'")
