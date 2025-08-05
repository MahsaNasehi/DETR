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
    for epoch in range(num_epochs):
        train_one_epoch(model, data_loader, optimizer, device)
        torch.save(model.state_dict(), "saved_model.pth")
        print("✅ Model saved to 'saved_model.pth'")  