import torch.nn as nn, torch
import torch.nn.functional as F



    
class LeNet5Adapter(nn.Module):
    def __init__(self, num_classes=10, rank=6):
        super(LeNet5Adapter, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.lora1 = LoRA2d(self.conv1.in_channels, self.conv1.out_channels, self.conv1.kernel_size,
                            self.conv1.stride, self.conv1.padding, self.conv1.dilation, self.conv1.groups,
                            rank=rank, lora_alpha=1, lora_dropout=0)
        self.bn1 = nn.InstanceNorm2d(6, affine=True)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.lora2 = LoRA2d(self.conv2.in_channels, self.conv2.out_channels, self.conv2.kernel_size,
                            self.conv2.stride, self.conv2.padding, self.conv2.dilation, self.conv2.groups,
                            rank=rank, lora_alpha=1, lora_dropout=0)
        self.bn2 = nn.InstanceNorm2d(16, affine=True)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.lora3 = LoRALinear(self.fc1.in_features, self.fc1.out_features,
                            rank=rank, lora_alpha=1, lora_dropout=0)
        
        self.bn3 = nn.LayerNorm(120, elementwise_affine=True)
        self.fc2 = nn.Linear(120, 84)
        self.lora4 = LoRALinear(self.fc2.in_features, self.fc2.out_features,
                            rank=rank, lora_alpha=1, lora_dropout=0)
        self.bn4 = nn.LayerNorm(84, elementwise_affine=True)
        self.fc3 = nn.Linear(84, num_classes)
        self.lora5 = LoRALinear(self.fc3.in_features, self.fc3.out_features,
                            rank=rank, lora_alpha=1, lora_dropout=0)

    def forward(self, x):
        x = self.conv1(x) + self.lora1(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, 2)
        x = self.conv2(x) + self.lora2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x) + self.lora3(x)
        x = F.relu(self.bn3(x))

        x = self.fc2(x) + self.lora4(x)
        x = F.relu(self.bn4(x))
        x = self.fc3(x) + self.lora5(x)
        return x

import sys
sys.path.append("../../../")
from simulator.algorithms.dnn.torch.convert import synchronize

class Trainer:
    def __init__(self, model, optimizer, num_epochs, scheduler, train_dataset, val_dataset,
                 batch_size=128, device=None, criterion=None, best_model_path='best_model.pt'):
        """
        Args:
            model: the PyTorch model.
            optimizer: optimizer for training.
            num_epochs: total number of epochs for training.
            scheduler: learning rate scheduler.
            train_dataset: training dataset (instance of torch.utils.data.Dataset).
            val_dataset: validation dataset (instance of torch.utils.data.Dataset).
            batch_size: batch size for both train and val.
            device: torch.device; if None it will use CUDA if available.
            criterion: loss function; if None, defaults to torch.nn.CrossEntropyLoss.
            best_model_path: file path to save the best model.
        """
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
        self.best_model_path = best_model_path

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False, pin_memory=True, num_workers=4)

        self.model.to(self.device)

    def validate(self, recalibrate=False):
        """Evaluates the model on the validation dataset.

        Returns:
            avg_loss (float): Average loss over the validation set.
            accuracy (float): Accuracy over the validation set.
        """
        if recalibrate:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                # For classification tasks assuming outputs are logits
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, crossSim = False, adapter=False):
        """Trains the model and evaluates it on the validation set each epoch.
        
        The best performing model (based on validation accuracy) is saved.
        """
        best_acc = 0.0
        for epoch in range(1, self.num_epochs + 1):
            # Training Phase
            if adapter:
                self.model.eval()
            else:
                self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if crossSim:
                    synchronize(self.model)

                running_loss += loss.item() * inputs.size(0)
            
            train_loss = running_loss / len(self.train_loader.dataset)
            
            # Validation Phase (using the validate method)
            val_loss, val_acc = self.validate()

            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

            # Step the scheduler if provided
            if self.scheduler is not None:
                self.scheduler.step()

            # Save the best model based on validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Saving best model with accuracy: {best_acc:.4f}")

        print("Training complete. Best validation accuracy: {:.4f}".format(best_acc))
