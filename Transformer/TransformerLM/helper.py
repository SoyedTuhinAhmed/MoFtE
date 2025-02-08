import math
import torch
from torch.utils.data import DataLoader

class LMTrainer:
    def __init__(self, model, optimizer, num_epochs, scheduler, train_dataset, val_dataset,
                 batch_size=128, device=None, criterion=None, best_model_path='best_model.pt'):
        """
        Args:
            model: the PyTorch language model.
            optimizer: optimizer for training.
            num_epochs: total number of epochs for training.
            scheduler: learning rate scheduler.
            train_dataset: training dataset (instance of torch.utils.data.Dataset) that returns (input, target) pairs.
            val_dataset: validation dataset (instance of torch.utils.data.Dataset) that returns (input, target) pairs.
            batch_size: batch size for both train and validation.
            device: torch.device; if None, it will use CUDA if available.
            criterion: loss function; if None, defaults to torch.nn.CrossEntropyLoss().
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

        # Create data loaders.
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       pin_memory=True, num_workers=4)
        # Use a large batch size for validation if possible (to speed up evaluation)
        self.val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False,
                                     pin_memory=True, num_workers=4)

        self.model.to(self.device)

    def validate(self, recalibrate=False):
        """
        Evaluates the model on the validation dataset.
        For LM tasks, returns the average loss and perplexity.

        Args:
            recalibrate (bool): if True, run in train mode (e.g. for recalibration).
        
        Returns:
            avg_loss (float): Average cross-entropy loss over the validation set.
            perplexity (float): Exponential of the average loss.
        """
        if recalibrate:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Forward pass. Depending on your LM model architecture,
                # the outputs might have shape (seq_len, batch_size, vocab_size)
                # or (batch_size, seq_len, vocab_size). Adjust flattening accordingly.
                outputs = self.model(inputs)
                
                # If outputs are 3-dimensional, flatten them:
                if outputs.dim() == 3:
                    # Assume outputs shape: (seq_len, batch_size, vocab_size)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = self.criterion(outputs, targets)
                # Multiply loss by number of tokens (for proper averaging)
                total_loss += loss.item() * targets.size(0)
                total_tokens += targets.size(0)

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def train(self, crossSim=False, adapter=False):
        """
        Trains the model and evaluates it on the validation set each epoch.
        The best performing model (based on validation perplexity) is saved.
        
        Args:
            crossSim (bool): if True, synchronize model across devices (if using distributed training).
            adapter (bool): if True, skip training (e.g. using adapter weights).
        """
        best_perplexity = float('inf')  # Lower perplexity is better.
        for epoch in range(1, self.num_epochs + 1):
            # Set model mode.
            if adapter:
                self.model.eval()
            else:
                self.model.train()

            running_loss = 0.0
            total_tokens_train = 0

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                if outputs.dim() == 3:
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                if crossSim:
                    # If using a distributed framework that requires synchronization.
                    synchronize(self.model)

                running_loss += loss.item() * targets.size(0)
                total_tokens_train += targets.size(0)

            train_loss = running_loss / total_tokens_train

            # Validation Phase
            val_loss, val_perplexity = self.validate()
            print(f"Epoch [{epoch}/{self.num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Perplexity: {val_perplexity:.4f}")

            # Step the scheduler if provided.
            if self.scheduler is not None:
                self.scheduler.step()

            # Save the best model based on validation perplexity.
            if val_perplexity < best_perplexity:
                best_perplexity = val_perplexity
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"Saving best model with perplexity: {best_perplexity:.4f}")

        print("Training complete. Best validation perplexity: {:.4f}".format(best_perplexity))