# Hyperparameters (adjust as needed)
batch_size = 32
num_epochs = 50
learning_rate = 1e-3

# Model hyperparameters -- adjust these based on your dataset/task.
input_dim = 32      # Should match feature dimension from your LRA data.
embed_dim = 64
num_heads = 4
ff_hidden_dim = 128
num_layers = 2
num_classes = 10    # Set to the number of classes in your LRA task.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model.
model = TinyFormer(input_dim, embed_dim, num_heads, ff_hidden_dim, num_layers, num_classes, dropout=0.1)
model.to(device)

# Define loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load datasets.
# You must download and preprocess the LRA dataset; here we assume preprocessed .npz files.
train_dataset = LRADataset("lra_train.npz")
val_dataset = LRADataset("lra_val.npz")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop.
for epoch in range(num_epochs):
  train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
  val_loss, val_acc = evaluate(model, val_loader, criterion, device)
  print(f"Epoch {epoch+1}/{num_epochs}: "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

# Save the model checkpoint.
torch.save(model.state_dict(), "tinyformer_lra.pth")
print("Training complete. Model saved to tinyformer_lra.pth")
