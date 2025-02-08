from MoFtE.Transformer.TinyFormer.Transformer import *

if __name__ == '__main__':
    # Assume you have already built a vocabulary (vocab) using your data loader.
    # For demonstration, let's use a dummy vocabulary size:
    vocab_size = 33278  # for WikiText-2, typical vocab sizes are in this range
    batch_size = 20
    seq_len = 35  # typical sequence length during training

    # Instantiate the model with hyperparameters tuned for SOTA performance.
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.2,         # slightly higher dropout for regularization
        max_seq_len=5000
    )
    
    # Generate a subsequent mask for autoregressive training.
    src_mask = model.generate_square_subsequent_mask(seq_len)
    
    # Dummy input: batch of token indices of shape (batch_size, seq_len)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass: returns logits of shape (seq_len, batch_size, vocab_size)
    logits = model(dummy_input, src_mask)
    print("Logits shape:", logits.shape)
    
    # Example loss calculation: shift targets by one.
    # Flatten logits and targets for computing cross-entropy loss.
    criterion = nn.CrossEntropyLoss()
    # Reshape logits: (seq_len * batch_size, vocab_size)
    logits_flat = logits.view(-1, vocab_size)
    # Assume target tokens are dummy_input shifted by one position.
    target = dummy_input[:, 1:]  # (batch_size, seq_len-1)
    # For simplicity, take first seq_len-1 tokens from logits.
    logits_flat = logits[:-1].reshape(-1, vocab_size)
    target = target.reshape(-1)
    loss = criterion(logits_flat, target)
    print("Loss:", loss.item())
    
    # Perplexity can be computed as the exponential of the loss.
    perplexity = math.exp(loss.item())
    print("Perplexity:", perplexity)