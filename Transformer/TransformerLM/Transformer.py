import math, torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Create sinusoidal positional encodings.
        Args:
            d_model (int): Embedding dimensionality.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor of the same shape with positional encodings added.
        """
        x = x + self.pe[:, :x.size(0)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=5000,
    ):
        """
        A Transformer-based language model.
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length.
        """
        super(TransformerLM, self).__init__()
        self.model_type = "TransformerLM"
        self.d_model = d_model

        # Token embedding layer.
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding.
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder: a stack of TransformerEncoderLayers.
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Final linear layer projects encoder outputs to vocabulary logits.
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights uniformly.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with -inf.
        Args:
            sz (int): Size of the sequence (seq_len).
        Returns:
            Tensor: Mask of shape (sz, sz)
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor of token indices of shape (batch_size, seq_len)
            src_mask: Optional mask tensor for the transformer.
        Returns:
            logits: Tensor of shape (seq_len, batch_size, vocab_size)
        """
        # PyTorch's Transformer modules expect input shape: (seq_len, batch_size, d_model)
        src = src.t()  # transpose: (seq_len, batch_size)
        embedded = self.embedding(src) * math.sqrt(self.d_model)  # scale embeddings
        embedded = self.pos_encoder(embedded)  # add positional encoding
        output = self.transformer_encoder(embedded, src_mask)  # (seq_len, batch_size, d_model)
        logits = self.fc_out(output)  # project to vocabulary size
        return logits