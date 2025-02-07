
class LoRA2d(nn.Module):
    """
    A Conv2d layer with an added LoRA (Low-Rank Adaptation) update.
    
    The forward pass computes:
    
        y = conv(x) + scaling * (lora_up(lora_down(x)))
    
    where:
      - conv is a standard convolution (whose weights can optionally be frozen),
      - lora_down is a convolution that reduces the input channels to `rank` (using the same
        kernel size, stride, padding, etc. as `conv`),
      - lora_up is a 1x1 convolution that maps from `rank` channels back to `out_channels`,
      - scaling = lora_alpha / rank.
      
    Typically, lora_up is initialized to zero so that initially the module behaves exactly
    like the original conv.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        rank=4,
        lora_alpha=1.0,
        lora_dropout=0.0,
    ):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        # scaling factor: note that if rank==0, we could choose to skip the LoRA branch.
        self.scaling = lora_alpha / rank if rank > 0 else 1.0
        
        # Optional dropout applied before the LoRA branch
        self.lora_dropout = nn.Dropout2d(lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        
        if rank > 0:
            # LoRA "down" layer: uses the same kernel size and other hyperparameters so that the 
            # sliding-window (patch) structure is preserved.
            self.lora_down = nn.Conv2d(
                in_channels, rank, kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=False
            )
            # LoRA "up" layer: a 1x1 convolution to bring the channels back to out_channels.
            self.lora_up = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            
            # Initialize the LoRA layers:
            # It is common to initialize the up-projection to zero so that initially the LoRA branch
            # contributes nothing.
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
        else:
            # If rank == 0, the LoRA branch is omitted.
            self.lora_down = None
            self.lora_up = None

    def forward(self, x):
        
        # Add LoRA update if enabled.
        if self.rank > 0:
            # Optionally apply dropout on the input.
            lora_x = self.lora_dropout(x)
            # Compute the low-rank update.
            lora_update = self.lora_up(self.lora_down(lora_x))
            # Scale and add to the original conv output.
            y = self.scaling * lora_update
        
        return y

class LoRALinear(nn.Module):
    """
    A Linear layer with an added LoRA (Low-Rank Adaptation) update.

    The forward pass computes:

        y = linear(x) + scaling * (lora_up(lora_down(x)))

    where:
      - linear is the standard linear layer,
      - lora_down is a linear layer mapping from `in_features` to `rank` (with no bias),
      - lora_up is a linear layer mapping from `rank` to `out_features` (with no bias),
      - scaling = lora_alpha / rank.

    Typically, lora_up is initialized to zeros so that initially the module behaves like
    the original linear layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.rank = rank
        self.lora_alpha = lora_alpha
        # The scaling factor ensures the LoRA branch is properly scaled.
        self.scaling = lora_alpha / rank if rank > 0 else 1.0
        # Optional dropout for the LoRA branch.
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        if rank > 0:
            # LoRA "down" projection: from in_features -> rank (no bias)
            self.lora_down = nn.Linear(in_features, rank, bias=False)
            # LoRA "up" projection: from rank -> out_features (no bias)
            self.lora_up = nn.Linear(rank, out_features, bias=False)
            # Initialize lora_down using kaiming initialization
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            # Initialize lora_up to zeros so that the update starts off as zero.
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_down = None
            self.lora_up = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard linear transformation.
        if self.rank > 0:
            # Apply dropout (if any) to the input.
            lora_x = self.lora_dropout(x)
            # Compute the low-rank update.
            lora_update = self.lora_up(self.lora_down(lora_x))
            # Add the scaled update to the original output.
            y = self.scaling * lora_update
        return y
