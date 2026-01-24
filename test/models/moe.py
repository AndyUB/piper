"""
MoE implementation: https://github.com/peytontolbert/simple-moe
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Any, Tuple

class ExpertBase(nn.Module):
    """Base class for expert networks in the Mixture of Experts model."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize the expert network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dim: Dimension of hidden layer (if None, uses 4x input_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or 4 * input_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert."""
        return self.network(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get expert configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim
        }


class FFNExpert(ExpertBase):
    """Feed-forward neural network expert implementation."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize the FFN expert.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__(input_dim, output_dim, hidden_dim)
        
        layers = []
        current_dim = input_dim
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = self.hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'num_layers': len(self.network) // 3 + 1,
            'dropout': self.network[2].p if len(self.network) > 2 else 0.0
        })
        return config 


class MixtureOfExperts(nn.Module):
    """Mixture of Experts model implementation."""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 num_experts: int,
                 expert_class: type = FFNExpert,
                 expert_kwargs: Optional[Dict[str, Any]] = None,
                 k: int = 1,
                 capacity_factor: float = 1.0,
                 router_noise_epsilon: float = 1e-2):
        """
        Initialize the MoE model.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            num_experts: Number of experts
            expert_class: Expert class to use
            expert_kwargs: Additional arguments for expert initialization
            k: Number of experts to route to
            capacity_factor: Expert capacity multiplier
            router_noise_epsilon: Noise factor for router load balancing
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k

        assert k == 1
        
        # Initialize experts
        expert_kwargs = expert_kwargs or {}
        self.experts = nn.ModuleList([
            expert_class(input_dim=input_dim, output_dim=output_dim, **expert_kwargs)
            for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the MoE model.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of:
            - output: Model output of shape [batch_size, output_dim]
            - aux_loss: Auxiliary load balancing loss (None if not training)
        """

        batch_size = x.shape[0]
        
        outputs = []
        for batch_idx in range(batch_size):
            expert_idx = batch_idx % self.num_experts
            batch = x[batch_idx:batch_idx+1]
            with torch.fx.traceback.annotate({"expert": expert_idx}):
                expert_output = self.experts[expert_idx](batch)
            outputs.append(expert_output)
        output = torch.stack(outputs)
            
        return output

    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_experts': self.num_experts,
            'k': self.k,
            'expert_class': self.experts[0].__class__.__name__,
            'expert_config': self.experts[0].get_config()
        } 