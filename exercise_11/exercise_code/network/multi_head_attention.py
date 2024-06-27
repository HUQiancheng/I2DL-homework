from torch import nn
import torch    
from ..network import ScaledDotAttention

class MultiHeadAttention(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 dropout: float = 0.0):
        """
        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            dropout: Dropout probability
        """
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Initialize all weight layers as linear layers                #
        #       - Initialize the ScaledDotAttention                            #
        #       - Initialize the projection layer as a linear layer            #
        #  Task 13:                                                            #
        #       - Initialize the dropout layer (torch.nn implementation)       #
        #                                                                      #
        ########################################################################

        self.weights_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.weights_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.attention = ScaledDotAttention(d_k, dropout)
        self.project = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Mask

        Shape:
            - q: (batch_size, sequence_length_queries, d_model)
            - k: (batch_size, sequence_length_keys, d_model)
            - v: (batch_size, sequence_length_keys, d_model)
            - mask: (batch_size, sequence_length_queries, sequence_length_keys)
            - outputs: (batch_size, sequence_length_queries, d_model)
        """

        # You will need these here!
        batch_size, sequence_length_queries, _ = q.size()
        _, sequence_length_keys, _ = k.size()

        ########################################################################
        # TODO:                                                                #
        #   Task 3:                                                            #
        #       - Pass q,k and v through the linear layer                      #
        #       - Split the last dimensions into n_heads and d_k or d_v        #
        #       - Swap the dimensions so that the shape matches the required   #
        #         input shapes of the ScaledDotAttention layer                 #
        #       - Pass them through the ScaledDotAttention layer               #
        #       - Swap the dimensions of the output back                       #
        #       - Combine the last two dimensions again                        #
        #       - Pass the outputs through the projection layer                #
        #   Task 8:                                                            #
        #       - If a mask is given, add an empty dimension at dim=1          #
        #       - Pass the mask to the ScaledDotAttention layer                #
        #  Task 13:                                                            #
        #       - Add dropout as a final step after the projection layer       #
        #                                                                      #
        ########################################################################

        def shape(x, d):
            return x.view(batch_size, -1, self.n_heads, d).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        q = shape(self.weights_q(q), self.d_k)
        k = shape(self.weights_k(k), self.d_k)
        v = shape(self.weights_v(v), self.d_v)

        if mask is not None:
            mask = mask.unsqueeze(1)

        x = self.attention(q, k, v, mask=mask)

        x = unshape(x)
        outputs = self.project(x)
        outputs = self.dropout(outputs)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
