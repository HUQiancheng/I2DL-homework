from torch import nn
import torch
from ..network import SCORE_SAVER

class ScaledDotAttention(nn.Module):

    def __init__(self,
                 d_k,
                 dropout: float = 0.0):
        """
        Args:
            d_k: Dimension of Keys and Queries
            dropout: Dropout probability
        """
        super().__init__()
        self.d_k = d_k

        ########################################################################
        # TODO:                                                                #
        #   Task 2: Initialize the softmax layer (torch.nn implementation)     #
        #                                                                      #           
        ########################################################################

        self.softmax = nn.Softmax(dim=-1)
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
        Computes the scaled dot attention given query, key and value inputs. Stores the scores in SCORE_SAVER for
        visualization

        Args:
            q: Query Inputs
            k: Key Inputs
            v: Value Inputs
            mask: Optional Causal or Padding Boolean Mask

        Shape:
            - q: (*, sequence_length_queries, d_model)
            - k: (*, sequence_length_keys, d_model)
            - v: (*, sequence_length_keys, d_model)
            - mask: (*, sequence_length_queries, sequence_length_keys)
            - outputs: (*, sequence_length_queries, d_v)
        """
        ########################################################################
        # TODO:                                                                #
        #   Task 2:                                                            #
        #       - Calculate the scores using the queries and keys              #
        #       - Normalize the scores using the softmax function              #
        #       - Compute the updated embeddings and return the output         #
        #   Task 8:                                                            #
        #       - Add a negative infinity mask if a mask is given              #
        #   Task 13:                                                           #
        #       - Add dropout to the scores right BEFORE the final outputs     #
        #         (scores * V) are calculated                                  #
        ########################################################################

        # Calculate the scores using the queries and keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Add a negative infinity mask if a mask is given
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Normalize the scores using the softmax function
        scores = self.softmax(scores)
        
        # Add dropout to the scores
        scores = self.dropout(scores)
        
        # Compute the updated embeddings and return the output
        outputs = torch.matmul(scores, v)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        SCORE_SAVER.save(scores)

        return outputs
