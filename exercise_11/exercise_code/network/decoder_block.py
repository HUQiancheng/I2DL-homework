from torch import nn
import torch

from ..network import MultiHeadAttention
from ..network import FeedForwardNeuralNetwork
from ..util.transformer_util import create_causal_mask

class DecoderBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.0):
        """

        Args:
            d_model: Dimension of Embedding
            d_k: Dimension of Keys and Queries
            d_v: Dimension of Values
            n_heads: Number of Attention Heads
            d_ff: Dimension of hidden layer
            dropout: Dropout probability
        """
        super().__init__()

        self.causal_multi_head = None
        self.layer_norm1 = None
        self.cross_multi_head = None
        self.layer_norm2 = None
        self.ffn = None
        self.layer_norm3 = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Initialize the Decoder Block                               #
        #            You will need:                                            #
        #                           - Causal Multi-Head Self-Attention layer   #
        #                           - Layer Normalization                      #
        #                           - Multi-Head Cross-Attention layer         #
        #                           - Layer Normalization                      #
        #                           - Feed forward neural network layer        #
        #                           - Layer Normalization                      #
        #                                                                      #
        # Hint 9: Check out the pytorch layer norm module                      #
        ########################################################################

        self.causal_multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.cross_multi_head = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNeuralNetwork(d_model, d_ff, dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self,
                inputs: torch.Tensor,
                context: torch.Tensor,
                causal_mask: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        """

        Args:
            inputs: Inputs from the Decoder
            context: Context from the Encoder
            causal_mask: Mask used for Causal Self Attention
            pad_mask: Optional Padding Mask used for Cross Attention

        Shape: 
            - inputs: (batch_size, sequence_length_decoder, d_model)
            - context: (batch_size, sequence_length_encoder, d_model)
            - causal_mask: (batch_size, sequence_length_decoder, sequence_length_decoder)
            - pad_mask: (batch_size, sequence_length_decoder, sequence_length_encoder)
            - outputs: (batch_size, sequence_length_decoder, d_model)
        """
        outputs = None

        ########################################################################
        # TODO:                                                                #
        #   Task 9: Implement the forward pass of the decoder block            #
        #   Task 12: Pass on the padding mask                                  #
        #                                                                      #
        # Hint 9:                                                              #
        #       - Don't forget the residual connections!                       #
        #       - Remember where we need the causal mask, forget about the     #
        #         other mask for now!                                          #
        # Hints 12:                                                            #
        #       - We have already combined the causal_mask with the pad_mask   #
        #         for you, all you have to do is pass it on to the "other"     #
        #         module                                                       #
        ########################################################################

        # Causal Multi-Head Self-Attention Layer
        causal_attention_output = self.causal_multi_head(inputs, inputs, inputs, causal_mask)
        causal_attention_output = self.dropout(causal_attention_output)
        causal_attention_output = self.layer_norm1(inputs + causal_attention_output)

        # Multi-Head Cross-Attention Layer
        cross_attention_output = self.cross_multi_head(causal_attention_output, context, context, pad_mask)
        cross_attention_output = self.dropout(cross_attention_output)
        cross_attention_output = self.layer_norm2(causal_attention_output + cross_attention_output)

        # Feed-Forward Neural Network
        ffn_output = self.ffn(cross_attention_output)
        ffn_output = self.dropout(ffn_output)
        outputs = self.layer_norm3(cross_attention_output + ffn_output)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return outputs
