import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    LinearBlock,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    SinusoidalPositionalEncoding)

"""
Reconstruction-Driven Anomaly Detection approach using Transformer-based
model architecture.
"""
class Transformer_Recon_Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        """
        Encoder Section.
        """
        self.positional_encoding = SinusoidalPositionalEncoding(kwargs["embedding_dim"], dropout=kwargs["dropout"])

        # Encooder's learnable representation embeddings.
        self.num_encodings = kwargs["num_encodings"]
        self.encoders_embeddings = nn.parameter.Parameter(
            torch.randn(
                self.num_encodings,
                kwargs["embedding_dim"]))  # (Seq_enc,Dim_emb)

        # Transforms BATADAL vector dimension into embedding dimension.
        self.init_encoder_linear_block = nn.Sequential(
            LinearBlock(
                use_activation=True,
                in_dim=kwargs["in_dim"],
                out_dim=kwargs["hidden_dim"],
                activation_type=kwargs["activation_type"]),
            LinearBlock(
                use_activation=True,
                in_dim=kwargs["hidden_dim"],
                out_dim=kwargs["embedding_dim"],
                activation_type=kwargs["activation_type"])
        )  # (Batch,Seq_window,Dim_emb)

        self.encoder_transformer_blocks = nn.ModuleList()
        for _ in range(kwargs["num_enc_blocks"] - 1):
            self.encoder_transformer_blocks.append(
                TransformerEncoderBlock(
                    heads=kwargs["num_heads"],
                    embedding_dim=kwargs["embedding_dim"],
                    hidden_dim=kwargs["hidden_dim"],
                    dropout=kwargs["dropout"],
                    activation_type=kwargs["activation_type"]))  # (Batch,Seq_window + Seq_enc,Dim_emb)

        self.encoder_transformer_blocks.append(
            TransformerEncoderBlock(
                heads=kwargs["num_heads"],
                embedding_dim=kwargs["embedding_dim"],
                hidden_dim=kwargs["hidden_dim"],
                dropout=0.0,
                activation_type=kwargs["activation_type"]))  # (Batch,Seq_window + Seq_enc,Dim_emb)

        """
        Decoder Section.
        """
        # Decoder's learnable representation embeddings.
        self.num_window = kwargs["num_window"]
        self.decoder_embedding = nn.parameter.Parameter(
            torch.randn(
                self.num_window,  # BATADAL window sequence i.e. 12 or 24.
                kwargs["embedding_dim"]))  # (Seq_window,Dim_emb)

        self.decoder_transformer_blocks = nn.ModuleList()
        for _ in range(kwargs["num_dec_blocks"]):
            self.decoder_transformer_blocks.append(
                TransformerDecoderBlock(
                    heads=kwargs["num_heads"],
                    embedding_dim=kwargs["embedding_dim"],
                    hidden_dim=kwargs["hidden_dim"],
                    dropout=kwargs["dropout"],
                    activation_type=kwargs["activation_type"]))  # (Batch,Seq_window,Dim_emb)

        self.recon_block = nn.Sequential(
            nn.LayerNorm(kwargs["embedding_dim"]),
            LinearBlock(
                use_activation=True,
                in_dim=kwargs["embedding_dim"],
                out_dim=kwargs["hidden_dim"],
                activation_type=kwargs["activation_type"]),
            LinearBlock(
                use_activation=False,
                in_dim=kwargs["hidden_dim"],
                out_dim=kwargs["in_dim"]),
        )  # (Bach,Seq_window,Dim_in)

    def forward_encoder(self, x):
        N, _, _ = x.shape
        x_enc = self.encoders_embeddings[None,:,:].repeat(N,1,1)  # (Batch,Seq_enc,Dim_emb)

        x = self.init_encoder_linear_block(x)  # (Batch,Seq_window,Dim_emb)
        x = torch.cat((x_enc, x), dim=1)  #  (Batch,Seq_enc+Seq_window,Dim_emb)

        x = self.positional_encoding(x)

        for encoder_transformer_block in self.encoder_transformer_blocks:
            x = encoder_transformer_block(x)

        x_enc = x[:, :self.num_encodings, :]
        return x_enc

    def forward_decoder(self, x_enc):
        N, _, _ = x_enc.shape
        x = self.decoder_embedding[None,:,:].repeat(N,1,1)  # (Batch,Seq_window,Dim_emb)

        for decoder_transformer_block in self.decoder_transformer_blocks:
            x = decoder_transformer_block(x, x_enc)  # (Batch,Seq_window,Dim_emb)
        
        x = self.recon_block(x)  # (Batch,Seq_window,Dim_in)
        return x

    def forward(self, x):
        x_enc = self.forward_encoder(x)
        x_dec = self.forward_decoder(x_enc)
        return x_dec

