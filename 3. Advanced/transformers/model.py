import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Transformer model from scratch.

The Transformer model consists of an encoder and a decoder. The encoder processes the input sequence
and generates a fixed-size representation of the input. The decoder takes this representation and
generates the output sequence.
"""

class PositionalEncoding(nn.Module):
    """
    Unlike the original paper which uses sinusoidal positional encodings (fixed),
    this implementation uses learned positional embeddings. This is because 
    learned positional embeddings have been shown to be more effective in practice (e.g. GPT, BERT).
    """
    def __init__(self, max_len, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), seq_len) # (batch_size, seq_len)
        pos_enc = self.pos_embedding(positions)
        return x + pos_enc


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.d_k = embed_dim // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear operation and split into num_heads
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None: # Mask padding tokens if provided
            scores = scores.masked_fill(mask == 0, -1e10)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)

        # Concatenate heads and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k * self.num_heads)
        output = self.out(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout(attn_output)) # Skip connection
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output)) # Skip connection
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.enc_attn = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.layernorm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, enc_mask=None):
        attn_output = self.self_attn(x, x, x, self_mask)
        x = self.layernorm1(x + self.dropout(attn_output)) # Skip connection
        attn_output = self.enc_attn(x, enc_out, enc_out, enc_mask)
        x = self.layernorm2(x + self.dropout(attn_output)) # Skip connection
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout(ffn_output)) # Skip connection
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, pad_idx=0, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class TransformerClassifier(nn.Module):
    """
    Transformer model (Encoder-only) for text classification.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, src, src_mask=None):
        enc_output = self.encoder(src, src_mask)
        pooled_output = enc_output.mean(dim=1)
        output = self.fc_out(pooled_output)
        return output


class TransformerGenerator(nn.Module):
    """
    Transformer model (Decoder-only) for text generation.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout=0.1):
        super(TransformerGenerator, self).__init__()
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_mask=None):
        dec_output = self.decoder(tgt, None, None, tgt_mask)
        output = self.fc_out(dec_output)
        return output


class TransformerTranslator(nn.Module):
    """
    Transformer model (Encoder-Decoder) for translation tasks.
    """
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size_src, vocab_size_tgt, max_len, dropout=0.1):
        super(TransformerTranslator, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, vocab_size_src, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, vocab_size_tgt, max_len, dropout)
        self.fc_out = nn.Linear(d_model, vocab_size_tgt)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output
