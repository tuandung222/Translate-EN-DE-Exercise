"""
Model architectures for Seq2seq translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StackedGRUEncoder(nn.Module):
    """Stacked GRU Encoder for sequence-to-sequence translation."""

    def __init__(self, input_size, hidden_size, num_layers=8, dropout_p=0.1):
        super(StackedGRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_p,
        )
        self._initialize_weights()

    def forward(self, input, hidden):
        # input: (batch, seq_len) - token indices
        # hidden: (num_layers, batch, hidden_size) - initial hidden state
        embedded = self.embedding(input)  # (batch, seq_len, hidden_size)
        output = embedded  # (batch, seq_len, hidden_size)
        output, hidden = self.gru(
            output, hidden
        )  # output: (batch, seq_len, hidden_size), hidden: (num_layers, batch, hidden_size)
        return (
            output,
            hidden,
        )  # output: (batch, seq_len, hidden_size), hidden: (num_layers, batch, hidden_size)

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=torch.bfloat16
        )  # (num_layers, batch_size, hidden_size)


class StackedGRUDecoder(nn.Module):
    """Stacked GRU Decoder without attention mechanism."""

    def __init__(self, hidden_size, output_size, num_layers=8, dropout_p=0.1):
        super(StackedGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self._initialize_weights()

    def forward(self, input, hidden, encoder_outputs):
        # input: (batch, seq_len) - token indices
        # hidden: (num_layers, batch, hidden_size) - decoder hidden state
        # encoder_outputs: (batch, src_len, hidden_size) - not used (no attention)
        embedded = self.embedding(input)  # (batch, seq_len, hidden_size)
        embedded = self.dropout(embedded)  # (batch, seq_len, hidden_size)
        output, hidden = self.gru(
            embedded, hidden
        )  # output: (batch, seq_len, hidden_size), hidden: (num_layers, batch, hidden_size)
        output = F.log_softmax(self.out(output), dim=-1)  # (batch, seq_len, output_size)
        return (
            output,
            hidden,
            None,
        )  # output: (batch, seq_len, output_size), hidden: (num_layers, batch, hidden_size), attn_weights: None

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=torch.bfloat16
        )  # (num_layers, batch_size, hidden_size)


class AdditiveAttention(nn.Module):
    """
    Implements the additive attention as proposed in
    "Neural Machine Translation by Jointly Learning to Align and Translate".
    """

    def __init__(self, q_dim, k_dim, attn_dim):
        super(AdditiveAttention, self).__init__()
        self.proj_q = nn.Linear(q_dim, attn_dim, bias=False)
        self.proj_k = nn.Linear(k_dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))  # (attn_dim,)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, query, key, value):
        # query: (batch, T_q, q_dim) - decoder hidden states
        # key: (batch, T_k, k_dim) - encoder outputs
        # value: (batch, T_v, d_v) - encoder outputs (same as key)
        q_ = self.proj_q(query)  # (batch, T_q, attn_dim)
        k_ = self.proj_k(key)  # (batch, T_k, attn_dim)
        q_ = q_.unsqueeze(-2)  # (batch, T_q, 1, attn_dim)
        k_ = k_.unsqueeze(-3)  # (batch, 1, T_k, attn_dim)
        attn_hid = torch.tanh(
            q_ + k_ + self.bias
        )  # (batch, T_q, T_k, attn_dim) - broadcast addition
        attn_logits = self.w(attn_hid)  # (batch, T_q, T_k, 1)
        attn_logits = attn_logits.squeeze(-1)  # (batch, T_q, T_k)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, T_q, T_k) - attention distribution
        output = torch.matmul(attn_weights, value)  # (batch, T_q, d_v) - context vector
        return output, attn_weights  # output: (batch, T_q, d_v), attn_weights: (batch, T_q, T_k)


class StackedGRUAttnDecoder(nn.Module):
    """Stacked GRU Decoder with additive attention mechanism."""

    def __init__(self, hidden_size, output_size, num_layers=8, dropout_p=0.1, max_length=64):
        super(StackedGRUAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p,
        )
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = AdditiveAttention(self.hidden_size, self.hidden_size, self.hidden_size // 2)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self._initialize_weights()

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # (batch, seq_len) -> (batch, seq_len, hidden_size)
        embedded = self.dropout(embedded)  # (batch, seq_len, hidden_size)
        x, hidden = self.gru(
            embedded, hidden
        )  # (batch, seq_len, hidden_size), (num_layers, batch, hidden_size)
        context, attn_weights = self.attn(
            query=x, key=encoder_outputs, value=encoder_outputs
        )  # (batch, seq_len, hidden_size), (batch, seq_len, T_k)
        x_w_context = torch.cat((x, context), dim=-1)  # (batch, seq_len, hidden_size * 2)
        x_w_context = self.attn_combine(
            x_w_context
        )  # (batch, seq_len, hidden_size * 2) -> (batch, seq_len, hidden_size)
        output = F.log_softmax(
            self.out(x_w_context), dim=-1
        )  # (batch, seq_len, hidden_size) -> (batch, seq_len, output_size)

        return output, hidden, attn_weights

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device, dtype=torch.bfloat16
        )  # (num_layers, batch_size, hidden_size)


# Residual Models with Layer Normalization


class ResidualStackedGRUEncoder(nn.Module):
    """Stacked GRU Encoder with residual connections and layer normalization."""

    def __init__(self, input_size, hidden_size, num_layers=4, dropout_p=0.1):
        super(ResidualStackedGRUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(input_size, hidden_size)
        # Individual GRU layers for residual connections
        self.gru_layers = nn.ModuleList(
            [nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)]
        )
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_p)
        self._initialize_weights()

    def forward(self, input, hidden):
        # input: (batch, seq_len) - token indices
        # hidden: (num_layers, batch, hidden_size) - initial hidden state
        embedded = self.embedding(input)  # (batch, seq_len, hidden_size)
        embedded = self.dropout(embedded)  # (batch, seq_len, hidden_size)

        output = embedded  # (batch, seq_len, hidden_size)
        hiddens = []

        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            h = hidden[i : i + 1]  # (1, batch, hidden_size) - hidden for this layer
            gru_out, h_out = gru_layer(
                output, h
            )  # gru_out: (batch, seq_len, hidden_size), h_out: (1, batch, hidden_size)
            if i > 0:
                output = gru_out + output  # (batch, seq_len, hidden_size) - residual connection
            else:
                output = gru_out  # (batch, seq_len, hidden_size)
            output = layer_norm(output)  # (batch, seq_len, hidden_size) - layer normalization
            if i < self.num_layers - 1:
                output = self.dropout(output)  # (batch, seq_len, hidden_size) - dropout
            hiddens.append(h_out)  # collect hidden states

        hidden = torch.cat(hiddens, dim=0)  # (num_layers, batch, hidden_size)

        return (
            output,
            hidden,
        )  # output: (batch, seq_len, hidden_size), hidden: (num_layers, batch, hidden_size)

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class ResidualStackedGRUDecoder(nn.Module):
    """Stacked GRU Decoder with residual connections and layer normalization (no attention)."""

    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.3):
        super(ResidualStackedGRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru_layers = nn.ModuleList(
            [nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self._initialize_weights()

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        output = embedded
        hiddens = []
        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            h = hidden[i : i + 1]
            gru_out, h_out = gru_layer(output, h)
            if i > 0:
                output = gru_out + output
            else:
                output = gru_out
            output = layer_norm(output)
            if i < self.num_layers - 1:
                output = self.dropout(output)
            hiddens.append(h_out)
        hidden = torch.cat(hiddens, dim=0)
        output = F.log_softmax(self.out(output), dim=-1)
        return output, hidden, None

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class ResidualStackedGRUAttnDecoder(nn.Module):
    """Stacked GRU Decoder with residual connections, layer normalization, and attention."""

    def __init__(self, hidden_size, output_size, num_layers=4, dropout_p=0.3, max_length=64):
        super(ResidualStackedGRUAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        #  GRU layers for residual connections
        self.gru_layers = nn.ModuleList(
            [nn.GRU(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.attn = AdditiveAttention(self.hidden_size, self.hidden_size, self.hidden_size // 2)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self._initialize_weights()

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        x = embedded
        hiddens = []

        for i, (gru_layer, layer_norm) in enumerate(zip(self.gru_layers, self.layer_norms)):
            h = hidden[i : i + 1]
            gru_out, h_out = gru_layer(x, h)
            if i > 0:
                x = gru_out + x
            else:
                x = gru_out
            x = layer_norm(x)
            if i < self.num_layers - 1:
                x = self.dropout(x)
            hiddens.append(h_out)
        hidden = torch.cat(hiddens, dim=0)
        context, attn_weights = self.attn(query=x, key=encoder_outputs, value=encoder_outputs)
        x_w_context = torch.cat((x, context), dim=-1)
        x_w_context = self.attn_combine(x_w_context)
        output = F.log_softmax(self.out(x_w_context), dim=-1)

        return output, hidden, attn_weights

    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def initHidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
