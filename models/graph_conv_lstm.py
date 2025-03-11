import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer as described in Kipf & Welling (2017)
    with modifications for time series data.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights using Glorot & Bengio (2010)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass of Graph Convolution
        
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        # Apply graph convolution: AXW where A is adjacency, X is input, W is weight
        # Compute XW first
        support = torch.bmm(x, self.weight.unsqueeze(0).expand(x.size(0), -1, -1))
        
        # Then multiply by A: AXW
        output = torch.bmm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class ChebGraphConv(nn.Module):
    """
    Chebyshev Graph Convolutional Layer for higher-order graph convolutions
    using Chebyshev polynomials of the first kind.
    """
    def __init__(self, in_features, out_features, K=3, bias=True):
        super(ChebGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K  # Order of Chebyshev polynomial
        
        # Learnable weight matrix for each order
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(K)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, laplacian):
        """
        Forward pass for Chebyshev Graph Convolution
        
        Args:
            x: Node features (batch_size, num_nodes, in_features)
            laplacian: Normalized Laplacian (batch_size, num_nodes, num_nodes)
            
        Returns:
            Output features (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, in_features = x.size()
        
        # Initialize Chebyshev polynomials
        Tx_0 = x  # T_0(L)x = x
        out = torch.bmm(Tx_0, self.weights[0].unsqueeze(0).expand(batch_size, -1, -1))
        
        if self.K > 1:
            Tx_1 = torch.bmm(laplacian, x)  # T_1(L)x = Lx
            out = out + torch.bmm(Tx_1, self.weights[1].unsqueeze(0).expand(batch_size, -1, -1))
            
            # Recurrence relation for higher order terms: T_k(L) = 2LT_{k-1}(L) - T_{k-2}(L)
            for k in range(2, self.K):
                Tx_2 = 2 * torch.bmm(laplacian, Tx_1) - Tx_0
                out = out + torch.bmm(Tx_2, self.weights[k].unsqueeze(0).expand(batch_size, -1, -1))
                Tx_0, Tx_1 = Tx_1, Tx_2
        
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important timesteps
    """
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Attention mechanism
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.W_o = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        """
        Apply temporal attention to the input sequence
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            
        Returns:
            Attended tensor (batch_size, seq_len, hidden_size)
        """
        # Linear projections for query, key, value
        q = self.W_q(x)  # (batch_size, seq_len, hidden_size)
        k = self.W_k(x)  # (batch_size, seq_len, hidden_size)
        v = self.W_v(x)  # (batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.hidden_size)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.bmm(attention_weights, v)
        
        # Apply output projection
        output = self.W_o(context)
        
        # Residual connection and layer normalization
        return self.layer_norm(output + x)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important nodes in the graph
    """
    def __init__(self, hidden_size):
        super(SpatialAttention, self).__init__()
        self.hidden_size = hidden_size
        
        # Spatial attention mechanism
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.W_t = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(hidden_size, 1))
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.v)
    
    def forward(self, x, adj=None):
        """
        Apply spatial attention to focus on important nodes
        
        Args:
            x: Input tensor (batch_size, num_nodes, hidden_size)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
            
        Returns:
            Attended tensor (batch_size, num_nodes, hidden_size)
        """
        batch_size, num_nodes, _ = x.size()
        
        # Compute attention scores
        s_scores = self.W_s(x)  # (batch_size, num_nodes, hidden_size)
        t_scores = self.W_t(x)  # (batch_size, num_nodes, hidden_size)
        
        # Combine spatial and temporal scores
        scores = torch.tanh(s_scores.unsqueeze(2) + t_scores.unsqueeze(1))
        
        # Apply attention vector
        scores = scores.view(-1, self.hidden_size)
        scores = torch.matmul(scores, self.v).view(batch_size, num_nodes, num_nodes)
        
        # If adjacency matrix is provided, constrain attention to graph structure
        if adj is not None:
            # Use adjacency matrix as a mask (only connected nodes attend to each other)
            scores = scores.masked_fill(adj == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights
        context = torch.bmm(attention_weights, x)
        
        # Apply layer normalization
        return self.layer_norm(context)


class GraphLSTMCell(nn.Module):
    """
    Graph-augmented LSTM Cell that incorporates graph structure
    into the traditional LSTM computation.
    """
    def __init__(self, input_size, hidden_size, num_nodes, bias=True):
        super(GraphLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        
        # Traditional LSTM gates
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size, bias)
        
        # Graph convolution for incorporating spatial information
        self.graph_conv = GraphConvolution(hidden_size, hidden_size, bias)
        
        # Spatial attention mechanism
        self.spatial_attention = SpatialAttention(hidden_size)
        
        # Combine graph and LSTM representations
        self.combine = nn.Linear(2 * hidden_size, hidden_size)
        
    def forward(self, input, adj, hidden=None):
        """
        Forward pass for the Graph LSTM Cell
        
        Args:
            input: Input tensor (batch_size, num_nodes, input_size)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes)
            hidden: Tuple of (h, c) where h and c are tensors of shape 
                  (batch_size, num_nodes, hidden_size)
                  
        Returns:
            h_new, c_new: New hidden and cell states
        """
        batch_size = input.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = torch.zeros(batch_size, self.num_nodes, self.hidden_size, device=input.device)
            c = torch.zeros(batch_size, self.num_nodes, self.hidden_size, device=input.device)
        else:
            h, c = hidden
        
        # Reshape for LSTM cell input
        input_reshaped = input.view(batch_size * self.num_nodes, self.input_size)
        h_reshaped = h.view(batch_size * self.num_nodes, self.hidden_size)
        c_reshaped = c.view(batch_size * self.num_nodes, self.hidden_size)
        
        # Apply LSTM cell
        h_lstm, c_lstm = self.lstm_cell(input_reshaped, (h_reshaped, c_reshaped))
        
        # Reshape back
        h_lstm = h_lstm.view(batch_size, self.num_nodes, self.hidden_size)
        c_lstm = c_lstm.view(batch_size, self.num_nodes, self.hidden_size)
        
        # Apply graph convolution
        h_graph = self.graph_conv(h_lstm, adj)
        
        # Apply spatial attention
        h_attended = self.spatial_attention(h_graph, adj)
        
        # Combine LSTM and graph outputs
        h_combined = torch.cat([h_lstm, h_attended], dim=-1)
        h_new = self.combine(h_combined)
        
        return h_new, c_lstm


class GraphLSTM(nn.Module):
    """
    Graph LSTM layer that processes a sequence using GraphLSTMCell
    """
    def __init__(self, input_size, hidden_size, num_nodes, num_layers=1, bias=True, dropout=0.0):
        super(GraphLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create a list of GraphLSTMCells
        self.graph_lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.graph_lstm_cells.append(GraphLSTMCell(layer_input_size, hidden_size, num_nodes, bias))
    
    def forward(self, x, adj, hidden=None):
        """
        Forward pass for Graph LSTM
        
        Args:
            x: Input tensor (batch_size, seq_len, num_nodes, input_size)
            adj: Adjacency matrix - either:
                - Static: (batch_size, num_nodes, num_nodes)
                - Dynamic: (batch_size, seq_len, num_nodes, num_nodes)
            hidden: Initial hidden states, tuple of (h, c) where h and c are tensors
                   of shape (num_layers, batch_size, num_nodes, hidden_size)
                   
        Returns:
            output: Output tensor (batch_size, seq_len, num_nodes, hidden_size)
            hidden: Final hidden states
        """
        batch_size, seq_len, num_nodes, input_size = x.size()
        
        # Check if we have a dynamic adjacency matrix
        is_dynamic_adj = len(adj.shape) == 4 and adj.shape[1] == seq_len
        
        # Initialize hidden states if not provided
        if hidden is None:
            h = [torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, num_nodes, self.hidden_size, device=x.device) 
                 for _ in range(self.num_layers)]
        else:
            h, c = hidden
            h = [h[i] for i in range(self.num_layers)]
            c = [c[i] for i in range(self.num_layers)]
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]  # (batch_size, num_nodes, input_size)
            adj_t = adj[:, t] if is_dynamic_adj else adj  # Get current/static adjacency
            
            # Process each layer
            for layer in range(self.num_layers):
                # Apply dropout between layers
                if layer > 0 and self.dropout > 0:
                    x_t = F.dropout(x_t, p=self.dropout, training=self.training)
                
                # Process with GraphLSTMCell
                if layer == 0:
                    h[layer], c[layer] = self.graph_lstm_cells[layer](x_t, adj_t, (h[layer], c[layer]))
                else:
                    h[layer], c[layer] = self.graph_lstm_cells[layer](h[layer-1], adj_t, (h[layer], c[layer]))
            
            # Collect output from the last layer
            outputs.append(h[-1])
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, num_nodes, hidden_size)
        
        # Stack hidden states for return
        h_stacked = torch.stack(h, dim=0)  # (num_layers, batch_size, num_nodes, hidden_size)
        c_stacked = torch.stack(c, dim=0)  # (num_layers, batch_size, num_nodes, hidden_size)
        
        return outputs, (h_stacked, c_stacked)


class SpatioTemporalGraphCNN(nn.Module):
    """
    Spatiotemporal Graph Convolutional Neural Network for time-series prediction.
    Inspired by Yu et al. "Spatio-Temporal Graph Convolutional Networks" (2018).
    """
    def __init__(self, input_size, hidden_size, num_nodes, num_layers=2, kernel_size=3, dropout=0.3):
        super(SpatioTemporalGraphCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # Temporal convolution (1D)
        self.temporal_conv1 = nn.Conv1d(
            in_channels=input_size, 
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        
        # Spatial graph convolution
        self.spatial_gconv = GraphConvolution(hidden_size, hidden_size)
        
        # Second temporal convolution
        self.temporal_conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size//2
        )
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        """
        Forward pass for ST-GCN block
        
        Args:
            x: Input tensor (batch_size, seq_len, num_nodes, input_size)
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes) or 
                (batch_size, seq_len, num_nodes, num_nodes) for dynamic graphs
            
        Returns:
            Output tensor (batch_size, seq_len, num_nodes, hidden_size)
        """
        batch_size, seq_len, num_nodes, input_size = x.size()
        
        # Check if we have a dynamic adjacency matrix
        is_dynamic_adj = len(adj.shape) == 4 and adj.shape[1] == seq_len
        
        # Process each time step with ST-GCN
        outputs = []
        for t in range(seq_len):
            # Extract current time step features
            x_t = x[:, t]  # (batch_size, num_nodes, input_size)
            
            # Get adjacency matrix for current time step if dynamic
            adj_t = adj[:, t] if is_dynamic_adj else adj
            
            # Temporal convolution 1
            # Reshape for temporal conv: [batch*nodes, input_size, 1]
            x_t_reshaped = x_t.reshape(batch_size * num_nodes, input_size, 1)
            x_t = self.temporal_conv1(x_t_reshaped)  # [batch*nodes, hidden_size, 1]
            x_t = x_t.reshape(batch_size, num_nodes, self.hidden_size)  # [batch, nodes, hidden]
            
            # Spatial graph convolution
            x_st = self.spatial_gconv(x_t, adj_t)  # [batch, nodes, hidden]
            
            # Temporal convolution 2
            x_st_reshaped = x_st.reshape(batch_size * num_nodes, self.hidden_size, 1)
            x_st = self.temporal_conv2(x_st_reshaped)  # [batch*nodes, hidden_size, 1]
            x_st = x_st.reshape(batch_size, num_nodes, self.hidden_size)  # [batch, nodes, hidden]
            
            # Apply batch normalization and activation
            x_out = x_st.reshape(batch_size * num_nodes, self.hidden_size)
            x_out = F.relu(self.bn(x_out))
            x_out = self.dropout(x_out)
            
            # Reshape back to original dimensions
            x_out = x_out.reshape(batch_size, num_nodes, self.hidden_size)
            
            outputs.append(x_out)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs, dim=1)  # [batch, seq, nodes, hidden]
        
        return outputs


class CryptoGraphConvLSTM(nn.Module):
    """
    Enhanced cryptocurrency price prediction model using Graph Convolutional LSTM.
    Combines spatial (inter-cryptocurrency relationships) and temporal features
    to predict daily open and close prices for the next 6 months.
    """
    def __init__(self, input_size, hidden_size, num_nodes, num_layers=2, 
                 dropout=0.3, l2_reg=1e-5, use_attention=True):
        super(CryptoGraphConvLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        
        # Graph structure learning module
        self.graph_learn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Spatio-Temporal blocks
        self.st_blocks = nn.ModuleList([
            SpatioTemporalGraphCNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_nodes=num_nodes,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Graph LSTM for temporal modeling
        self.graph_lstm = GraphLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_nodes=num_nodes,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Temporal attention mechanism
        if use_attention:
            self.temporal_attention = TemporalAttention(hidden_size)
        
        # Feature aggregation layer
        self.feature_aggregate = nn.Linear(hidden_size * 2, hidden_size)
        
        # Output projection layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, 14 * 26)  # 6 months (26 weeks) * 2 values (open, close) * 7 days
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name and 'lstm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
    
    def _create_adjacency_matrix(self, x):
        """
        Create dynamic adjacency matrix based on feature similarity
        
        Args:
            x: Input features (batch_size, seq_len, num_nodes, input_size)
            
        Returns:
            Adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        
        # Use the features from the last timestamp to compute adjacency
        last_features = x[:, -1]  # (batch_size, num_nodes, input_size)
        
        # Compute similarity scores between nodes
        node_embeddings = self.graph_learn(last_features)  # (batch_size, num_nodes, hidden_size)
        
        # Compute adjacency using dot product similarity
        adj = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))  # (batch_size, num_nodes, num_nodes)
        
        # Apply softmax to normalize
        adj = F.softmax(adj, dim=2)
        
        # Add self-loops - each node is connected to itself
        identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + identity
        
        # Normalize adjacency matrix (D^(-1/2) A D^(-1/2))
        rowsum = adj.sum(dim=-1, keepdim=True)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt.squeeze(-1))
        adj_normalized = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        return adj_normalized
    
    def forward(self, x, adj):
        """
        Forward pass of the model
        
        Args:
            x: Input features (batch_size, seq_len, num_nodes, feat_dim)
            adj: Adjacency matrix - either:
                - Static: (batch_size, num_nodes, num_nodes)
                - Dynamic: (batch_size, seq_len, num_nodes, num_nodes)
            
        Returns:
            Predicted prices (batch_size, num_forecasts)
        """
        batch_size, seq_len, num_nodes, _ = x.size()
        
        # Check if we have a dynamic adjacency matrix
        is_dynamic_adj = len(adj.shape) == 4 and adj.shape[1] == seq_len
        
        # Create adjacency matrix if not provided
        if adj is None:
            adj = self._create_adjacency_matrix(x)
            is_dynamic_adj = False
        
        # Project and normalize input
        x_proj = self.input_proj(x)
        x_proj = self.input_norm(x_proj)
        x_proj = self.input_dropout(x_proj)
        
        # Process with ST-GCN blocks
        x_st = x_proj
        for block in self.st_blocks:
            # Apply block with appropriate adjacency matrix
            if is_dynamic_adj:
                x_st = block(x_st, adj) + x_st  # Residual connection
            else:
                x_st = block(x_st, adj) + x_st  # Residual connection
        
        # Process with Graph LSTM
        x_lstm, _ = self.graph_lstm(x_st, adj)
        
        # Apply temporal attention if enabled
        if self.use_attention:
            # Extract features for main cryptocurrency (first node)
            x_main = x_lstm[:, :, 0, :]  # (batch_size, seq_len, hidden_size)
            
            # Apply temporal attention
            x_main_attended = self.temporal_attention(x_main)
            
            # Use last timestep
            last_timestep = x_main_attended[:, -1, :]  # (batch_size, hidden_size)
        else:
            # Without attention, just use the last timestep of the main cryptocurrency
            last_timestep = x_lstm[:, -1, 0, :]  # (batch_size, hidden_size)
        
        # Concatenate ST-GCN and Graph LSTM features for the main cryptocurrency
        x_combined = torch.cat([x_st[:, -1, 0, :], last_timestep], dim=-1)  # (batch_size, hidden_size*2)
        
        # Aggregate features
        x_features = self.feature_aggregate(x_combined)  # (batch_size, hidden_size)
        
        # Generate predictions
        predictions = self.output_proj(x_features)  # (batch_size, num_forecasts)
        
        return predictions
    
    def get_l2_regularization_loss(self):
        """Calculate L2 regularization loss for the model parameters"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)
        return self.l2_reg * l2_loss


# Utility function to create correlation-based adjacency matrix
def create_corr_adjacency(price_data, threshold=0.5):
    """
    Create adjacency matrix based on price correlations
    
    Args:
        price_data: Historical price data (num_cryptos, num_timesteps)
        threshold: Correlation threshold for creating edges
        
    Returns:
        Adjacency matrix (num_cryptos, num_cryptos)
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(price_data)
    
    # Apply threshold to create binary adjacency matrix
    adj_matrix = np.zeros_like(corr_matrix)
    adj_matrix[np.abs(corr_matrix) > threshold] = 1
    
    # Ensure self-loops
    np.fill_diagonal(adj_matrix, 1)
    
    return adj_matrix