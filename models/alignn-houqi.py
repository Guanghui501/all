"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F
from models.utils import RBFExpansion
from utils import BaseSettings

from transformers import AutoTokenizer
from transformers import AutoModel
from tokenizers.normalizers import BertNormalizer

"""**VoCab Mapping and Normalizer**"""

f = open('vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

"""**Custom Dataset**"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert',model_max_length=512)
text_model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
text_model.to(device)


class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=64,dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CrossModalAttention(nn.Module):
    """Cross-modal attention between graph and text features.

    This module enables bidirectional attention mechanism where:
    - Graph features attend to text features
    - Text features attend to graph features
    Both modalities are enhanced through this interaction.
    """

    def __init__(self, graph_dim=256, text_dim=64, hidden_dim=256, num_heads=4, dropout=0.1):
        """Initialize cross-modal attention.

        Args:
            graph_dim: Dimension of graph features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Graph-to-Text attention (graph queries text)
        self.g2t_query = nn.Linear(graph_dim, hidden_dim)
        self.g2t_key = nn.Linear(text_dim, hidden_dim)
        self.g2t_value = nn.Linear(text_dim, hidden_dim)

        # Text-to-Graph attention (text queries graph)
        self.t2g_query = nn.Linear(text_dim, hidden_dim)
        self.t2g_key = nn.Linear(graph_dim, hidden_dim)
        self.t2g_value = nn.Linear(graph_dim, hidden_dim)

        # Output projections
        self.graph_output = nn.Linear(hidden_dim, graph_dim)
        self.text_output = nn.Linear(hidden_dim, text_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_graph = nn.LayerNorm(graph_dim)
        self.layer_norm_text = nn.LayerNorm(text_dim)

        self.scale = self.head_dim ** -0.5

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    def forward(self, graph_feat, text_feat):
        """Forward pass of cross-modal attention.

        Args:
            graph_feat: Graph features [batch_size, graph_dim]
            text_feat: Text features [batch_size, text_dim]

        Returns:
            enhanced_graph: Graph features enhanced by text [batch_size, graph_dim]
            enhanced_text: Text features enhanced by graph [batch_size, text_dim]
        """
        batch_size = graph_feat.size(0)

        # Add sequence dimension if needed
        if graph_feat.dim() == 2:
            graph_feat_seq = graph_feat.unsqueeze(1)  # [batch, 1, graph_dim]
        else:
            graph_feat_seq = graph_feat

        if text_feat.dim() == 2:
            text_feat_seq = text_feat.unsqueeze(1)  # [batch, 1, text_dim]
        else:
            text_feat_seq = text_feat

        # Graph-to-Text Attention: Graph attends to Text
        Q_g2t = self.g2t_query(graph_feat_seq)  # [batch, 1, hidden]
        K_g2t = self.g2t_key(text_feat_seq)     # [batch, 1, hidden]
        V_g2t = self.g2t_value(text_feat_seq)   # [batch, 1, hidden]

        # Multi-head attention
        Q_g2t = self.split_heads(Q_g2t, batch_size)
        K_g2t = self.split_heads(K_g2t, batch_size)
        V_g2t = self.split_heads(V_g2t, batch_size)

        attn_g2t = torch.matmul(Q_g2t, K_g2t.transpose(-2, -1)) * self.scale
        attn_g2t = F.softmax(attn_g2t, dim=-1)
        attn_g2t = self.dropout(attn_g2t)

        context_g2t = torch.matmul(attn_g2t, V_g2t)
        context_g2t = context_g2t.permute(0, 2, 1, 3).contiguous()
        context_g2t = context_g2t.view(batch_size, 1, self.hidden_dim)
        context_g2t = self.graph_output(context_g2t).squeeze(1)  # [batch, graph_dim]

        # Text-to-Graph Attention: Text attends to Graph
        Q_t2g = self.t2g_query(text_feat_seq)   # [batch, 1, hidden]
        K_t2g = self.t2g_key(graph_feat_seq)    # [batch, 1, hidden]
        V_t2g = self.t2g_value(graph_feat_seq)  # [batch, 1, hidden]

        # Multi-head attention
        Q_t2g = self.split_heads(Q_t2g, batch_size)
        K_t2g = self.split_heads(K_t2g, batch_size)
        V_t2g = self.split_heads(V_t2g, batch_size)

        attn_t2g = torch.matmul(Q_t2g, K_t2g.transpose(-2, -1)) * self.scale
        attn_t2g = F.softmax(attn_t2g, dim=-1)
        attn_t2g = self.dropout(attn_t2g)

        context_t2g = torch.matmul(attn_t2g, V_t2g)
        context_t2g = context_t2g.permute(0, 2, 1, 3).contiguous()
        context_t2g = context_t2g.view(batch_size, 1, self.hidden_dim)
        context_t2g = self.text_output(context_t2g).squeeze(1)  # [batch, text_dim]

        # Residual connection and layer normalization
        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)
        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

        return enhanced_graph, enhanced_text


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # Cross-modal attention settings
    use_cross_modal_attention: bool = True
    cross_modal_hidden_dim: int = 256
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self,g: dgl.DGLGraph,lg: dgl.DGLGraph,x: torch.Tensor,y: torch.Tensor,z: torch.Tensor,):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0,vmax=8.0,bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1,vmax=1.0,bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features,config.hidden_features)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(config.hidden_features, config.hidden_features)
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        self.graph_projection = ProjectionHead(embedding_dim=256)
        self.text_projection = ProjectionHead(embedding_dim=768)

        # Cross-modal attention module
        self.use_cross_modal_attention = config.use_cross_modal_attention
        if self.use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                graph_dim=64,  # After graph_projection
                text_dim=64,   # After text_projection
                hidden_dim=config.cross_modal_hidden_dim,
                num_heads=config.cross_modal_num_heads,
                dropout=config.cross_modal_dropout
            )
            # Fusion layer after cross-modal attention
            self.fc1 = nn.Linear(64, 64)  # Single modality after fusion
            self.fc = nn.Linear(64, config.output_features)
        else:
            # Original simple concatenation
            self.fc1 = nn.Linear(128, 64)
            self.fc = nn.Linear(64, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            # g, lg = g
            g, lg, text = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()


        # CLS Embedding
        norm_sents = [normalize(s) for s in text]
        encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)
        if torch.cuda.is_available():
            encodings.to(device)
        with torch.no_grad():
            # last_hidden_state = self.text_model(**encodings)[0]
            last_hidden_state = text_model(**encodings)[0]

        cls_emb = last_hidden_state[:, 0, :]
        text_emb = self.text_projection(cls_emb)
        text_emb = torch.squeeze(text_emb)


        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        graph_emb = self.readout(g, x)
        h = self.graph_projection(graph_emb)

        # Multi-Modal Representation Fusion
        if self.use_cross_modal_attention:
            # Cross-modal attention fusion
            enhanced_graph, enhanced_text = self.cross_modal_attention(h, text_emb)
            # Average the enhanced features
            h = (enhanced_graph + enhanced_text) / 2.0
            h = F.relu(self.fc1(h))
            out = self.fc(h)
        else:
            # Original simple concatenation
            h = torch.cat((h, text_emb), 1)
            h = F.relu(self.fc1(h))
            out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
