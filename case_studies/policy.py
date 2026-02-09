import torch 
import torch.nn as nn
import torch.nn.functional as F 

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax



class HeteroGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=8):
        super(HeteroGATConv, self).__init__(node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x_src, x_dst, edge_index):
        # Project features
        x_src = self.lin(x_src).view(-1, self.heads, self.out_channels)
        x_dst = self.lin(x_dst).view(-1, self.heads, self.out_channels)

        return self.propagate(edge_index, x=(x_src, x_dst))

    def message(self, x_i, x_j, edge_index_i, size_i):
        # x_i: destination, x_j: source
        # Standard GAT attention: e = a^T [Wh_i || Wh_j]
        x = torch.cat([x_i, x_j], dim=-1)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # Average the heads
        return aggr_out.mean(dim=1)
    




class HGNN_Extractor(nn.Module):
    def __init__(self, op_dim=4, mc_dim=2, embed_dim=64):
        super(HGNN_Extractor, self).__init__()

        # 1. Input Encoders (Projection Layers)
        self.op_encoder = nn.Linear(op_dim, embed_dim)
        self.mc_encoder = nn.Linear(mc_dim, embed_dim)

        # 2. Stage 1: O -> M (Machines learn which Ops are waiting for them)
        self.conv_o_to_m = HeteroGATConv(embed_dim, embed_dim)

        # 3. Stage 2: M -> O (Ops learn when Machines will be free)
        self.conv_m_to_o = HeteroGATConv(embed_dim, embed_dim)

        # ELU is specifically recommended in the paper
        self.activation = nn.ELU()

    def forward(self, op_nodes, mc_nodes, edge_index_om):
        """
        op_nodes: [17, 3]
        mc_nodes: [10, 2]
        edge_index_om: [2, E] (Source is Op, Dest is Mc)
        """
        # Initial Embeddings
        h_op = self.activation(self.op_encoder(op_nodes))
        h_mc = self.activation(self.mc_encoder(mc_nodes))

        # --- STAGE 1: Update Machine Embeddings (O -> M) ---
        # The edge_index_om is [Op, Mc]. Since we update Machines,
        # Source=Op, Dest=Mc.
        # We flip the edge index for the update if needed, but here Mc is Dest.
        h_mc_new = self.conv_o_to_m(h_op, h_mc, edge_index_om)
        h_mc = self.activation(h_mc + h_mc_new) # Residual connection

        # --- STAGE 2: Update Operation Embeddings (M -> O) ---
        # We need the edges reversed: [Mc, Op]
        edge_index_mo = torch.stack([edge_index_om[1], edge_index_om[0]])

        h_op_new = self.conv_m_to_o(h_mc, h_op, edge_index_mo)
        h_op = self.activation(h_op + h_op_new) # Residual connection

        return h_op, h_mc
    




class ActorCritic(nn.Module):
    def __init__(self, op_dim=4, mc_dim=2, embed_dim=64):
        super(ActorCritic, self).__init__()
        # HGNN Feature Extractor (assumes this class is defined elsewhere)
        self.feature_extractor = HGNN_Extractor(op_dim, mc_dim, embed_dim)
        
        # Actor: Score for (Op, Mc) pairs
        self.pair_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Actor: Score for "Wait" action (Action 0)
        self.wait_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Critic: State Value
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, op_nodes, mc_nodes, edge_index_om, info):
        """
        Args:
            op_nodes: [17, 4] Tensor
            mc_nodes: [10, 2] Tensor
            edge_index_om: [2, E] Tensor
            info: Metadata dictionary containing 'job_to_op_map' and 'compatible_pairs'
        """
        # 1. Extract context-aware embeddings
        h_op, h_mc = self.feature_extractor(op_nodes, mc_nodes, edge_index_om)
        
        # 2. Global State Representation (Aggregation)
        # Using mean pooling across all nodes to represent the factory state
        global_state = torch.mean(h_op, dim=0) + torch.mean(h_mc, dim=0)
        
        # 3. Critic: Predict State Value V(s)
        v_s = self.critic(global_state)
        
        # 4. Actor: Calculate 'Wait' (Action 0) Logit
        wait_logit = self.wait_scorer(global_state).view(1) # Shape: [1]
        
        # 5. Actor: Calculate Pair Logits
        job_to_op = info['job_to_op_map']
        pairs = info['compatible_pairs']
        
        pair_logits = []
        for j_idx, m_name in pairs:
            # Map job index to current operation node index using info dict
            op_node_idx = job_to_op[j_idx]
            # Map 'cppuX' string to integer index
            mc_node_idx = int(m_name.replace('cppu', ''))
            
            # Combine Op and Machine embeddings
            pair_embed = torch.cat([h_op[op_node_idx], h_mc[mc_node_idx]], dim=-1)
            pair_logits.append(self.pair_scorer(pair_embed))
        
        # 6. Final Logit Assembly
        if len(pair_logits) > 0:
            pair_logits_tensor = torch.cat(pair_logits)
            # Action 0 is Wait, Indices 1+ are the compatible pairs
            all_logits = torch.cat([wait_logit, pair_logits_tensor])
        else:
            all_logits = wait_logit
            
        return v_s, all_logits
    



