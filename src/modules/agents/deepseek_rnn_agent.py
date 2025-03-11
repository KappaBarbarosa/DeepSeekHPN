import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from modules.layer.deepseek_decoder import Transformer

def kaiming_uniform_(tensor_w, tensor_b, mode='fan_in', gain=12 ** (-0.5)):
    fan = nn.init._calculate_correct_fan(tensor_w.data, mode)
    std = gain / math.sqrt(fan)
    bound_w = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    bound_b = 1 / math.sqrt(fan)
    with th.no_grad():
        tensor_w.data.uniform_(-bound_w, bound_w)
        if tensor_b is not None:
            tensor_b.data.uniform_(-bound_b, bound_b)


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)


class DeepSeek_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DeepSeek_RNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.cache = th.zeros(1, 1, self.rnn_hidden_dim)
        

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]

        if self.args.obs_agent_id:
            # embedding table for agent_id
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        if self.args.obs_last_action:
            # embedding table for action id
            self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_enemy = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, ((self.enemy_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
        )  # output shape: (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)

        if self.args.map_type == "MMM":
            assert self.n_enemies >= self.n_agents, "For MMM map, for the reason that the 'attack' and 'rescue' use the same ids in SMAC, n_enemies must >= n_agents"
            self.hyper_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, ((self.ally_feats_dim + 1) * self.rnn_hidden_dim + 1) * self.n_heads)
            )  # output shape: ally_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1, for 'rescue actions'
            self.unify_output_heads_rescue = Merger(self.n_heads, 1)
        else:
            self.hyper_ally = nn.Sequential(
                nn.Linear(self.ally_feats_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, self.ally_feats_dim * self.rnn_hidden_dim * self.n_heads)
            )  # output shape: ally_feats_dim * rnn_hidden_dim

        self.unify_input_heads = Merger(self.n_heads, self.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # (no_op, stop, up, down, right, left)
        self.unify_output_heads = Merger(self.n_heads, 1)


        self.initial_embedding = None
        self.decoder = Transformer(args)    
        # Reset parameters for hypernets
        # self._reset_hypernet_parameters(init_type="xavier")
        # self._reset_hypernet_parameters(init_type="kaiming")

    def _reset_hypernet_parameters(self, init_type='kaiming'):
        gain = 2 ** (-0.5)
        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        for m in self.hyper_enemy.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)
        for m in self.hyper_ally.modules():
            if isinstance(m, nn.Linear):
                if init_type == "kaiming":
                    kaiming_uniform_(m.weight, m.bias, gain=gain)
                else:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.)

    def _reset_initial_embedding(self,bs):
        # print("reset initial embedding", bs, self.n_agents)
        self.initial_embedding = th.zeros(bs*self.n_agents, self.rnn_hidden_dim).to(self.fc1_own.weight.device)


    def init_hidden(self):
        # make hidden states on same device as model
        self._reset_initial_embedding(1)
        self.cache = None
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()
    
    def start_one_training(self):
        self.decoder.start_one_training()
    def end_one_training(self,t_env):
        self.decoder.end_one_training(t_env)

    def forward(self, inputs, hidden_state):
        # [bs, n_agents, mv_fea_dim], [bs * n_agents * n_enemies, enemy_fea_dim], [bs * n_agents * n_allies, ally_fea_dim], [bs, n_agents, own_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs

        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

        # (2) ID embeddings
        if self.args.obs_agent_id:
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim]
            embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim]
                embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        # (3) Enemy feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1)
        hyper_enemy_out = self.hyper_enemy(enemy_feats_t)
        fc1_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
            -1, self.enemy_feats_dim, self.rnn_hidden_dim * self.n_heads
        )  # [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim]
        # [bs * n_agents * n_enemies, 1, enemy_fea_dim] * [bs * n_agents * n_enemies, enemy_fea_dim, rnn_hidden_dim] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), fc1_w_enemy).view(
            bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

        # (4) Ally features
        hyper_ally_out = self.hyper_ally(ally_feats_t)
        if self.args.map_type == "MMM":
            # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
            fc1_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
                -1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads
            )
        else:
            # [bs * n_agents * n_allies, ally_fea_dim, rnn_hidden_dim * head]
            fc1_w_ally = hyper_ally_out.view(-1, self.ally_feats_dim, self.rnn_hidden_dim * self.n_heads)
        # [bs * n_agents * n_allies, 1, ally_fea_dim] * [bs * n_agents * n_allies, ally_fea_dim, n_heads* rnn_hidden_dim] = [bs * n_agents * n_allies, 1, n_heads*rnn_hidden_dim]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), fc1_w_ally).view(
            bs * self.n_agents, self.n_allies, self.n_heads, self.rnn_hidden_dim
        )  # [bs * n_agents, n_allies, head, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

        # Final embedding
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )  # [bs * n_agents, head, rnn_hidden_dim]

        obs_embedding = F.relu(embedding, inplace=True)

        if self.cache is not None:
            # 取出 cache 中每個 sample 最後一個時間步的 embedding
            # last_embedding = self.cache[:, -1, :]  # shape: [bs, feature]
            # # 計算每個 sample 當前步與上一個步的 Euclidean 距離
            # euclidean_distance = th.norm(obs_embedding - last_embedding, dim=-1)
            
            # print("Last embedding (first sample):", last_embedding[0, :10])
            # print("Current obs embedding (first sample):", obs_embedding[0, :10])
            # print("Euclidean Distance of obs embedding(max/mean/min):", euclidean_distance.max(), euclidean_distance.mean(), euclidean_distance.min())
            
            # 將當前步的 embedding 加入 cache，unsqueeze 在時間步維度 (dim=1)
            self.cache = th.cat([self.cache, obs_embedding.unsqueeze(1).detach()], dim=1)
        else:
            # print("Cache is empty, initializing with current step embeddings.")
            self.cache = obs_embedding.unsqueeze(1).detach()

        # print()
        
        hh = self.decoder(self.cache)  ## [bs * n_agents, rnn_hidden_dim]
        # print(hh[0])
        # cache: concat
        # moe: share for common knowledge, other experts for finetuning, transfer learning is important
        # hpn: 
        # grpo


        # Q-values of normal actions
        q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

        # Q-values of attack actions: [bs * n_agents * n_enemies, rnn_hidden_dim * n_heads]
        fc2_w_attack = hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
            bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
        ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_enemies, n_heads]
            bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
        )  # [bs * n_agents, rnn_hidden_dim, n_enemies * heads]
        fc2_b_attack = hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_enemies * self.n_heads)

        # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
        q_attacks = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(
            bs * self.n_agents * self.n_enemies, self.n_heads, 1
        )  # [bs * n_agents, n_enemies*head] -> [bs * n_agents * n_enemies, head, 1]

        # Merge multiple heads into one.
        q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
            bs, self.n_agents, self.n_enemies
        )  # [bs, n_agents, n_enemies]


        # Concat 2 types of Q-values
        q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1)  # [bs, n_agents, 6 + n_enemies]
