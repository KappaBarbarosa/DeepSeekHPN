import os

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from utils.th_utils import get_parameters_num
from collections import defaultdict

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape = self._get_input_shape(scheme)
        self._build_agents(self.input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)
        self.stats_by_layer = defaultdict(list)
        self.hidden_states = None
        if getattr(self.args, "transfer_checkpoint_path", None):
            if self.args.transfer:
                self.load_models(self.args.transfer_checkpoint_path)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        if t_ep == 0:
            self.set_evaluation_mode()
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        stats = None
        agent_outs, self.hidden_states,stats = self.agent(agent_inputs, self.hidden_states)
        if stats:
            for lid, stats in stats.items():
             self.stats_by_layer[lid].append(stats)   # 直接塞進 dict(list)
        # print("agents_out:", agent_outs[0])
        # print("hidden_states:", self.hidden_states[0])
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1),stats

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def pop_router_stats(self):
        stats = self.stats_by_layer
        self.stats_by_layer = defaultdict(list)
        return stats
    
    def set_train_mode(self):
        self.agent.train()

    def set_evaluation_mode(self):
        self.agent.eval()

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def cpu(self):
        self.agent.cpu()

    def get_device(self):
        return next(self.parameters()).device

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        loaded_state = th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        assert isinstance(loaded_state, dict), "Loaded state must be a dictionary."
        model_state = self.agent.state_dict()
        
        transfer = getattr(self.args, "transfer", False)
        n_new_tokens = getattr(self.args, "n_new_tokens", 0)
        assert isinstance(n_new_tokens, int) and n_new_tokens >= 0, "n_new_tokens must be a non-negative integer."
        freeze_old = getattr(self.args, "freeze_old", False)
        
        if transfer and n_new_tokens > 0:
            print(f"Transferring with {n_new_tokens} new tokens per Pattention layer...")
        
            for name, param in loaded_state.items():
                print (name)
            for name, param in loaded_state.items():
                if "tokenformer" not in name:
                    print(f"Skipped non-tokenformer param: {name}")
                    continue
                if name in model_state and isinstance(param, th.Tensor) and param.shape != model_state[name].shape:
                    old_shape, new_shape = param.shape, model_state[name].shape
                    assert len(old_shape) == len(new_shape), f"Shape mismatch: {name} has {old_shape} vs {new_shape}"
                    if len(old_shape) == 2 and old_shape[0] < new_shape[0]:
                        print(f"  Transferring: {name} from {old_shape} into {new_shape}")
                        model_state[name][:old_shape[0], :] = param
                    else:
                        print(f"  Skipped mismatched param: {name} shape {old_shape} vs {new_shape}")
                        assert model_state[name].shape == param.shape or transfer, f"Unexpected shape mismatch in {name}: loaded {param.shape}, expected {model_state[name].shape}"
                        model_state[name][:] = param
                elif name in model_state:
                    model_state[name][:] = param
                else:
                    print(f"  Ignored extra param in loaded model: {name}")
        
            self.agent.load_state_dict(model_state)
            print("Model loaded successfully.")
        else:
            missing_keys, unexpected_keys = self.agent.load_state_dict(loaded_state, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state_dict: {unexpected_keys}")
            print("Model loaded successfully.")
        
        if transfer and freeze_old:
            print("Freezing transferred weights...")
            for name, param in self.agent.named_parameters():
                if name in loaded_state and "tokenformer" in name and any(dim == 2 and loaded_state[name].shape[0] < param.shape[0] for dim in [len(param.shape)]):
                    param.requires_grad = True
                    param.data[:loaded_state[name].shape[0],:] = loaded_state[name]
                    param.requires_grad = False
                    print(f"Froze first {loaded_state[name].shape[0]} tokens of: {name}")

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        print("&&&&&&&&&&&&&&&&&&&&&&", self.args.agent, get_parameters_num(self.parameters()))
        # for p in list(self.parameters()):
        #     print(p.shape)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
