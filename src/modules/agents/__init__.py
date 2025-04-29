REGISTRY = {}

from .hpn_rnn_agent import HPN_RNNAgent
from .hpns_rnn_agent import HPNS_RNNAgent
from .asn_rnn_agent import AsnRNNAgent
from .deepset_hyper_rnn_agent import DeepSetHyperRNNAgent
from .deepset_rnn_agent import DeepSetRNNAgent
from .gnn_rnn_agent import GnnRNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_agent import RNNAgent
from .updet_agent import UPDeT
from .updeept_agent import UPDeepT
from .deepseek_rnn_agent import DeepSeek_RNNAgent
from .pattention_rnn_agent import Pattention_RNNAgent
from .tokenformer_agent import TokenformerAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["hpn_rnn"] = HPN_RNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
REGISTRY["deepset_rnn"] = DeepSetRNNAgent
REGISTRY["deepset_hyper_rnn"] = DeepSetHyperRNNAgent
REGISTRY["updet_agent"] = UPDeT
REGISTRY["updeept_agent"] = UPDeepT
REGISTRY["asn_rnn"] = AsnRNNAgent
REGISTRY["gnn_rnn"] = GnnRNNAgent
REGISTRY["deepseek_rnn"] = DeepSeek_RNNAgent
REGISTRY["pattention_rnn"] = Pattention_RNNAgent
REGISTRY["tokenformer"] = TokenformerAgent