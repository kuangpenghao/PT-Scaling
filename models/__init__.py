from .configuration_pt import PtConfig
from .modeling_pt import PtModel, PtForMaskedLM

# register to transformer callbacks
from transformers.integrations import INTEGRATION_TO_CALLBACK, rewrite_logs
from .wandb_callback import WandbCallback
INTEGRATION_TO_CALLBACK["wandb"] = WandbCallback

from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
AutoConfig.register("pt", PtConfig)
AutoModel.register(PtConfig, PtModel)
AutoModelForMaskedLM.register(PtConfig, PtForMaskedLM)

from .modeling_llama import LlamaConfig, LlamaForMaskedLM
AutoModelForMaskedLM.register(LlamaConfig, LlamaForMaskedLM)

from .configuration_ut import UtConfig
from .modeling_ut import UtModel, UtForMaskedLM
AutoConfig.register("ut", UtConfig)
AutoModel.register(UtConfig, UtModel)
AutoModelForMaskedLM.register(UtConfig, UtForMaskedLM)
