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
