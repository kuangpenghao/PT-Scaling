# Codes adapted from https://github.com/lacoco-lab/sensitivity-hardness/blob/main/src/measurements.py

from typing import Optional
from tqdm import tqdm, trange
import torch

from transformers import (
    TrainerCallback, 
    TrainerControl, 
    TrainerState, 
    TrainingArguments
)

from transformers.integrations import WandbCallback as OriginWandbCallback

class WandbCallback(OriginWandbCallback):
    """Add sharpness measurement to the log."""
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        model = kwargs.get("model", None)
        eval_dataloader = kwargs.get("eval_dataloader", None)
        if model is None or eval_dataloader is None:
            return
        
        sharpness = 0
        for step, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            sharpness += self.hessian_norm_proxy(inputs, model)
        
        print(f"Sharpness: {sharpness}")
        self._wandb.log({"eval/sharpness": sharpness, "train/global_step": state.global_step, "train/epoch": state.epoch})

    @torch.inference_mode()
    def hessian_norm_proxy(
        self,
        inputs: dict,
        model: torch.nn.Module,
        epsilon: Optional[float] = 1e-3,
        n_repeats: int = 10,
        rng = None,
        rho: Optional[float] = None
    ) -> torch.Tensor:
        """This function takes a batch and measures sharpness of loss landscape on this batch

            epsilon (float, optional): standard deviation of noise. Defaults to 1e-3.
            n_repeats (int, optional): number of noise additions. Defaults to 10.
            rng (_type_, optional): torch random number generator. Defaults to None.
            rho (float, optional): epsilon * sqrt(number of parameters). Is needed for consistently adding noise in experiments with varying number of parameteres. Defaults to None.
        """

        if epsilon is not None and rho is not None:
            raise ValueError("You can't specify both epsilon and rho")
        
        if rho is not None:
            n_params = sum([p.numel() for n, p in model.named_parameters() if "positional" not in n])
            epsilon = rho / (n_params ** 0.5)

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        avg_diff = - loss.item()

        for _ in range(n_repeats):

            noised_params = {
                name: param + torch.randn(param.size(), generator=rng, device=param.device) * epsilon
                for name, param in model.state_dict().items()
                if "positional_encoding" not in name
            }

            outputs = torch.func.functional_call(model, noised_params, (), kwargs=inputs, tie_weights=False)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            avg_diff += loss.item() / n_repeats

        return avg_diff


