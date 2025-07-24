import torch
import pyvene as pv
from torch.utils.data import DataLoader, DistributedSampler
from pyreft import (
    TaskType,
    get_reft_model,
    ReftConfig, 
    ReftDataCollator,
    ReftSupervisedDataset,
    LoreftIntervention
)
from pyreft.reft_trainer import ReftTrainer,make_dataloader
import torch.nn as nn



class SvdNULLTrainer(ReftTrainer):
    def compute_loss(
        self,
        intervenable: pv.IntervenableModel,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        raw_return = super().compute_loss(
            intervenable, inputs, return_outputs=True, **kwargs
        )

        loss = raw_return[0].loss
        cf_outputs = raw_return[1] if len(raw_return) > 1 else None

        reg_loss = 0.0
        if hasattr(intervenable, "interventions"):
            for intervention in intervenable.interventions.values():
                if isinstance(intervention, SubloreftIntervention):
                    mask_prior,flag = intervention._get_mask_prior(intervention.subspaces)

                    if flag ==True:
                        selected_dims = [0, 2, 4, 6]
                    else:
                        selected_dims = [0, 4, 6]
                    
                    reg_loss += torch.nn.functional.mse_loss(
                        intervention.mask_output.mean(dim=1)[:, selected_dims], 
                        mask_prior.to(intervention.mask_output.device)[:, selected_dims]
                    )

        align_loss = 0.0
        
        if hasattr(intervenable, "interventions"):
            for intervention in intervenable.interventions.values():
                if isinstance(intervention, SubloreftIntervention):
                    align_loss += intervention.compute_reg_loss()
        
        lm_logits = cf_outputs.logits
        labels = inputs["labels"]
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
       
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        total_loss = loss + self.args.reg_lambda * reg_loss +self.args.align_lambda * align_loss 
        
        return (total_loss, cf_outputs) if return_outputs else total_loss

class ReftSvdNULLTrainerForCausalLM(SvdNULLTrainer):
    def get_train_dataloader(self) -> DataLoader:
        return make_dataloader(self.train_dataset, self._train_batch_size, self.data_collator, shuffle=True)

class SubloreftIntervention(LoreftIntervention):
    def __init__(self, layer_no, pretrained_R, mask_prior_config=None, **kwargs):
        super().__init__(**kwargs)
        if pretrained_R is not None:
            self.pretrained_R = torch.nn.Parameter(torch.tensor(pretrained_R[layer_no]), requires_grad=False)
        else:
            self.pretrained_R = None
            
        self.dtype = self.rotate_layer.weight.dtype
        self.low_rank_dimension = kwargs["low_rank_dimension"]

        self.mask_net = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.Dropout(0.1),  
            nn.ReLU(),
            nn.Linear(128, len(mask_prior_config)),
            nn.Sigmoid()
        ).to(self.dtype)

        self.mask_prior_config = mask_prior_config 

    def compute_reg_loss(self):
        if self.pretrained_R is None:
            return 0.0
        cosine_sim = torch.nn.functional.cosine_similarity(
            self.rotate_layer.weight, self.pretrained_R.T, dim=0
        )
        return -cosine_sim.mean()
    

    def _get_mask_prior(self, subspaces):
        batch_size = len(subspaces)
        prior = torch.zeros(batch_size, self.low_rank_dimension)

        if len(self.mask_prior_config) == 4:
            shared_start, shared_end = self.mask_prior_config["shared_rank"]
            sub1_start, sub1_end = self.mask_prior_config["sub_rank1"]
            sub2_start, sub2_end = self.mask_prior_config["sub_rank2"]
            sub3_start, sub3_end = self.mask_prior_config["sub_rank3"]

            for i, sub in enumerate(subspaces):
                has_sub1 = any(sub1_start <= r <= sub1_end for r in sub)
                has_sub2 = any(sub2_start <= r <= sub2_end for r in sub)
                has_sub3 = any(sub3_start <= r <= sub3_end for r in sub)

                prior[i, shared_start:shared_end+1] = 1

                if has_sub1 and not has_sub2 and not has_sub3:
                    prior[i, sub1_start:sub1_end+1] = 0.9
                    prior[i, sub2_start:sub2_end+1] = 0.1
                    prior[i, sub3_start:sub3_end+1] = 0.1
                elif has_sub2 and not has_sub1 and not has_sub3:
                    prior[i, sub1_start:sub1_end+1] = 0.1
                    prior[i, sub2_start:sub2_end+1] = 0.9
                    prior[i, sub3_start:sub3_end+1] = 0.1
                elif has_sub3 and not has_sub1 and not has_sub2:
                    prior[i, sub1_start:sub1_end+1] = 0.1
                    prior[i, sub2_start:sub2_end+1] = 0.1
                    prior[i, sub3_start:sub3_end+1] = 0.9
                else:
                    print(f"Warning: Sample {i} contains features from multiple datasets simultaneously")

        else:
            shared_start, shared_end = self.mask_prior_config["shared_rank"]
            sub1_start, sub1_end = self.mask_prior_config["sub_rank1"]
            sub2_start, sub2_end = self.mask_prior_config["sub_rank2"]

            for i, sub in enumerate(subspaces):

                has_sub1 = any(sub1_start <= r <= sub1_end for r in sub)
                has_sub2 = any(sub2_start <= r <= sub2_end for r in sub)
                
                prior[i, shared_start:shared_end+1] = 1

                if has_sub1 and not has_sub2:
                    prior[i, sub1_start:sub1_end+1] = 0.9
                    prior[i, sub2_start:sub2_end+1] = 0.1
                elif has_sub2 and not has_sub1:
                    prior[i, sub1_start:sub1_end+1] = 0.1
                    prior[i, sub2_start:sub2_end+1] = 0.9
                else:
                    print(f"Warning: Sample {i} contains features from multiple datasets simultaneously")
                    
        return prior,len(self.mask_prior_config) == 4

    def forward(self, base, source=None, subspaces=None):
        
        rotated_base = self.rotate_layer(base)
        diff = self.act_fn(self.learned_source(base)) - rotated_base
        mask = self.mask_net(base.to(self.dtype))  # [batch_size, seq_len, rank_dim]
        
        if len(self.mask_prior_config) == 4:
            shared_start, shared_end = self.mask_prior_config["shared_rank"]
            sub1_start, sub1_end = self.mask_prior_config["sub_rank1"]
            sub2_start, sub2_end = self.mask_prior_config["sub_rank2"]
            sub3_start, sub3_end = self.mask_prior_config["sub_rank3"]


            l0 = shared_end - shared_start + 1
            l1 = sub1_end - sub1_start + 1
            l2 = sub2_end - sub2_start + 1
            l3 = sub3_end - sub3_start + 1
            
            repeats = torch.tensor([l0, l1, l2 , l3], device=mask.device)
        else:
            shared_start, shared_end = self.mask_prior_config["shared_rank"]
            sub1_start, sub1_end = self.mask_prior_config["sub_rank1"]
            sub2_start, sub2_end = self.mask_prior_config["sub_rank2"]

            l0 = shared_end - shared_start + 1
            l1 = sub1_end - sub1_start + 1
            l2 = sub2_end - sub2_start + 1
            
            repeats = torch.tensor([l0, l1, l2], device=mask.device)

        self.mask_output = torch.repeat_interleave(mask, repeats, dim=2)

        self.subspaces = subspaces

        #train
        weighted_diff = diff * self.mask_output # [batch, seq, rank] * [batch, seq, rank, 1]
        #inference
        #weighted_diff = diff

        batched_subspace = []
        batched_weights = []
        
        if subspaces is None:
            LHS = weighted_diff.squeeze(0)
            RHS = self.rotate_layer.weight[..., :].T

            batched_subspace.append(LHS)
            batched_weights.append(RHS)
        else:
            for example_i in range(len(subspaces)):
                #LHS = weighted_diff[example_i, :, subspaces[example_i]]
                #RHS = self.rotate_layer.weight[..., subspaces[example_i]].T

                LHS = weighted_diff[example_i, :, :]
                RHS = self.rotate_layer.weight[..., :].T

                batched_subspace.append(LHS)
                batched_weights.append(RHS)

        batched_subspace = torch.stack(batched_subspace, dim=0)
        batched_weights = torch.stack(batched_weights, dim=0)
        
        output = base + torch.bmm(batched_subspace, batched_weights)
        
        return self.dropout(output.to(base.dtype))

