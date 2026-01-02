"""
Muon Optimizer
"""

import torch
from torch import Tensor 
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G:Tensor,steps:int) -> Tensor:
    """
    Newton-Schulz iteration to compute zeroth power / orthogonalization of G. 
    """
    assert G.ndim >= 2 
    a,b,c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        # .mT returns a matrix transpose
        X = X.mT
    
    # Ensure spectral norm is at most 1 
    # spectral norm is largest possible length of matrix by all unit vectors
    X = X / (X.norm(dim=(-2,-1),keepdim=True) + 1e-7)
    # peform NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b*A + c*A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B@X
        
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X 

class Muon(torch.optim.Optimizer):
    def __init__(self,params,lr=0.02,momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            # .numel() is number of elements
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups,defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf:Tensor = state["momentum_buffer"]
                buf.lerp_(g,1-group["momentum"]) if group["nesterov"] else buf 
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g,alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1) ** 0.5))
                
    class DistMuon(torch.optim.Optimizer):
        """
        Muon: SGD-momentum + (optional) Nesterov, then orthogonalize the 2d update via N-S
        and apply aspect ratio scaled step. 
        """
        def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov:bool = True, ns_steps:int = 5):
            defaults = dict(lr=lr,momentum=momentum,nesterov=nesterov, ns_steps=ns_steps)
            params = list(params)
            # assert all 2d params 
            assert all(p.ndim == 2 for p in params), "Muon expects 2d params"
            rank = dist.get_rank()
            # group all params by their shape
            shapes = sorted({p.shape for p in params}) # sort to ensure consistency
            param_groups = []
            for shape in shapes:
                group_params = [p for p in params if p.shape == shape]
                device, dtype = group_params[0].device, group_params[0].dtype
                assert all(p.device == device for p in group_params)
                assert all(p.dtype == dtype for p in group_params)
                if rank == 0:
                    print(f"Muon: grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
                param_groups.append(dict(params=group_params,zero_buffer=torch.zeros_like(group_params[0])))
            super().__init__(param_groups, defaults)
        
        @torch.no_grad()
        def step(self):
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # Ensure all grads exist
            assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

            # Kick off all the reduce scatter ops to average the gradients across all ranks 
            all_reduce_futures = []
            for group in self.param_groups:
                params = group["params"]
                zero_buffer = group["zero_buffer"]
                # go through params in groups of world_size
                for base_i in range(0, len(params), world_size):
                    # compute owner of each param is rank i % world_size
                    owner_idx = base_i + rank
                    # each rank stacks up its chunk of world_size into a list 
                    rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                    # pad rs_input with zero buf to complete the group
                    rs_input.extend([zero_buffer] * world_size - len(rs_input))
                    # output buffer get strided across the group based on rank
                    rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                    # reduce scatter the gradients within this group of world_size params
                    work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                    all_reduce_futures.append(work)
            
            # now each rank computes update and gathers
            future_idx = 0
            all_gather_features = []
            for group in self.param_groups:
                params = group["params"]
                zero_buffer = group["zero_buffer"]
                # go through params in groups of world_size
                for base_i in range(0, len(params), world_size):
                    # compute owner of each rank i % world_size
                    owner_idx = base_i + rank
                    all_reduce_futures[future_idx].wait()
                    future_idx += 1 
                    # if owner computes muon update, result is in its param
                    if owner_idx < len(params):
                        p = params[owner_idx]
                        g = p.grad
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf: Tensor = state["momentum_buffer"]
                        # .lerp_ is the in-place linear interpolation operation
                        buf.lerp_(g, 1.0- group["momentum"])
                        g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                        g = zeropower_via_newtonschulz5(g,steps=group["ns_steps"])
                        scale = (max(1.0,p.size(-2) / p.size(-1)) ** 0.5)
                        p.add_(g, alpha=-group["lr"] * scale)
                    # replicate updated parameters to all ranks
                    ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                    ag_output = params[base_i:base_i + world_size]
                    ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))])
                    work = dist.all_gather(ag_output, ag_input, async_op = True).get_future()
                    all_gather_features.append(work)
                    
            # wait for all worl to finish
            torch.futures.collect_all(all_gather_features).wait()
                        

                
                
                
                
                
                
                
                