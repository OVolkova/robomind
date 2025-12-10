import inspect
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfigTiny:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in batch
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.TO_SCALE = True

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # # mask
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                      .view(1, 1, config.block_size, config.block_size))

        # dropout
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.c_dropout = nn.Dropout(config.resid_pdrop)
        # self.resid_pdrop = config.resid_pdrop

    def forward(self, x):
        # input shape is [B, T, C]
        # B - batch size, T - sequence length, C - channels
        B, T, C = x.size()

        # calculate query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # dim: [B, T, C] -> [B, T, n_head, C // n_head] -> [B, n_head, T, C // n_head]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # # attention: [B, n_head, T, C // n_head] @ [B, n_head, C // n_head, T] -> [B, n_head, T, T]
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        #
        # y = att @ v  # [B, n_head, T, T] @ [B, n_head, T, C // n_head] -> [B, n_head, T, C // n_head]
        # no dropout for Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.c_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.TO_SCALE = True

        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.035
        if isinstance(module, nn.Linear):
            if hasattr(module, "TO_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, x):
        # input shape is [B, T]
        # B - batch size, T - sequence length
        B, T = x.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} is longer than block size {self.config.block_size}"
        )

        # position embeddings
        pos = torch.arange(T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        # token embeddings
        tok_emd = self.transformer.wte(x)
        # sum both embeddings
        x = tok_emd + pos_emb
        x = self.transformer.drop(x)

        # transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # layer normalization and linear layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        # output shape is [B, T, vocab_size]
        return logits

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pre-trained model from Hugging Face's transformers library."""
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel

        print(f"Loading pre-trained model {model_type}...")

        # n_layer, n_head, n_embd are determined by the model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M parameters
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M parameters
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M parameters
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M parameters
        }[model_type]
        config_args["vocab_size"] = 50257  # GPT2 vocab size
        config_args["block_size"] = 1024  # GPT2 block size
        config = GPTConfig(**config_args)
        print(f"config_args: {config}")
        model = GPT(config)
        # print(model)
        sd = model.state_dict()
        # print([k for k in sd.keys() if k.endswith('.attn.bias')])
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        # Load the pre-trained model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # print(model_hf)
        sd_hf = model_hf.state_dict()

        # Copy the weights
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not k.endswith(".attn.bias") and not k.endswith(".attn.masked_bias")
        ]
        # transposed = []
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert sorted(sd_keys) == sorted(sd_keys_hf), "mismatched keys"
        assert len(sd_keys) == len(sd_keys_hf), (
            f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                assert sd_hf[k].shape == sd[k].shape, (
                    f"mismatched shape for {k}: {sd_hf[k].shape} != {sd[k].shape}"
                )
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                assert torch.allclose(sd_hf[k].t(), sd[k], atol=1e-5), (
                    f"transposed not close {k}"
                )

            else:
                assert sd_hf[k].shape == sd[k].shape, (
                    f"mismatched shape for {k}: {sd_hf[k].shape} != {sd[k].shape}"
                )
                assert torch.allclose(sd_hf[k], sd[k], atol=1e-5), (
                    f"transposed not close {k}"
                )

        print(f"Loaded.")
        return model

    def get_loss(self, logits, target, reduction="mean"):
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), reduction=reduction
        )
        return loss

    def generate(self, x, max_length, k=50):
        self.eval()
        while x.size(1) < max_length:
            with torch.no_grad():
                logits = self(x)
                logits = logits[:, -1, :]  # take the logits of the last position
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        return x

    def generate_till_eot(self, x, eot_token, wait_time=5, k=50):
        self.eval()
        start_time = time.time()
        total_generated = 0
        while time.time() - start_time <= wait_time:
            with torch.no_grad():
                logits = self(x)
                logits = logits[:, -1, :]  # take the logits of the last position
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, -1, ix)
                if xcol.item() == eot_token and total_generated > 0:
                    break
                x = torch.cat((x, xcol), dim=1)
                total_generated += 1

        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device):
        param_dict = {k: v for k, v in self.named_parameters() if v.requires_grad}
        decay_parameters = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_parameters = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": nodecay_parameters, "weight_decay": 0.0},
        ]
        # num_decay_params = sum(p.numel() for p in decay_parameters)
        # num_nodecay_params = sum(p.numel() for p in nodecay_parameters)
        # print(f"num_decay_tensors: {len(decay_parameters)},num_decay_params: {num_decay_params}")
        # print(f"num_nodecay_tensors: {len(nodecay_parameters)},num_nodecay_params: {num_nodecay_params}")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (device == "cuda" or isinstance(device, int))
        # print(f"use_fused: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args
        )
        return optimizer
