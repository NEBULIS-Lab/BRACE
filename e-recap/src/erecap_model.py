from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ERECAPModel(nn.Module):
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            # Reuse an existing model instance
            self.model = model.to(self.device)
            if tokenizer is not None:
                self.tokenizer = tokenizer
            else:
                # Fallback: try to infer from model_name or config
                if model_name is not None:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                else:
                    # Use config name if available
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        else:
            assert model_name is not None, "Either model_name or model must be provided"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=None,
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        for param in self.model.parameters():
            param.requires_grad = False

        self.pruning_modules = nn.ModuleDict()
        self.gumbel_tau = 1.0

    def attach_pruning_module(self, layer_idx: int, module: nn.Module) -> None:
        self.pruning_modules[str(layer_idx)] = module.to(self.device)

    def _extract_keep_scores(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            return logits[..., 0]
        return logits

    def apply_pruning(
        self,
        hidden_states: torch.Tensor,
        scores: torch.Tensor,
        keep_ratio: float,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Validate input and saliency scores
        if seq_len == 0:
            raise ValueError(f"apply_pruning: input sequence length is 0!")
        scores_flat = scores.flatten()
        scores_abs_sum = scores_flat.abs().sum().item()
        scores_min = scores_flat.min().item()
        scores_max = scores_flat.max().item()
        
        if scores_abs_sum == 0 or (scores_max - scores_min) < 1e-8:
            # All scores are zero or identical - add small random noise
            print(f"[WARNING] Layer {layer_idx}: All saliency scores are zero/identical, adding noise")
            scores = scores + 1e-6 * torch.randn_like(scores)
        
        # Compute keep_k with safeguards (each layer prunes based on CURRENT sequence length)
        keep_k = int(seq_len * keep_ratio)
        # Guarantee minimum: at least 1 token
        keep_k = max(1, keep_k)
        # Ensure we don't keep all tokens (must prune at least 1)
        if keep_k >= seq_len:
            keep_k = max(1, seq_len - 1)
        # Final safeguard: ensure keep_k < seq_len
        keep_k = min(keep_k, seq_len - 1) if seq_len > 1 else 1

        topk = scores.topk(keep_k, dim=-1, largest=True)
        topk_indices = topk.indices.sort(dim=-1).values

        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1))
        pruned_hidden_states = torch.gather(hidden_states, dim=1, index=gather_index)
        
        # Final validation - ensure output is not empty
        if pruned_hidden_states.size(1) == 0:
            raise ValueError(f"apply_pruning: Output sequence length is 0! seq_len={seq_len}, keep_k={keep_k}")

        pruned_attention_mask = None
        if attention_mask is not None:
            pruned_attention_mask = torch.gather(attention_mask, dim=1, index=topk_indices)
            # Ensure attention mask matches pruned length
            assert pruned_attention_mask.size(1) == pruned_hidden_states.size(1), \
                f"Attention mask length {pruned_attention_mask.size(1)} != pruned hidden states length {pruned_hidden_states.size(1)}"

        return pruned_hidden_states, pruned_attention_mask, topk_indices

    def prune_past_key_values(
        self,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        kept_indices: torch.Tensor,
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        if past_key_values is None:
            return None
        pruned_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key, value in past_key_values:
            pruned_key = key.index_select(dim=2, index=kept_indices.squeeze(0))
            pruned_value = value.index_select(dim=2, index=kept_indices.squeeze(0))
            pruned_cache.append((pruned_key, pruned_value))
        return pruned_cache

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare attention mask for decoder-only models.
        Compatible with both LLaMA (has _prepare_decoder_attention_mask) and Qwen2 (doesn't).
        """
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, dtype=torch.long, device=hidden_states.device)
        
        # Try to use model's _prepare_decoder_attention_mask if available (LLaMA)
        # Otherwise, create a simple causal mask manually (Qwen2)
        if hasattr(self.model, '_prepare_decoder_attention_mask'):
            return self.model._prepare_decoder_attention_mask(
                attention_mask,
                input_shape,
                hidden_states.dtype,
                hidden_states.device,
                past_key_values_length=0,
            )
        else:
            # Qwen2: create causal attention mask manually
            # Create a 4D causal mask: [batch_size, 1, seq_len, seq_len]
            batch_size, seq_len = input_shape
            dtype = hidden_states.dtype
            device = hidden_states.device
            
            # Create causal mask: upper triangle is -inf, lower triangle (including diagonal) is 0
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=dtype, device=device) * float("-inf"),
                diagonal=1
            )
            # Expand to [batch_size, 1, seq_len, seq_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
            
            # Apply attention_mask to mask out padding tokens
            if attention_mask.dim() == 2:
                # attention_mask: [batch_size, seq_len]
                # Expand to [batch_size, 1, 1, seq_len] for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
                # Where attention_mask is 0, set causal_mask to -inf
                causal_mask = causal_mask.masked_fill(attention_mask_expanded == 0, float("-inf"))
            
            return causal_mask

    def forward_prefill_with_pruning(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        keep_ratio: float,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Training/inference prefill with pruning (legacy method).
        
        NOTE: Currently not used in end2end inference (fallback mode uses
        prefill_with_pruning from inference_erecap instead). Kept for potential
        future use or training scenarios.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=self.device)

        hidden_states = self.model.model.embed_tokens(input_ids)
        position_ids = torch.arange(hidden_states.size(1), device=self.device).unsqueeze(0)
        extended_attention = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)

        kept_indices: Dict[int, torch.Tensor] = {}
        past_key_values = None

        for idx, block in enumerate(self.model.model.layers):
            # ModuleDict doesn't support .get() in all PyTorch versions, use try-except instead
            try:
                pruning_module = self.pruning_modules[str(idx)]
            except KeyError:
                pruning_module = None

            if pruning_module is not None:
                pruning_module = pruning_module.to(self.device)
                if self.training:
                    logits = pruning_module(hidden_states)
                    gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)
                    keep_prob = gumbel[..., 0].unsqueeze(-1)
                    hidden_states = hidden_states * keep_prob
                else:
                    scores = pruning_module(hidden_states)
                    scores = self._extract_keep_scores(scores)
                    # Each layer prunes based on CURRENT sequence length (not cumulative)
                    hidden_states, attention_mask, kept = self.apply_pruning(
                        hidden_states, scores, keep_ratio, attention_mask, layer_idx=idx
                    )
                    kept_indices[idx] = kept.detach()
                    position_ids = torch.arange(hidden_states.size(1), device=self.device).unsqueeze(0)
                    extended_attention = self._prepare_attention_mask(attention_mask, hidden_states.shape[:2], hidden_states)
                    # Ensure KV cache matches pruned lengths
                    past_key_values = self.prune_past_key_values(past_key_values, kept)

            block_outputs = block(
                hidden_states,
                attention_mask=extended_attention,
                position_ids=position_ids,
                past_key_value=past_key_values[idx] if past_key_values is not None else None,
                use_cache=past_key_values is not None,
            )

            hidden_states = block_outputs[0]
            if self.model.config.use_cache:
                if past_key_values is None:
                    past_key_values = [None] * len(self.model.model.layers)
                past_key_values[idx] = block_outputs[1]

        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=self.device)

        return logits, kept_indices, attention_mask, hidden_states

    def prefill_with_pruning_infer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        keep_ratio: float,
    ) -> Tuple[torch.Tensor, Dict, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Inference-only E-RECAP prefill with KV cache management.
        
        NOTE: Currently not used in end2end inference (fallback mode uses
        prefill_with_pruning from inference_erecap instead). This function contains
        complex KV cache manipulation logic that was causing GQA compatibility issues.
        
        Kept for reference or potential future use if decode-phase pruning is needed.
        
        Original purpose:
        - Uses use_cache=True to build per-layer KV cache.
        - Applies token pruning at selected layers.
        - Prunes both hidden_states and that layer's past_key_values.
        
        Returns:
            logits: [batch, final_seq_len, vocab]
            pruning_stats: dict with global + per-layer stats
            past_key_values: list of (key, value) tuples with pruned sequence length
            attention_mask: pruned attention mask [batch, final_seq_len]
        """
        self.model.eval()
        device = self.device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        batch_size, seq_len = input_ids.shape

        # Initial embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)
        # Build causal attention mask as in training
        extended_attention = self._prepare_attention_mask(
            attention_mask, hidden_states.shape[:2], hidden_states
        )

        # We'll build a full list of past_key_values, one per layer
        past_key_values = [None] * len(self.model.model.layers)

        # Global pruning stats
        pruning_stats = {
            "total_pruning_steps": 0,
            "total_tokens_pruned": 0,
            "final_length": seq_len,
            "layer_stats": [],
        }

        for idx, block in enumerate(self.model.model.layers):
            # Position ids always match current sequence length
            curr_len = hidden_states.size(1)
            position_ids = torch.arange(curr_len, device=device).unsqueeze(0)

            # Forward through transformer block with use_cache=True
            block_outputs = block(
                hidden_states,
                attention_mask=extended_attention,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=True,
            )
            
            # Handle different output formats: tuple or BaseModelOutputWithPast
            if isinstance(block_outputs, tuple):
                hidden_states = block_outputs[0]
                layer_past = block_outputs[1]  # (key, value) or None
            else:
                # BaseModelOutputWithPast or similar object
                hidden_states = block_outputs[0]
                layer_past = getattr(block_outputs, 'past_key_value', None)
                if layer_past is None and hasattr(block_outputs, 'past_key_values'):
                    # Some models return past_key_values as a list
                    past_key_values_list = block_outputs.past_key_values
                    if past_key_values_list and len(past_key_values_list) > 0:
                        layer_past = past_key_values_list[0] if isinstance(past_key_values_list[0], tuple) else None
            
            # If layer_past is None, manually extract or create KV cache
            # Qwen2 models may not return past_key_value on first forward even with use_cache=True
            if layer_past is None:
                # Try to get KV cache by calling attention layer directly
                attn_layer = None
                if hasattr(block, 'self_attn'):
                    attn_layer = block.self_attn
                    try:
                        attn_outputs = attn_layer(
                            hidden_states,
                            attention_mask=extended_attention,
                            position_ids=position_ids,
                            past_key_value=None,
                            use_cache=True,
                            output_attentions=False,
                        )
                        if isinstance(attn_outputs, tuple) and len(attn_outputs) >= 2:
                            layer_past = attn_outputs[1]
                        elif hasattr(attn_outputs, 'past_key_value'):
                            layer_past = attn_outputs.past_key_value
                    except Exception:
                        pass  # Will create KV cache manually below
                
                # If still None, create KV cache manually
                if layer_past is None:
                    batch_size, current_seq_len, hidden_size = hidden_states.shape
                    # Get num_heads from attention layer or use default
                    if attn_layer is not None:
                        num_heads = getattr(attn_layer, 'num_heads', 
                                          getattr(attn_layer, 'num_attention_heads', 32))
                    else:
                        num_heads = getattr(block, 'num_heads', 32)
                    head_dim = hidden_size // num_heads
                    # Create placeholder KV cache matching current sequence length
                    key = torch.zeros(batch_size, num_heads, current_seq_len, head_dim,
                                     dtype=hidden_states.dtype, device=hidden_states.device)
                    value = torch.zeros(batch_size, num_heads, current_seq_len, head_dim,
                                      dtype=hidden_states.dtype, device=hidden_states.device)
                    layer_past = (key, value)

            # Optional pruning module
            # ModuleDict doesn't support .get() in all PyTorch versions, use try-except instead
            try:
                pruning_module = self.pruning_modules[str(idx)]
            except KeyError:
                pruning_module = None

            if pruning_module is not None and not self.training:
                pruning_module = pruning_module.to(device)
                # Compute saliency scores
                scores = pruning_module(hidden_states)   # [batch, seq_len, 1] or [batch, seq_len]
                scores = self._extract_keep_scores(scores)  # [batch, seq_len]

                # For now we support batch_size == 1 simplification
                assert scores.size(0) == 1, "Current E-RECAP inference assumes batch_size = 1"
                scores_1d = scores[0]  # [seq_len]

                # Reuse apply_pruning logic to get indices and pruned hidden_states & attention
                pruned_h, pruned_mask, kept_idx = self.apply_pruning(
                    hidden_states, scores_1d.unsqueeze(0), keep_ratio,
                    attention_mask=attention_mask,
                    layer_idx=idx,
                )
                # pruned_h: [1, new_len, hidden]
                # pruned_mask: [1, new_len]
                new_len = pruned_h.size(1)
                original_len = hidden_states.size(1)

                # Prune this layer's KV cache along sequence dim=2
                # Validate layer_past format
                if not isinstance(layer_past, tuple) or len(layer_past) != 2:
                    raise ValueError(
                        f"Layer {idx} past_key_value has unexpected format: {type(layer_past)}. "
                        f"Expected tuple of (key, value)."
                    )
                key, value = layer_past
                # key, value shape: [batch, num_heads, seq_len, head_dim]
                kept_flat = kept_idx.squeeze(0)  # [new_len]
                key = key.index_select(dim=2, index=kept_flat)
                value = value.index_select(dim=2, index=kept_flat)
                layer_past = (key, value)

                # Update state / stats
                hidden_states = pruned_h
                attention_mask = pruned_mask
                extended_attention = self._prepare_attention_mask(
                    attention_mask, hidden_states.shape[:2], hidden_states
                )

                tokens_kept = new_len
                tokens_pruned = original_len - new_len
                pruning_ratio = tokens_pruned / float(original_len)

                layer_stat = {
                    "layer": idx,
                    "tokens_kept": int(tokens_kept),
                    "tokens_pruned": int(tokens_pruned),
                    "pruning_ratio": float(pruning_ratio),
                    "original_length": int(original_len),
                }
                pruning_stats["total_pruning_steps"] += 1
                pruning_stats["total_tokens_pruned"] += tokens_pruned
                pruning_stats["final_length"] = tokens_kept
                pruning_stats["layer_stats"].append(layer_stat)
            else:
                # For layers without pruning, ensure KV cache matches current sequence length
                # (may be shorter if previous layers were pruned)
                if layer_past is not None and isinstance(layer_past, tuple) and len(layer_past) == 2:
                    key, value = layer_past
                    current_seq_len = hidden_states.size(1)
                    kv_seq_len = key.size(2)
                    
                    if kv_seq_len != current_seq_len:
                        if kv_seq_len > current_seq_len:
                            # Truncate to match current length
                            key = key[:, :, :current_seq_len, :]
                            value = value[:, :, :current_seq_len, :]
                            layer_past = (key, value)
                        else:
                            raise ValueError(
                                f"Layer {idx}: KV cache length {kv_seq_len} < current sequence length {current_seq_len}. "
                                f"This should not happen in normal E-RECAP flow."
                            )

            # Store (possibly pruned or adjusted) KV cache for this layer
            past_key_values[idx] = layer_past

        # Final norm + LM head
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        return logits, pruning_stats, past_key_values, attention_mask
