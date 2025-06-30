(Pdb) model
PagedLlmModelV1(
  (token_embedding): TokenEmbeddingLayer()
  (attention_embedding): ModuleList(
    (0): RotaryEmbeddingLayer()
  )
  (output_norm): RMSNormLayer()
  (output_lm_head): LinearLayer()
  (attn_blocks): ModuleList(
    (0): AttentionFFNBlock(
      (attn): PagedLlamaAttentionBlock(
        (attn_q): LinearLayer()
        (attn_k): LinearLayer()
        (attn_v): LinearLayer()
        (qk_norm): L2Norm()
        (attn_norm): RMSNormLayer()
        (attn_output): LinearLayer()
        (attn_output_norm): Identity()
      )
      (ffn_norm): RMSNormLayer()
      (ffn): FFN(
        (ffn_gate): LinearLayer()
        (ffn_up): LinearLayer()
        (ffn_down): LinearLayer()
      )
    )
    (1): AttentionFFNBlock(
      (attn): PagedLlamaAttentionBlock(
        (attn_q): LinearLayer()
        (attn_k): LinearLayer()
        (attn_v): LinearLayer()
        (qk_norm): L2Norm()
        (attn_norm): RMSNormLayer()
        (attn_output): LinearLayer()
        (attn_output_norm): Identity()
      )
      (ffn_norm): RMSNormLayer()
      (ffn): MoeBlock(
        (layer_output_norm): Identity()
        (ffn_gate_inp): LinearLayer()
        (routed_experts): PreGatherFFNMOE()
        (shared_experts): FFN(
          (ffn_gate): LinearLayer()
          (ffn_up): LinearLayer()
          (ffn_down): LinearLayer()
        )
      )
    )
    (2): AttentionFFNBlock(
      (attn): PagedLlamaAttentionBlock(
        (attn_q): LinearLayer()
        (attn_k): LinearLayer()
        (attn_v): LinearLayer()
        (qk_norm): L2Norm()
        (attn_norm): RMSNormLayer()
        (attn_output): LinearLayer()
        (attn_output_norm): Identity()
      )
      (ffn_norm): RMSNormLayer()
      (ffn): FFN(
        (ffn_gate): LinearLayer()
        (ffn_up): LinearLayer()
        (ffn_down): LinearLayer()
      )
    )
    (3): AttentionFFNBlock(
      (attn): PagedLlamaAttentionBlock(
        (attn_q): LinearLayer()
        (attn_k): LinearLayer()
        (attn_v): LinearLayer()
        (attn_norm): RMSNormLayer()
        (attn_output): LinearLayer()
        (attn_output_norm): Identity()
      )
      (ffn_norm): RMSNormLayer()
      (ffn): MoeBlock(
        (layer_output_norm): Identity()
        (ffn_gate_inp): LinearLayer()
        (routed_experts): PreGatherFFNMOE()
        (shared_experts): FFN(
          (ffn_gate): LinearLayer()
          (ffn_up): LinearLayer()
          (ffn_down): LinearLayer()
        )
      )
    )
  )
)
(Pdb) hf_model
Llama4ForCausalLM(
  (model): Llama4TextModel(
    (embed_tokens): Embedding(19, 160)
    (layers): ModuleList(
      (0): Llama4TextDecoderLayer(
        (self_attn): Llama4TextAttention(
          (q_proj): Linear(in_features=160, out_features=160, bias=False)
          (k_proj): Linear(in_features=160, out_features=32, bias=False)
          (v_proj): Linear(in_features=160, out_features=32, bias=False)
          (o_proj): Linear(in_features=160, out_features=160, bias=False)
          (qk_norm): Llama4TextL2Norm(eps=0.01)
        )
        (feed_forward): Llama4TextMLP(
          (gate_proj): Linear(in_features=160, out_features=23, bias=False)
          (up_proj): Linear(in_features=160, out_features=23, bias=False)
          (down_proj): Linear(in_features=23, out_features=160, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
        (post_attention_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
      )
      (1): Llama4TextDecoderLayer(
        (self_attn): Llama4TextAttention(
          (q_proj): Linear(in_features=160, out_features=160, bias=False)
          (k_proj): Linear(in_features=160, out_features=32, bias=False)
          (v_proj): Linear(in_features=160, out_features=32, bias=False)
          (o_proj): Linear(in_features=160, out_features=160, bias=False)
          (qk_norm): Llama4TextL2Norm(eps=0.01)
        )
        (feed_forward): Llama4TextMoe(
          (experts): Llama4TextExperts(
            (act_fn): SiLU()
          )
          (router): Linear(in_features=160, out_features=3, bias=False)
          (shared_expert): Llama4TextMLP(
            (gate_proj): Linear(in_features=160, out_features=29, bias=False)
            (up_proj): Linear(in_features=160, out_features=29, bias=False)
            (down_proj): Linear(in_features=29, out_features=160, bias=False)
            (activation_fn): SiLU()
          )
        )
        (input_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
        (post_attention_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
      )
      (2): Llama4TextDecoderLayer(
        (self_attn): Llama4TextAttention(
          (q_proj): Linear(in_features=160, out_features=160, bias=False)
          (k_proj): Linear(in_features=160, out_features=32, bias=False)
          (v_proj): Linear(in_features=160, out_features=32, bias=False)
          (o_proj): Linear(in_features=160, out_features=160, bias=False)
          (qk_norm): Llama4TextL2Norm(eps=0.01)
        )
        (feed_forward): Llama4TextMLP(
          (gate_proj): Linear(in_features=160, out_features=23, bias=False)
          (up_proj): Linear(in_features=160, out_features=23, bias=False)
          (down_proj): Linear(in_features=23, out_features=160, bias=False)
          (activation_fn): SiLU()
        )
        (input_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
        (post_attention_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
      )
      (3): Llama4TextDecoderLayer(
        (self_attn): Llama4TextAttention(
          (q_proj): Linear(in_features=160, out_features=160, bias=False)
          (k_proj): Linear(in_features=160, out_features=32, bias=False)
          (v_proj): Linear(in_features=160, out_features=32, bias=False)
          (o_proj): Linear(in_features=160, out_features=160, bias=False)
        )
        (feed_forward): Llama4TextMoe(
          (experts): Llama4TextExperts(
            (act_fn): SiLU()
          )
          (router): Linear(in_features=160, out_features=3, bias=False)
          (shared_expert): Llama4TextMLP(
            (gate_proj): Linear(in_features=160, out_features=29, bias=False)
            (up_proj): Linear(in_features=160, out_features=29, bias=False)
            (down_proj): Linear(in_features=29, out_features=160, bias=False)
            (activation_fn): SiLU()
          )
        )
        (input_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
        (post_attention_layernorm): Llama4TextRMSNorm((160,), eps=0.01)
      )
    )
    (norm): Llama4TextRMSNorm((160,), eps=0.01)
    (rotary_emb): Llama4TextRotaryEmbedding()
  )
  (lm_head): Linear(in_features=160, out_features=19, bias=False)
)
