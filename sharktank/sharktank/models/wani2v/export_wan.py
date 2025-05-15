from sharktank.models.flux.export import export_flux_transformer_model_mlir


flux_transformer_default_batch_sizes = [1]
model_path="/home/chi/src/compute-poc/multimodal/wani2v/score_model_state.irpa"
mlir_output_path="/home/chi/src/compute-poc/multimodal/wani2v/score_model_state.mlir"
# model_path="/data/flux/FLUX.1-dev/transformer/model.irpa"
# mlir_output_path="/home/chi/src/compute-poc/multimodal/wani2v/flux_model_state.mlir"
# # k:  meta
# # v:  {'_class_name': 'FluxTransformer2DModel', '_diffusers_version': '0.30.0.dev0', '_name_or_path': '../checkpoints/flux-dev/transformer'}
# # k:  hparams
# # v:  {'attention_head_dim': 128, 'guidance_embeds': True, 'in_channels': 64, 'joint_attention_dim': 4096, 'num_attention_heads': 24, 'num_layers': 19, 'num_single_layers': 38, 'patch_size': 1, 'pooled_projection_dim': 768}
# # k:  SHARK_DATASET_VERSION
# # v:  1
export_flux_transformer_model_mlir(
    model_path, output_path=mlir_output_path, batch_sizes=flux_transformer_default_batch_sizes
)



# (.venv.py311) (.venv) ➜  shark-ai git:(wan) ✗ python -m sharktank.models.wani2v.export_wan                                                                                                            

# 2025-05-13 14:13:08.852834: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2025-05-13 14:13:08.854136: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
# 2025-05-13 14:13:08.857570: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
# 2025-05-13 14:13:08.864019: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
# 2025-05-13 14:13:08.872901: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
# 2025-05-13 14:13:08.875519: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
# 2025-05-13 14:13:08.884069: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
# To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2025-05-13 14:13:09.527314: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
# /home/chi/src/compute-poc/.venv.py311/lib/python3.11/site-packages/iree/turbine/aot/params.py:163: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
#   return torch.from_numpy(wrapper)
# Traceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code
#   File "/home/chi/src/shark-ai/sharktank/sharktank/models/flux/wan.py", line 7, in <module>
#     export_flux_transformer_model_mlir(
#   File "/home/chi/src/shark-ai/sharktank/sharktank/models/flux/export.py", line 38, in export_flux_transformer_model_mlir
#     params=FluxParams.from_hugging_face_properties(dataset.properties),
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/chi/src/shark-ai/sharktank/sharktank/models/flux/flux.py", line 130, in from_hugging_face_properties
#     **cls.translate_hugging_face_config_dict_into_init_kwargs(properties)
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/home/chi/src/shark-ai/sharktank/sharktank/models/flux/flux.py", line 84, in translate_hugging_face_config_dict_into_init_kwargs
#     vec_in_dim = properties["pooled_projection_dim"]
#                  ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
# KeyError: 'pooled_projection_dim'