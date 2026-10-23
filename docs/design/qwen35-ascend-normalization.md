# Qwen3.5 normalization on Ascend

The unquantized Qwen3.5 model uses the existing runtime model and residual
flow. Its `GemmaRMSNorm` path on Ascend adds the residual in the model dtype,
computes the mean square, reciprocal square root and `1 + weight` multiplier
in FP32, then casts the normalized output to the input dtype. The returned
residual is the sum before normalization. The FP8/FP4 convenience methods
accept only a `None` scale on Ascend and return no quantized copy; requesting
a quantized copy raises an explicit error.

The optional MoE shared-expert stream is created only on NVIDIA. Ascend
executes that work inline without constructing a CUDA stream. These changes
do not establish full model support by themselves: the attention and Gated
DeltaNet backends must also support Ascend.

`test/runtime/layers/test_ascend_gemma_rmsnorm.py` exercises the actual class's
FP32/BF16 arithmetic, weight loading, empty batches, residual outputs and
unquantized convenience methods on CPU. Set `TOKENSPEED_TEST_NPU=1` after
selecting an available chip with `ASCEND_RT_VISIBLE_DEVICES` to also run six
cases importing the full runtime class and comparing actual NPU outputs
against a CPU reference. Complete-model numerical comparisons are separate.
