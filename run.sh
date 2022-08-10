# Setup the output directory
rm -rf timm_logs
mkdir timm_logs

# Commands for timm_models for device=cuda, dtype=float32 for training
python benchmarks/timm_models.py --float32 -dcuda --no-skip --output=timm_logs/ts_nvfuser_timm_models_float32_training_cuda.csv --training --nvfuser --speedup-dynamo-ts --use-eval-mode --isolate
python benchmarks/timm_models.py --float32 -dcuda --no-skip --output=timm_logs/aot_nvfuser_timm_models_float32_training_cuda.csv --training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode --isolate
python benchmarks/timm_models.py --float32 -dcuda --no-skip --output=timm_logs/inductor_cudagraphs_timm_models_float32_training_cuda.csv --training --inductor --use-eval-mode --isolate

