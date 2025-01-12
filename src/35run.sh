export OPENAI_API_SECRET_KEY=sk-GVuDrA85VVuWwFG77d828428274a4eFa9fC263268374C473


#export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/financial-evaluation:$PYTHONPATH
echo "Current directory: $(pwd)"

python eval.py \
    --model "gpt-3.5-turbo" \
    --tasks "LanMeF_fbp" \
    --model_args use_accelerate=True,max_gen_toks=80,use_fast=False,dtype=float16,trust_remote_code=True \
    --no_cache \
    --batch_size "2" \
    --write_out
