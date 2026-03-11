# poorman_autoresearch
Automated neural networks training guided by LLMs, following the same ideas as the karpathy's "autoresearch" repo, but doing this without a coding agent, and just a harness inside the code that can call tools

# Examples
python run.py --max-runs 5             # quick test, ~25 min total

python run.py --skip-prepare           # data already downloaded

python run.py --model llama3.3:70b     # different LLM

python run.py --num-shards 4           # smaller download for testing
