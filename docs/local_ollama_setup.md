# Local Ollama Setup Guide

This guide describes how to set up this customized AI-Scientist-v2 repository on a new Linux server using local Ollama models.

## 1. Clone The Repository

Use the SSH host alias that points to the `HaeSung-Oh` GitHub account.

```bash
git clone git@github.com-haesung:HaeSung-Oh/AI-Scientist-v2-custom.git
cd AI-Scientist-v2-custom
```

If the server does not have the `github.com-haesung` SSH alias yet, add this to `~/.ssh/config` after creating and registering an SSH key in GitHub:

```sshconfig
Host github.com-haesung
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_haesung
  IdentitiesOnly yes
```

Verify access:

```bash
ssh -T git@github.com-haesung
```

Expected account:

```text
Hi HaeSung-Oh! You've successfully authenticated...
```

## 2. Create Conda Environment

```bash
conda create -n ai_scientist python=3.11 -y
conda activate ai_scientist
```

Install PyTorch. Adjust the CUDA version to match the server driver if needed.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Install system/PDF utilities used by writeup and review stages:

```bash
conda install anaconda::poppler conda-forge::chktex -y
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Quick import check:

```bash
python - <<'PY'
import torch, sklearn, scipy, cv2, skimage, timm
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sklearn", sklearn.__version__)
print("scipy", scipy.__version__)
print("cv2", cv2.__version__)
print("skimage", skimage.__version__)
print("timm", timm.__version__)
PY
```

## 3. Install Ollama

If Ollama is not installed:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:

```bash
ollama serve
```

If using `tmux`:

```bash
tmux new -s ollama
ollama serve
```

Detach with `Ctrl-b`, then `d`.

## 4. Download Local Models

Default model for this repo:

```bash
ollama pull qwen3:32b
```

Recommended code model:

```bash
ollama pull qwen2.5-coder:32b
```

Optional alternatives:

```bash
ollama pull codestral:22b
ollama pull deepseek-coder-v2:16b
```

Check installed models:

```bash
ollama list
```

Quick API test:

```bash
python - <<'PY'
import openai
client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
resp = client.chat.completions.create(
    model="qwen3:32b",
    messages=[{"role": "user", "content": "Return one short sentence."}],
    temperature=0,
)
print(resp.choices[0].message.content)
PY
```

## 5. Optional API Keys

Some code paths use the OpenAI Python client directly. Even when the model is local through Ollama, the client may still require an API key string. Use a dummy key:

```bash
export OPENAI_API_KEY="ollama"
export OPENAI_BASE_URL="http://localhost:11434/v1"
```

These values do not send requests to OpenAI when the base URL points to the local Ollama server.

Semantic Scholar works without a key, but rate limits are stricter.

```bash
export S2_API_KEY="YOUR_SEMANTIC_SCHOLAR_KEY"
```

SerpAPI is only needed for Google Scholar/general web search tools:

```bash
export SERPAPI_API_KEY="YOUR_SERPAPI_KEY"
```

For fully local LLM usage, no OpenAI/Claude key is required by the customized launcher/config.

## 6. Prepare Ideas

Local research idea files are intentionally ignored by Git:

```text
ai_scientist/ideas/*.md
ai_scientist/ideas/*.json
ai_scientist/ideas/*.py
```

Create your topic file locally, for example:

```bash
mkdir -p ai_scientist/ideas
nano ai_scientist/ideas/my_topic.md
```

Generate ideas with the local Qwen model:

```bash
python -u ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/my_topic.md \
  --model ollama/qwen3:32b \
  --max-num-generations 1 \
  --num-reflections 10
```

Expected output:

```text
ai_scientist/ideas/my_topic.json
```

## 7. Run BFTS Experiments

Default run uses `ollama/qwen3:32b` for code generation and other local-model stages:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --idea_idx 0 \
  --num_cite_rounds 2
```

Use the coder preset for code generation:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --idea_idx 0 \
  --num_cite_rounds 2 \
  --code coder
```

This maps to:

```text
ollama/qwen2.5-coder:32b
```

Use an explicit code model:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --idea_idx 0 \
  --num_cite_rounds 2 \
  --code-model ollama/codestral:22b
```

For faster debugging, skip writeup/review:

```bash
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_topic.json \
  --idea_idx 0 \
  --num_cite_rounds 2 \
  --skip_writeup \
  --skip_review
```

## 8. Logs And Outputs

Outputs are written under:

```text
experiments/<timestamp>_<idea_name>_attempt_0/
```

Important files are also copied into a lightweight backup folder during the run:

```text
experiment_backups/<timestamp>_<idea_name>_attempt_0/
```

Useful files:

```text
run_console.log
idea.md
idea.json
logs/0-run/unified_tree_viz.html
logs/0-run/stage_*/journal.json
logs/0-run/stage_*/tree_plot.html
logs/0-run/stage_*/best_solution_*.py
```

`experiments/` and `experiment_backups/` are ignored by Git.

## 9. Git Workflow Across Servers

Pull latest code on another server:

```bash
git pull
```

Commit and push code changes:

```bash
git status
git add <files>
git commit -m "Describe the change"
git push
```

Keep the original SakanaAI repository as `upstream`:

```bash
git remote -v
```

Expected:

```text
origin   git@github.com-haesung:HaeSung-Oh/AI-Scientist-v2-custom.git
upstream https://github.com/SakanaAI/AI-Scientist-v2.git
```

Fetch upstream changes if needed:

```bash
git fetch upstream
git merge upstream/main
```

## 10. Troubleshooting

If `git push` authenticates as the wrong GitHub account, check:

```bash
ssh -T git@github.com-haesung
```

If Ollama cannot be reached:

```bash
ollama serve
ollama list
```

If local model responses are too slow, reduce `agent.num_workers` in `bfts_config.yaml` to `1`.

If code generation repeatedly fails with a coder model, use the default:

```bash
--code qwen
```
