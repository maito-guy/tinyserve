# Running on Colab / Kaggle (free T4)

Runtime > Change runtime type > T4 GPU. Then, in the first cell:

```python
!git clone https://github.com/<you>/tinyserve.git
%cd tinyserve
!pip install -q -r requirements.txt
!nvidia-smi --query-gpu=name,memory.total --format=csv
```

Then either write code in the repo files and `!python -m bench.run ...`,
or import from the package in cells:

```python
from tinyserve.model_loader import load
model, tok, device = load()
```

Commit from Colab when a step passes:

```python
!git config user.email "you@example.com" && git config user.name "you"
!git add -A && git commit -m "ch1: cached loop passes tests" && git push
```

(Use a GitHub token as the remote URL, or mount Drive and push from a
local clone. Push at the end of every session; Colab wipes the disk.)

Model download cache: `/root/.cache/huggingface`. It is ~1 GB and is
lost when the runtime resets; that is fine, it re-downloads in a minute.
