# DS340W_SK: LoRA rank and scaling experiments

This DS 340W project fine-tunes `google/t5-base-lm-adapt` for SST-2 sentiment classification using the original OpenDelta code. It compares original LoRA scaling (`alpha / rank`) with an optional rank-stabilized variant (`alpha / sqrt(rank)`) at ranks 2, 4, 8, and 16. The model generates `0` for a negative sentence and `1` for a positive sentence.

For the technical demo, follow the Windows instructions below. They load the models already trained on the research desktop and run new predictions on the full 436-example course Test set. You can run this path on a CPU with Python; it needs no NVIDIA GPU, CUDA, WSL, or Git.

## What is in this repository?

| Location | Contents |
| --- | --- |
| `demo/run.py` | CPU loader, live prediction, accuracy calculation, and checks against the retained prediction records. |
| `demo/checkpoints/` | Eight saved seed-42 adapters, including the trained LayerNorm parameters. These small files contain the learned changes to the shared base model. |
| `demo/data/sst2-test.jsonl` | The same 436 public SST-2 examples used as the course Test set in the experiments. |
| `demo/references/` | Retained prediction records used to verify the new predictions. They are never supplied as model inputs. |
| `demo/manifest.json` | Pinned base-model revision, file checksums, checkpoint steps, and expected correct counts. |
| `demo/low_rank_linear.py` | The `LowRankLinear` class copied from the original OpenDelta source with the same optional scaling patch used in training. |
| `requirements-demo.txt` | Separate, pinned dependencies for CPU evaluation. |
| `vendor/OpenDelta` | Original authors' code, pinned as a Git submodule. ZIP downloads omit the submodule contents; the standalone CPU demo does not need them. |
| `patches/` | Changes applied to the original training code, including `0006-optional-lora-scaling.patch`. |
| `src/`, `scripts/`, `configs/`, `requirements.lock` | Research experiment controls, configurations, and the original GPU training environment. |
| `data/results/`, `data/manifests/` | Retained research metrics, predictions, provenance, and split records. |

The protected final Validation (Unseen) partition is not included or opened by the demo. Course Test was used to select checkpoints, so these numbers reproduce the reported development comparison; they are not an untouched final assessment. The portable package covers the eight seed-42 models, not the separate confirmation seeds.

## Windows 11 setup

Use 64-bit Windows, 64-bit **Python 3.12**, and preferably 16 GB RAM. Allow about 6 GB of free disk space for Python packages, the repository, and the base model. Internet access is needed for installation and the first model download. Your laptop's integrated GPU is fine because evaluation uses the CPU.

### 1. Install Python

If Python 3.12 is not installed, open PowerShell and run:

```powershell
winget install --id Python.Python.3.12 --exact --source winget
```

You can also use the [official Python 3.12.10 Windows installer](https://www.python.org/downloads/release/python-31210/). Choose **Windows installer (64-bit)** and include the Python launcher. Close and reopen PowerShell after installation, then check:

```powershell
py -3.12 --version
```

Use Python 3.12 for the commands below even if another Python version is already installed. The separate training package's Python 3.10 requirement does not apply to this standalone demo.

On a clean Windows installation, also install Microsoft's [Visual C++ x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170), which supplies runtime libraries used by PyTorch. Choose the X64 download, run the installer, and restart if it requests a restart. This is a runtime installer; the demo does not require Visual Studio. PyTorch documents missing runtime libraries in its [Windows installation FAQ](https://docs.pytorch.org/docs/stable/notes/windows.html).

### 2. Download the project from GitHub

For the recording, show [this repository](https://github.com/BKim2002/DS340W_SK) in your browser. Choose **Code > Download ZIP**, extract the ZIP, and open PowerShell inside the extracted `DS340W_SK-main` folder. That folder must contain `README.md`, `requirements-demo.txt`, and `demo`.

Alternatively, these PowerShell commands download and extract the same repository into a new folder in Downloads. Start recording before downloading if this is your assignment demonstration:

```powershell
$demoFolder = Join-Path $env:USERPROFILE 'Downloads\DS340W-demo'
New-Item -ItemType Directory -Path $demoFolder
Invoke-WebRequest -Uri 'https://github.com/BKim2002/DS340W_SK/archive/refs/heads/main.zip' -OutFile (Join-Path $demoFolder 'DS340W_SK.zip')
Expand-Archive -LiteralPath (Join-Path $demoFolder 'DS340W_SK.zip') -DestinationPath $demoFolder
Set-Location (Join-Path $demoFolder 'DS340W_SK-main')
Get-ChildItem
```

If you already have that destination folder, choose a new empty folder name. For a stable version later, download the ZIP for the exact commit used in your recording from GitHub's commit page.

### 3. Install the CPU evaluation dependencies

Run these commands from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==24.3.1
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m pip check
```

The last command should print `No broken requirements found.` The commands use the virtual environment's Python directly, so you do not need to activate it or change PowerShell's execution policy. Install `requirements-demo.txt` for this demo; `requirements.lock` belongs to the separate GPU training environment. Do not run `pip install -e .` in the demo environment.

### 4. Run the original and modified rank-8 models

```powershell
.\.venv\Scripts\python.exe demo/run.py --model both
```

The first run downloads about **993 MB** of base-model and tokenizer files from a pinned Hugging Face revision. The eight trained adapters and the Test examples are already in the GitHub download. No Hugging Face account or token is needed. The base files are cached in `$env:USERPROFILE\.cache\ds340w-demo` and checked against their SHA-256 hashes before use.

The script evaluates one model at a time and prints its progress. When both finish, it shows this comparison:

```text
LIVE CPU RESULTS - course Test, seed 42
Model              Correct   Accuracy  Matches saved predictions
baseline           412/436      94.50%                        YES
modified           416/436      95.41%                        YES
```

Here `baseline` is original rank-8 LoRA, and `modified` is rank-8 LoRA with `alpha / sqrt(rank)` scaling. Each model must also print:

```text
VERIFIED: all 436 live CPU predictions match the retained training result.
```

These are freshly generated predictions, followed by a comparison with the saved records. If even one prediction differs, or a required file fails its checksum, the command exits with an error. Runtime depends on the CPU; allow several minutes per model on an older laptop, plus installation and download time.

### 5. Show the newly generated files

```powershell
Get-Content outputs/portable-demo/baseline/metrics.json
Get-Content outputs/portable-demo/modified/metrics.json
Get-Content outputs/portable-demo/modified/predictions.jsonl -TotalCount 5
```

Each metrics file records the computed accuracy, checkpoint hash, base-model revision, inference time, software versions, and whether all predictions matched. The prediction files contain the actual labels generated during this run. A later run of the same model replaces its files under `outputs/portable-demo/`.

## Run all eight seed-42 models

To reproduce the complete original-versus-modified rank comparison:

```powershell
.\.venv\Scripts\python.exe demo/run.py --model all
```

| Rank | Original LoRA | Modified scaling |
| --- | --- | --- |
| 2 | 413/436 (94.72%) | 412/436 (94.50%) |
| 4 | 414/436 (94.95%) | 412/436 (94.50%) |
| 8 | 412/436 (94.50%) | 416/436 (95.41%) |
| 16 | 417/436 (95.64%) | 416/436 (95.41%) |

All eight models use training seed 42 and `alpha = 2 * rank`. Their retained checkpoint steps are recorded separately in `demo/manifest.json`. The CPU demo restores those checkpoints; it does not select new ones or repeat training.

You can also evaluate one model:

```powershell
.\.venv\Scripts\python.exe demo/run.py --model original-r16
.\.venv\Scripts\python.exe demo/run.py --model modified-r16
```

The available names are `original-r2`, `modified-r2`, `original-r4`, `modified-r4`, `baseline`, `modified`, `original-r16`, and `modified-r16`.

## Recording walkthrough

Use a computer that has never run the project. You may install Python beforehand, but leave downloading and running the project for the recording. Sign into PSU Zoom, start recording, share your screen, and keep the required team members on webcam before downloading the GitHub repository.

Show the GitHub download, follow setup steps 3-5, and explain these parts of `demo/run.py` while it runs:

1. `ensure_backbone` downloads the pinned pretrained model and verifies its files.
2. `load_adapter` restores the saved LoRA and LayerNorm parameters in the corresponding attention layers.
3. `evaluate` turns each sentence into `sst2 sentence: ...`, generates `0` or `1`, and calculates accuracy from the Test labels.
4. The last check compares every newly generated prediction with the retained result. The final table and the new metrics files show the outcome.

Show the matching rows in the research paper. Use `--model both` for the two rank-8 rows or `--model all` for the complete seed-42 table. Explain that training was completed on the research desktop and that this demonstration runs saved-model evaluation on the unused laptop. Whether checkpoint evaluation satisfies the assignment is for the instructor to confirm.

## Options and troubleshooting

For a slower CPU or lower memory use, reduce the batch size and thread count:

```powershell
.\.venv\Scripts\python.exe demo/run.py --model both --batch-size 2 --threads 2
```

After the first successful download, the same computer can run without Internet:

```powershell
.\.venv\Scripts\python.exe demo/run.py --model both --offline
```

The `--offline` option checks the cached files and fails if any are missing or corrupt. To choose another cache or output location, use `--cache-dir 'D:\model-cache'` or `--output-dir 'D:\demo-results'`.

If `py -3.12` fails, install the 64-bit Python 3.12 launcher as described above and reopen PowerShell. If `requirements-demo.txt` cannot be found, use `Set-Location` to enter the folder containing this README. A failed base-model download can be retried with the same command; completed files are reused and the incomplete file is downloaded again. If a bundled checkpoint or Test file fails its checksum, download a fresh repository copy rather than editing the expected hash.

If importing PyTorch reports a missing DLL, install or repair the Microsoft Visual C++ x64 runtime linked in step 1, then reopen PowerShell. Wait for dependency installation to finish successfully before running the evaluation command.

Fast package and data checks are also available:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s demo -p 'test_*.py' -v
```

## Original training code and provenance

Training used the pinned OpenDelta source at commit `20e426f716ea53618657cd34bbfc4b347b98cd74`, with the project's patches and saved configurations. The original LoRA correction is `alpha / rank * (x @ A.T @ B.T)`; the modified block changes the divisor to `sqrt(rank)`. The portable loader attaches that same copied block in parallel with each attention projection and restores every learned LoRA and LayerNorm tensor. It uses CPU PyTorch 2.5.1 and Transformers 4.46.3, separate from the older GPU training stack.

To inspect the full authors' code, use Git with submodules:

```powershell
git clone --recurse-submodules https://github.com/BKim2002/DS340W_SK.git
```

The existing `scripts/bootstrap_wsl.sh` and research controllers are for Ubuntu 22.04 / Python 3.10 / NVIDIA CUDA training. The resolved research configurations contain workstation-specific paths and experiment authorization checks; they are not the fresh-laptop demo commands. The CPU path above is self-contained and does not read the research desktop's `artifacts/` or `runtime/` folders.

The pretrained base is [Google's T5-base-lm-adapt](https://huggingface.co/google/t5-base-lm-adapt/tree/bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf). The dataset is [Stanford SST-2](https://nlp.stanford.edu/sentiment/) from the public [GLUE archive](https://dl.fbaipublicfiles.com/glue/data/SST-2.zip); the course split is documented in `data/manifests/sst2-split-manifest.json`. Upstream code comes from [OpenDelta](https://github.com/thunlp/OpenDelta/tree/20e426f716ea53618657cd34bbfc4b347b98cd74) and [Microsoft LoRA](https://github.com/microsoft/LoRA). Their license texts and original code notice are included under `demo/licenses/` and in `demo/low_rank_linear.py`.
