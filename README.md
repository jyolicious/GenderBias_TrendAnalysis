# Gender Bias Trend Analysis

Analysis of gender representation in dialogue from Indian cinema subtitle files (SRT format).

## Project Structure

```
GenderBias_TrendAnalysis/
├── 01_gender_attribution.py    # Main gender attribution script
├── srt_to_csv.py               # Convert SRT files to CSV
├── diagnostic_script.py        # Validate CSV outputs
├── requirements.txt            # Python dependencies
├── cleaned_dir/                # Processed dialogue CSVs
├── dataset/                    # Input SRT files (organized by decade)
├── gendered_dir/              # Gender-attributed outputs
└── prominent_chars/           # Character gazetteers by decade
    ├── 1950s.txt
    ├── 1960s.txt
    ├── ...
    ├── male_names.txt
    └── female_names.txt
```

## Prerequisites

- Python 3.12+
- NVIDIA GPU (optional, for faster processing with Stanza)
- NVIDIA GPU drivers (if using GPU)

## Installation

### Windows

1. **Install Python 3.12+**
   ```cmd
   py install 3.12
   ```

2. **Clone the repository**
   ```cmd
   git clone <repository-url>
   cd GenderBias_TrendAnalysis
   ```

3. **Create virtual environment**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

4. **Install dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

5. **For GPU support (optional)**
   ```cmd
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

### Linux / WSL

1. **Install Python 3.12+**
   ```bash
   sudo apt update
   sudo apt install python3.12 python3.12-venv python3-pip
   ```

2. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd GenderBias_TrendAnalysis
   ```

3. **Create virtual environment**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **For GPU support (optional)**
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
   ```

## Usage

### Step 1: Convert SRT to CSV

Place your `.srt` subtitle files in `dataset/` folder (organized by decade folders like `1950s/`, `1960s/`, etc.)

```bash
# Windows
venv\Scripts\activate.bat
python srt_to_csv.py

# Linux
source venv/bin/activate
python srt_to_csv.py
```

Output: Processed dialogues in `cleaned_dir/`

### Step 2: Gender Attribution

Run gender attribution with character gazetteer matching:

```bash
# Windows
venv\Scripts\activate.bat
python 01_gender_attribution.py

# Linux
source venv/bin/activate
python 01_gender_attribution.py
```

Output: Gender-attributed dialogues in `gendered_dir/`

### Step 3: Validate Output (Optional)

```bash
python diagnostic_script.py
```

## Configuration

Edit these variables at the top of each script to customize paths:

### `srt_to_csv.py`
```python
RAW_DIR = os.path.join(BASE_DIR, "dataset")
OUT_DIR = os.path.join(BASE_DIR, "cleaned_dir")
```

### `01_gender_attribution.py`
```python
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_dir")
GENDERED_DIR = os.path.join(BASE_DIR, "gendered_dir")
GAZ_DIR = os.path.join(BASE_DIR, "prominent_chars")
```

## GPU Acceleration

The project uses **Stanza** for NLP processing, which can leverage GPU for significant speedup.

### Verify GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### GPU Configuration

In `01_gender_attribution.py`, GPU is configured with:
```python
nlp_en = stanza.Pipeline(
    lang='en',
    processors='tokenize,pos,lemma,depparse',
    use_gpu=True,      # Enable GPU
    device=0,          # GPU device ID
    batch_size=5       # Batch processing for speed
)
```

## Dependencies

Major packages:
- **PyTorch** - Deep learning framework
- **Stanza** - NLP library (Stanford NLP)
- **Transformers** - Hugging Face transformers
- **sentence-transformers** - Sentence embeddings
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning utilities
- **pysrt** - SRT file parsing

See `requirements.txt` for complete list.

## Troubleshooting

### Import errors
```bash
pip install -r requirements.txt --force-reinstall
```

### GPU not detected
- Verify NVIDIA drivers: `nvidia-smi`
- Reinstall CUDA-enabled PyTorch (see installation steps)
- Check CUDA compatibility with your GPU

### Path errors
All scripts use cross-platform paths with `os.path.join()`. Ensure you're running from project root.

## Notes

- All paths are now cross-platform compatible (Windows/Linux)
- Scripts auto-detect their location using `BASE_DIR`
- Virtual environment (`venv/`) should be excluded from version control
- Add `.gitignore` with: `venv/`, `myenv/`, `__pycache__/`, `*.pyc`

## License

[Add your license here]

## Contributors

[Add contributors here]
