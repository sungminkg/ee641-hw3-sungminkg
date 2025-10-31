# EE 641 - Homework 3: Attention Mechanisms and Transformers

**Full Name:** Sungmin Kang  
**USC Email:** kangsung@usc.edu  

Starter code for implementing multi-head attention and analyzing positional encoding strategies.

## Structure

- **`problem1/`** - Multi-head attention implementation and analysis
- **`problem2/`** - Positional encoding extrapolation experiments

See individual problem READMEs for detailed instructions:
- [Problem 1 README](problem1/README.md)
- [Problem 2 README](problem2/README.md)

## Requirements

```bash
pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0
```

## Quick Start

### Problem 1: Multi-Head Attention
```bash
cd problem1
python generate_data.py --seed 641
python train.py
python analyze.py
```

### Problem 2: Positional Encoding
```bash
cd problem2
python generate_data.py --seed 641
python scripts/generate_data.py --seed 641 --generate-extrapolation
python train.py --encoding sinusoidal
python train.py --encoding learned
python train.py --encoding none
python analyze.py
```

## Folder Tree: 

ee641-hw1-sungminkg/
├── problem1/
│   ├── model.py
│   ├── dataset.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── results/
│       ├── training_log.json
│       ├── best_model.pth
│       └── visualizations/
├── problem2/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline.py
│   └── results/
│       ├── training_log.json
│       ├── heatmap_model.pth
│       ├── regression_model.pth
│       └── visualizations/
├── report.pdf
└── README.md


See the course website for complete assignment instructions, deliverables, and submission requirements.
