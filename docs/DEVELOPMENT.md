# Development Guide

**Complete setup instructions for local development, testing, and contribution.**

---

## Prerequisites

- **Python:** 3.11+
- **Git:** Version control
- **CUDA Toolkit:** 11.8+ (for GPU development)
- **cuDNN:** Compatible with your CUDA version
- **Docker:** (Optional, for containerized development)

---

## 1. Clone & Initial Setup

```bash
# Clone repository
git clone https://github.com/Mattral/Improving-Trained-LLM-Models-with-RLHF
cd Improving-Trained-LLM-Models-with-RLHF

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install development dependencies
pip install -e ".[dev]"

```

## 2. Project Structure

```
src/rlhf_platform/
├── __init__.py
├── config.py              # Configuration system (Phase 1)
├── ppo_engine.py          # PPO algorithm (Phase 2)
├── cli.py                 # CLI orchestration (Phase 3)
├── dataset.py             # Dataset pipeline (Phase 3)
├── sft_engine.py          # SFT trainer (Phase 3)
├── reward_engine.py       # Reward trainer (Phase 3)
├── dpo_engine.py          # DPO trainer (Phase 4)
├── alignment/
│   ├── __init__.py
│   ├── loss.py
│   ├── ppo_engine.py
│   └── rollout.py
├── distributed/
│   ├── __init__.py
│   ├── async_io.py
│   ├── comm_hooks.py
│   └── topology.py
└── utils/
    ├── __init__.py
    └── telemetry.py

tests/
├── __init__.py
├── test_config.py         # Config system tests (Phase 1)
├── test_ppo_engine.py     # PPO algorithm tests (Phase 2)
├── test_pipelines.py      # Integration tests (Phase 3)
└── run_benchmark_comparison.py  # Benchmarks (Phase 4)

docs/
├── index.md               # Documentation hub
├── DEVELOPMENT.md         # This file
├── PHASE_1_CONFIG.md      # Config guide
├── PHASE_2_PPO.md         # PPO guide
├── PHASE_3_CLI.md         # CLI guide
├── PHASE_4_BENCHMARKS.md  # Benchmarking guide
├── core/
│   ├── ARCHITECTURE.md
│   └── philosophy.md
├── operations/
│   ├── system_design.md
│   └── setup.md
└── governance/
    ├── contributing.md
    └── security.md

configs/
├── toy.yaml               # Small model config
└── default.yaml           # Production config

results/
└── benchmarks.md          # Benchmark results table

```

---

## 3. Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_config.py -v

# Run with coverage
pytest tests/ --cov=src/rlhf_platform --cov-report=html

# Run specific test
pytest tests/test_config.py::TestTrainingConfig -v

```

### Type Checking

```bash
# Type check all code
mypy src/ --strict

# Type check specific module
mypy src/rlhf_platform/config.py --strict

```

### Code Formatting

```bash
# Format code with Black (100 char width)
black src/ tests/ --line-length 100

# Check formatting without changing
black src/ tests/ --line-length 100 --check

# Sort imports
isort src/ tests/

```

### Linting

```bash
# Lint with ruff
ruff check src/ tests/

# Fix common issues automatically
ruff check src/ tests/ --fix

```

### Running the CLI

```bash
# Train SFT model (toy mode)
python -m rlhf_platform.cli train-sft --toy --epochs 1

# Train reward model (toy mode)
python -m rlhf_platform.cli train-reward --toy --epochs 1

# Validate PPO config
python -m rlhf_platform.cli run-ppo --toy

# See all commands
python -m rlhf_platform.cli --help

```

---

## 4. Pre-Commit Hooks

Set up pre-commit to automatically check code quality:

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run hooks manually (optional)
pre-commit run --all-files

```

**Pre-commit configuration (`.pre-commit-config.yaml`):**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ['--profile', 'black']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.0.260
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        args: [--strict]
        additional_dependencies: [pydantic, torch, transformers]
```

---

## 5. Documentation Development

### Building Docs Locally

```bash
# Install documentation tools
pip install mkdocs mkdocs-material

# Serve docs locally
mkdocs serve

# Build static site
mkdocs build

```

### Writing Documentation

- Use Markdown for all documentation
- Include code examples in documentation
- Test all code examples before committing
- Keep lines under 100 characters (except code blocks)
- Use proper heading hierarchy (H2 for sections, H3 for subsections)

### MathJax/LaTeX Equations

For mathematical notation, use inline `$...$` or display `$$...$$`:

```markdown
Inline equation: $L^{\text{CLIP}}(\theta) = ...$

Display equation:
$$D_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$
```

---

## 6. Common Development Tasks

### Adding a New Module

1. Create file in `src/rlhf_platform/`
2. Add comprehensive docstrings (Google/NumPy style)
3. Write unit tests in `tests/`
4. Update documentation with examples
5. Run full test suite and type checking
6. Ensure 100% type hint coverage

### Adding a New CLI Command

1. Add function to `src/rlhf_platform/cli.py`
2. Use `@app.command()` decorator with Typer
3. Include help text for all arguments
4. Add validation before execution
5. Test with `python -m rlhf_platform.cli <command> --help`
6. Document in [PHASE_3_CLI.md](PHASE_3_CLI.md)

### Adding Tests

```python
# tests/test_new_feature.py
import pytest
from rlhf_platform.new_module import new_function

class TestNewFeature:
    def test_basic_functionality(self):
        result = new_function()
        assert result is not None
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            new_function(invalid_input)
```

### Updating Configuration

1. Modify `src/rlhf_platform/config.py`
2. Add corresponding YAML entries in `configs/`
3. Update type hints and docstrings
4. Add validation if needed
5. Test with unit tests
6. Document in [PHASE_1_CONFIG.md](PHASE_1_CONFIG.md)

---

## 7. Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via CLI
python -m rlhf_platform.cli train-sft --toy --verbose
```

### Using Python Debugger

```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

### GPU Memory Debugging

```python
import torch

# Check GPU usage
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.max_memory_allocated() / 1024**3, "GB")

# Reset cache
torch.cuda.empty_cache()
```

---

## 8. Performance Profiling

### Profile Training Loop

```bash
# Using cProfile
python -m cProfile -s cumtime -m rlhf_platform.cli train-sft --toy --epochs 1

# Using PyTorch profiler (in code)
from torch.profiler import profile, record_function

with profile(activities=[...], record_shapes=True) as prof:
    # Your code here
    pass

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Memory Profiling

```bash
# Using memory_profiler
pip install memory_profiler

python -m memory_profiler script.py
```

---

## 9. CI/CD & GitHub Actions

The project uses GitHub Actions for:
- ✅ Type checking (mypy)
- ✅ Linting (ruff, black)
- ✅ Unit tests (pytest)
- ✅ Integration tests
- ✅ Documentation build

All PRs must pass these checks before merging.

---

## 10. Dependency Management

### Adding Dependencies

```bash
# For core functionality
pip install package_name
pip freeze > requirements.txt

# For development only
pip install --upgrade pip
pip install black isort mypy ruff pytest
```

### Updating Dependencies

```bash
# Check for updates
pip list --outdated

# Upgrade all
pip install --upgrade -r requirements.txt

# Upgrade specific package
pip install --upgrade torch
```

---

## 11. Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Create GitHub Release with notes
6. Package and upload to PyPI (when ready)

---

## 12. Troubleshooting

### Import Errors

```bash
# Ensure package is installed in editable mode
pip install -e .

# Set PYTHONPATH if needed
export PYTHONPATH=/path/to/repo/src:$PYTHONPATH
```

### CUDA Not Available

```bash
# Check installation
python -c "import torch; print(torch.cuda.is_available())"

# Verify CUDA toolkit
nvidia-smi

# Reinstall torch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Type Checking Failures

```bash
# Run mypy with more verbosity
mypy src/ --strict --show-error-codes

# Update type stubs if needed
pip install --upgrade types-*
```

---

## 13. Contributing to the Project

Please see [governance/contributing.md](governance/contributing.md) for:
- Code quality standards
- Git workflow
- PR process
- Commit message format
- Review requirements

---

## 14. Resources

- **Python:** https://docs.python.org/3.11/
- **PyTorch:** https://pytorch.org/docs/stable/index.html
- **Pydantic:** https://docs.pydantic.dev/2.0/
- **Transformers:** https://huggingface.co/docs/transformers/
- **PEFT:** https://huggingface.co/docs/peft/
- **Typer:** https://typer.tiangolo.com/

---

## 15. Common Commands Reference

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Testing
pytest tests/ -v
pytest tests/ --cov=src

# Code Quality
black src/ tests/ --line-length 100
isort src/ tests/
mypy src/ --strict
ruff check src/ tests/

# CLI
python -m rlhf_platform.cli train-sft --toy
python -m rlhf_platform.cli train-reward --toy
python -m rlhf_platform.cli run-ppo --toy

# Documentation
mkdocs serve
mkdocs build

# Cleanup
rm -rf .pytest_cache __pycache__ *.egg-info
```

---

**Last Updated:** June 1, 2026  
**For Questions:** Open an issue on GitHub or check the documentation hub
