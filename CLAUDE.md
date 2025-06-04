# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the SynthWhisperer codebase.

## Project Overview

SynthWhisperer is an AI assistant that translates natural language descriptions into synthesizer parameters. It's built upon the Secret Sauce project's foundational research in automatic sound reverse engineering.

**Core Innovation**: Bridge the gap between musical intent ("warm bass sound") and technical parameters (filter_cutoff: 35.2, osc1_octave: "16f").

## Architecture

### Core Components

- **`demo.py`**: Main interactive interface
  - Chat-based UI for natural language requests
  - Supports multiple synthesizers (noisemaker, moog, guitar)
  - Fallback architecture: trained model → semantic mapping
  - Usage: `python demo.py`

- **`semantic_mapping.py`**: Rule-based parameter generation
  - Maps musical descriptors to parameter ranges
  - Handles context (bass, lead, pad, etc.)
  - Reliable fallback when ML model unavailable
  - Core vocabulary: warm, bright, aggressive, wobbly, etc.

- **`generate_training_data.py`**: Training data pipeline
  - Converts Secret Sauce parameter datasets to conversational format
  - Generates 6,160+ training examples across 5 synthesizers
  - Creates realistic user requests and expert responses
  - Usage: `python generate_training_data.py`

- **`model_trainer.py`**: ML model training
  - Trains lookup-based model on conversational data
  - Quick training approach for fast iteration
  - Outputs trained model for enhanced responses
  - Usage: `python model_trainer.py`

- **`synth-config/*.json`**: Synthesizer parameter definitions
  - Inherited from Secret Sauce project
  - Maps human-readable parameters to MIDI CC values
  - Supports complex synthesizers (Moog Sub Phatty, Guitar Rig)

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Core dependencies:
- No heavy ML frameworks required for basic operation
- Optional: transformers/torch for enhanced ML model training
- JSON for synthesizer configurations
- Standard Python libraries (random, re, etc.)

## Common Development Tasks

### Run Interactive Demo
```bash
python demo.py
```

### Generate New Training Data
```bash
python generate_training_data.py
```

### Train Enhanced Model
```bash
python model_trainer.py
```

### Test Semantic Mapping
```python
from semantic_mapping import SynthSemantics
semantics = SynthSemantics()
params = semantics.describe_to_params("warm bass", synth_config)
```

## Key Design Principles

1. **Graceful Degradation**: Always works even without trained model
2. **Musical Context**: Understands bass vs lead vs pad characteristics  
3. **Educational**: Explains parameters, doesn't just provide values
4. **Extensible**: Easy to add new synthesizers and descriptors
5. **Attribution**: Maintains clear connection to Secret Sauce heritage

## Secret Sauce Integration

SynthWhisperer builds on Secret Sauce foundations:
- **Training Data**: Derived from Secret Sauce parameter datasets
- **Synth Configs**: Uses Secret Sauce JSON parameter mappings
- **Methodology**: Extends Secret Sauce's parameter→sound research to language→parameter translation

## Development Notes

- Code is Python 3 compatible (unlike original Secret Sauce Python 2)
- Semantic mapping provides deterministic fallback behavior
- Training data generation is reproducible with fixed random seeds
- Model architecture prioritizes speed and interpretability over complexity
- All paths are relative for standalone repository deployment

## Important Context

SynthWhisperer represents the "input side" of the Secret Sauce vision:
- **Secret Sauce**: audio → parameters (reverse engineering)
- **SynthWhisperer**: description → parameters (forward generation)

Together they complete the creative workflow: describe → generate → refine → analyze.

## Attribution Requirements

When modifying or extending SynthWhisperer:
- Maintain credits to Secret Sauce creator in README
- Preserve attribution notice in LICENSE
- Reference Secret Sauce heritage in documentation
- Acknowledge training data provenance

The Secret Sauce creator's pioneering work made SynthWhisperer possible.