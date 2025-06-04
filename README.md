# 🎹 SynthWhisperer

**SynthWhisperer** is an AI assistant that speaks synthesizer fluently! Give it natural language descriptions like "warm bass" or "shimmering pad" and it'll translate them into specific synthesizer parameters.

> 🎵 **Built upon the groundbreaking Secret Sauce project** - a revolutionary machine learning system for automatically reverse engineering studio sounds. Training data and synthesizer configurations graciously provided by the Secret Sauce creator.

## ✨ What SynthWhisperer Does

- **Natural Language → Synth Parameters**: "I need a warm bass sound" → precise filter, oscillator, and envelope settings
- **Parameter Explanations**: "What does filter resonance do?" → clear, musical explanations
- **Musical Context Awareness**: Understands terms like "bass", "lead", "pad", "stab" and their typical characteristics
- **Multi-Synth Support**: Works with Noisemaker, Moog Sub Phatty, Guitar Rig, and more

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python generate_training_data.py
```

### 3. Train the Model (Optional)
```bash
python model_trainer.py
```

### 4. Try the Demo
```bash
python demo.py
```

## 🎛️ Usage Examples

**Request sounds:**
- "I need a warm bass sound for a house track"
- "Create a bright lead for synthwave"  
- "Make me something aggressive and harsh"
- "Generate a wobbly pad with movement"

**Ask questions:**
- "What does filter cutoff do?"
- "How does resonance affect the sound?"
- "Explain the amp attack parameter"

**Switch synthesizers:**
- "use moog" → switches to Moog Sub Phatty
- "use noisemaker" → switches to Noisemaker
- "synths" → lists available synthesizers

## 🧠 How It Works

SynthWhisperer combines:

1. **Semantic Mapping**: A knowledge base that maps musical terms to parameter ranges
2. **Training Data Generation**: Creates conversational examples from existing synth data
3. **Language Model Fine-tuning**: Trains a small GPT model on synthesizer knowledge
4. **Fallback System**: Works even without the trained model using semantic rules

## 📊 Training Data

The system generates training data from:
- **Semantic descriptions** → parameter mappings (warm, bright, aggressive, etc.)
- **Existing parameter data** from your Secret Sauce datasets
- **Parameter explanations** for educational responses
- **Musical context** (house, synthwave, ambient, etc.)

## 🎵 Supported Synthesizers

- **Noisemaker**: Software synthesizer with basic controls
- **Moog Sub Phatty**: Analog synthesizer with complex MIDI mappings  
- **Guitar Rig**: Guitar effects processor
- **Guitar Rig Light**: Simplified guitar effects
- **Noisemaker Light**: Simplified synthesizer controls

## 🔧 Architecture

```
User Input → SynthWhisperer → Parameter Output
     ↓            ↓              ↓
"warm bass"  → [Trained LM] → filter_cutoff: 45.2
              [Semantic Map]   osc1_octave: "16f"  
                              amp_attack: 12.8
```

## 📁 File Structure

- `semantic_mapping.py`: Core vocabulary and parameter mapping logic
- `generate_training_data.py`: Creates training dataset from existing data
- `model_trainer.py`: Trains the language model on synth knowledge
- `demo.py`: Interactive demo interface
- `requirements.txt`: Python dependencies

## 🎯 Integration with Music AI Workstation

SynthWhisperer is designed to integrate with your Music AI Workstation:

```python
from synthwhisperer.demo import SynthWhispererDemo

# Initialize
synth_ai = SynthWhispererDemo(model_path="./synthwhisperer_model")

# Get parameters from natural language
params = synth_ai.get_response("warm analog bass", synth_type="moog")

# Use in your DAW/synth integration
your_synth.apply_parameters(params)
```

## 🤖 Meet Your New Synth Assistant

SynthWhisperer bridges the gap between musical intent and technical parameters. No more endless knob-twisting – just describe the sound you want and let SynthWhisperer guide you there!

## 🙏 Credits & Attribution

SynthWhisperer is built upon the pioneering **Secret Sauce** project:

- **Training Data**: All synthesizer parameter datasets are derived from Secret Sauce's revolutionary reverse engineering system
- **Synthesizer Configurations**: JSON parameter mappings for Noisemaker, Moog Sub Phatty, Guitar Rig, and other synthesizers
- **Core Methodology**: The concept of mapping natural language to synthesizer parameters

**Special thanks https://github.com/tsellam/Secret-Sauce** for developing the foundational machine learning system that makes automatic sound reverse engineering possible, and for providing the training data that powers SynthWhisperer's intelligence.

---

**Built with ❤️ extending the Secret Sauce project**
