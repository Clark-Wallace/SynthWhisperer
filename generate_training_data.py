"""
SynthWhisperer Training Data Generator
Creates conversational training data from existing synth parameter datasets.
"""

import json
import csv
import os
import random
from typing import Dict, List, Any, Tuple
from semantic_mapping import SynthSemantics, TrainingDataGenerator

class SynthTrainingDataPipeline:
    """Converts existing Secret Sauce data into LLM training format"""
    
    def __init__(self, secret_sauce_root: str = None):
        # Auto-detect Secret Sauce root path
        if secret_sauce_root is None:
            # Try multiple locations
            possible_paths = [
                "/home/swai/secret-sauce",  # Original location
                os.path.join(os.path.dirname(__file__), ".."),  # Parent directory
                os.path.dirname(__file__)  # Current directory
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, "synth-config")) or os.path.exists(os.path.join(path, "data")):
                    self.root_path = path
                    break
            else:
                self.root_path = os.path.dirname(__file__)  # Fallback to current directory
        else:
            self.root_path = secret_sauce_root
            
        self.semantics = SynthSemantics()
        self.training_gen = TrainingDataGenerator(self.semantics)
        
        # Load synth configurations
        self.synth_configs = self._load_synth_configs()
        
        # Extended vocabulary for more diverse training data
        self.extended_descriptors = [
            # Single descriptors
            "warm", "bright", "dark", "punchy", "soft", "aggressive", "smooth",
            "harsh", "gentle", "rich", "thin", "buzzy", "wobbly", "static",
            "crisp", "muddy", "sharp", "mellow", "plucky", "sustained",
            
            # Combined descriptors
            "warm bass", "bright lead", "dark pad", "punchy stab", "smooth texture",
            "aggressive lead", "gentle pad", "rich bass", "thin lead", "buzzy bass",
            "wobbly pad", "crisp lead", "mellow pad", "plucky bass", "sustained pad",
            "shimmering lead", "percussive stab", "full bass", "hollow lead",
            
            # Context-specific
            "cinematic pad", "retro bass", "modern lead", "vintage texture",
            "digital stab", "analog warmth", "glitchy texture", "ethereal pad",
            "driving bass", "soaring lead", "atmospheric texture", "rhythmic stab",
        ]
        
    def _load_synth_configs(self) -> Dict[str, Dict]:
        """Load all synthesizer configuration files"""
        configs = {}
        config_path = os.path.join(self.root_path, "synth-config")
        
        config_files = {
            "noisemaker": "noisemaker_params.json",
            "noisemaker_light": "noisemaker_light_params.json", 
            "moog": "subphatty_params.json",
            "guitar": "guitar-rig.json",
            "guitar_light": "guitar-rig-light.json"
        }
        
        for synth_name, filename in config_files.items():
            filepath = os.path.join(config_path, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    configs[synth_name] = json.load(f)
        
        return configs
    
    def load_existing_data(self, synth_name: str) -> List[Dict[str, Any]]:
        """Load existing parameter data from CSV files"""
        data_path = os.path.join(self.root_path, "data", synth_name, f"{synth_name}.csv")
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}")
            return []
        
        params_list = []
        with open(data_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to float where applicable
                params = {}
                for key, value in row.items():
                    if key == 'file':
                        continue
                    try:
                        # Try to convert to float
                        params[key] = float(value)
                    except ValueError:
                        # Keep as string for discrete parameters
                        params[key] = value
                params_list.append(params)
        
        return params_list
    
    def generate_semantic_training_data(self, synth_name: str, num_examples: int = 1000) -> List[Dict]:
        """Generate training examples based on semantic descriptions"""
        if synth_name not in self.synth_configs:
            raise ValueError(f"Unknown synth: {synth_name}")
        
        training_examples = []
        synth_config = self.synth_configs[synth_name]
        
        for _ in range(num_examples):
            # Pick a random description
            description = random.choice(self.extended_descriptors)
            
            # Generate parameters from semantic description
            params = self.semantics.describe_to_params(description, synth_config)
            
            # Fill in missing parameters with random values
            params = self._fill_missing_params(params, synth_config)
            
            # Generate conversational training example
            example = self.training_gen.generate_training_example(description, params)
            example['synth'] = synth_name
            
            training_examples.append(example)
        
        return training_examples
    
    def augment_existing_data(self, synth_name: str) -> List[Dict]:
        """Create descriptions for existing parameter combinations"""
        existing_params = self.load_existing_data(synth_name)
        if not existing_params:
            return []
        
        training_examples = []
        synth_config = self.synth_configs[synth_name]
        
        for params in existing_params[:500]:  # Limit to avoid too much data
            # Generate description from parameters
            description = self.semantics.params_to_description(params, synth_config)
            
            # Create training example
            example = self.training_gen.generate_training_example(description, params)
            example['synth'] = synth_name
            example['source'] = 'existing_data'
            
            training_examples.append(example)
        
        return training_examples
    
    def generate_parameter_explanation_data(self, synth_name: str, num_examples: int = 200) -> List[Dict]:
        """Generate training data for parameter explanations"""
        training_examples = []
        synth_config = self.synth_configs[synth_name]
        
        explanations = {
            "filter_cutoff": "Controls how bright or dark the sound is. Higher values make it brighter.",
            "filter_resonance": "Adds emphasis around the filter cutoff frequency. Higher values create more resonant, 'honky' sounds.",
            "amp_attack": "How quickly the sound reaches full volume when a note is played. Lower values = faster attack.",
            "amp_decay": "How quickly the sound drops from peak to sustain level.",
            "amp_sustain": "The volume level maintained while holding a note.",
            "lfo1_rate": "Speed of the low frequency oscillator modulation. Higher values = faster wobble/vibrato.",
            "lfo1_amount": "Intensity of the LFO modulation effect.",
            "osc1_volume": "Volume level of the primary oscillator.",
            "osc2_volume": "Volume level of the secondary oscillator.",
            "osc1_pulse_width": "Shape of the pulse wave. 50% = square wave, other values create different timbres.",
        }
        
        question_patterns = [
            "What does {param} do?",
            "Explain the {param} parameter",
            "How does {param} affect the sound?",
            "What happens when I change {param}?",
            "Tell me about {param}",
        ]
        
        for param, explanation in explanations.items():
            if param in synth_config:
                for _ in range(num_examples // len(explanations)):
                    pattern = random.choice(question_patterns)
                    question = pattern.format(param=param.replace('_', ' '))
                    
                    training_examples.append({
                        "user": question,
                        "assistant": explanation,
                        "synth": synth_name,
                        "type": "parameter_explanation"
                    })
        
        return training_examples
    
    def _fill_missing_params(self, params: Dict[str, Any], synth_config: Dict) -> Dict[str, Any]:
        """Fill in missing parameters with reasonable random values"""
        filled_params = params.copy()
        
        for param_name, param_config in synth_config.items():
            if param_name not in filled_params:
                if 'range' in param_config:
                    # Random value in range
                    min_val, max_val = param_config['range']
                    filled_params[param_name] = random.uniform(min_val, max_val)
                elif 'set' in param_config:
                    # Random choice from set
                    filled_params[param_name] = random.choice(param_config['set'])
        
        return filled_params
    
    def create_full_dataset(self, output_path: str = "synthwhisperer_training_data.jsonl"):
        """Generate complete training dataset for all synthesizers"""
        all_examples = []
        
        for synth_name in self.synth_configs.keys():
            print(f"Generating data for {synth_name}...")
            
            # Generate semantic-based examples
            semantic_examples = self.generate_semantic_training_data(synth_name, 800)
            all_examples.extend(semantic_examples)
            
            # Augment with existing data
            augmented_examples = self.augment_existing_data(synth_name)
            all_examples.extend(augmented_examples)
            
            # Add parameter explanations
            explanation_examples = self.generate_parameter_explanation_data(synth_name, 100)
            all_examples.extend(explanation_examples)
        
        # Shuffle the dataset
        random.shuffle(all_examples)
        
        # Save as JSONL format
        output_path = os.path.join(self.root_path, "synthwhisperer", output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Generated {len(all_examples)} training examples")
        print(f"Saved to: {output_path}")
        
        # Create statistics
        self._print_dataset_stats(all_examples)
        
        return output_path
    
    def _print_dataset_stats(self, examples: List[Dict]):
        """Print statistics about the generated dataset"""
        synth_counts = {}
        type_counts = {}
        
        for example in examples:
            synth = example.get('synth', 'unknown')
            synth_counts[synth] = synth_counts.get(synth, 0) + 1
            
            example_type = example.get('type', 'semantic')
            type_counts[example_type] = type_counts.get(example_type, 0) + 1
        
        print("\nDataset Statistics:")
        print("=" * 30)
        print("Examples per synthesizer:")
        for synth, count in synth_counts.items():
            print(f"  {synth}: {count}")
        
        print("\nExample types:")
        for ex_type, count in type_counts.items():
            print(f"  {ex_type}: {count}")

def main():
    """Generate the training dataset"""
    pipeline = SynthTrainingDataPipeline()
    dataset_path = pipeline.create_full_dataset()
    print(f"\nTraining dataset created: {dataset_path}")

if __name__ == "__main__":
    main()