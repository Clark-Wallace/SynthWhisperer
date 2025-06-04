"""
SynthWhisperer - Semantic Mapping for Synthesizer Parameters
Maps natural language descriptions to synthesizer parameter ranges and combinations.
"""

import json
import random
from typing import Dict, List, Tuple, Any

class SynthSemantics:
    """Maps musical descriptors to synthesizer parameter patterns"""
    
    def __init__(self):
        # Core semantic vocabulary for synthesizer sounds
        self.descriptors = {
            # Brightness/Timbre
            "bright": {"filter_cutoff": (80, 127), "filter_resonance": (20, 60)},
            "dark": {"filter_cutoff": (0, 40), "filter_resonance": (10, 40)},
            "warm": {"filter_cutoff": (40, 80), "filter_resonance": (15, 45), "filter_multidrive": (20, 60)},
            "cold": {"filter_cutoff": (60, 127), "filter_resonance": (0, 20), "filter_multidrive": (0, 20)},
            "crisp": {"filter_cutoff": (70, 110), "filter_resonance": (30, 70)},
            "muddy": {"filter_cutoff": (0, 30), "filter_resonance": (0, 15)},
            "sharp": {"filter_cutoff": (90, 127), "filter_resonance": (60, 100)},
            "mellow": {"filter_cutoff": (30, 70), "filter_resonance": (10, 30)},
            
            # Dynamics/Envelope
            "punchy": {"amp_attack": (0, 10), "amp_decay": (20, 60), "amp_sustain": (40, 80)},
            "soft": {"amp_attack": (30, 80), "amp_decay": (60, 100), "amp_sustain": (60, 100)},
            "plucky": {"amp_attack": (0, 5), "amp_decay": (10, 40), "amp_sustain": (0, 30)},
            "smooth": {"amp_attack": (40, 80), "amp_decay": (80, 120), "amp_sustain": (70, 100)},
            "snappy": {"amp_attack": (0, 5), "amp_decay": (5, 25), "amp_sustain": (20, 50)},
            "sustained": {"amp_attack": (20, 60), "amp_decay": (40, 80), "amp_sustain": (80, 127)},
            "percussive": {"amp_attack": (0, 10), "amp_decay": (10, 50), "amp_sustain": (0, 40)},
            
            # Movement/Modulation
            "wobbly": {"lfo1_rate": (60, 100), "lfo1_amount": (40, 80), "lfo1_destination": ["FILTER", "OSC1PITCH"]},
            "trembling": {"lfo1_rate": (80, 127), "lfo1_amount": (30, 70), "lfo1_destination": ["FILTER", "OSC2PITCH"]},
            "pulsing": {"lfo1_rate": (40, 80), "lfo1_amount": (50, 90), "lfo1_destination": ["FILTER", "PW"]},
            "static": {"lfo1_rate": (0, 20), "lfo1_amount": (0, 20), "lfo1_destination": ["NOTHING"]},
            "vibrating": {"lfo1_rate": (90, 127), "lfo1_amount": (20, 60), "lfo1_destination": ["OSC1PITCH", "OSC2PITCH"]},
            "shimmering": {"lfo1_rate": (70, 110), "lfo1_amount": (30, 60), "lfo1_destination": ["FILTER"]},
            
            # Harmonic Content
            "rich": {"osc2_volume": (60, 100), "filter_resonance": (30, 70)},
            "thin": {"osc2_volume": (0, 30), "filter_resonance": (0, 25)},
            "thick": {"osc1_volume": (80, 127), "osc2_volume": (60, 100), "filter_cutoff": (40, 80)},
            "hollow": {"osc1_volume": (40, 80), "osc2_volume": (20, 50), "filter_cutoff": (60, 100)},
            "full": {"osc1_volume": (70, 127), "osc2_volume": (50, 90), "mixer_sub": (40, 80)},
            
            # Waveform Character
            "buzzy": {"osc1_wave": ["Saw"], "osc2_wave": ["Saw"], "filter_resonance": (40, 80)},
            "smooth": {"osc1_wave": ["Sine"], "osc2_wave": ["Sine"], "filter_resonance": (10, 40)},
            "aggressive": {"osc1_wave": ["Saw"], "filter_resonance": (60, 100), "filter_multidrive": (50, 100)},
            "gentle": {"osc1_wave": ["Triangle", "Sine"], "filter_resonance": (5, 30)},
            "harsh": {"osc1_wave": ["Saw", "Pulse"], "filter_resonance": (70, 127), "filter_multidrive": (60, 127)},
            "noisy": {"osc1_wave": ["Noise"], "mixer_noise": (40, 80)},
            
            # Musical Roles
            "bass": {"osc1_octave": ["16f"], "filter_cutoff": (20, 60), "amp_attack": (0, 20)},
            "lead": {"osc1_octave": ["4f", "2f"], "filter_cutoff": (60, 100), "filter_resonance": (30, 70)},
            "pad": {"amp_attack": (40, 100), "amp_sustain": (70, 127), "filter_cutoff": (40, 80)},
            "arp": {"amp_attack": (0, 30), "amp_decay": (20, 60), "amp_sustain": (30, 70)},
            "stab": {"amp_attack": (0, 10), "amp_decay": (10, 40), "amp_sustain": (0, 50)},
        }
        
        # Combinations of descriptors that work well together
        self.combinations = {
            "warm bass": ["warm", "bass"],
            "bright lead": ["bright", "lead"],
            "wobbly pad": ["wobbly", "pad"],
            "punchy stab": ["punchy", "stab"],
            "dark atmosphere": ["dark", "pad", "sustained"],
            "aggressive lead": ["aggressive", "lead", "bright"],
            "smooth bass": ["smooth", "bass", "warm"],
            "shimmering pad": ["shimmering", "pad", "soft"],
            "percussive lead": ["percussive", "lead", "punchy"],
            "rich texture": ["rich", "thick", "warm"],
        }
        
        # Intensity modifiers
        self.intensity_modifiers = {
            "very": 1.3,
            "quite": 1.2,
            "slightly": 0.8,
            "extremely": 1.5,
            "moderately": 1.0,
            "subtly": 0.7,
        }

    def describe_to_params(self, description: str, synth_config: Dict) -> Dict[str, Any]:
        """Convert natural language description to synthesizer parameters"""
        description = description.lower().strip()
        
        # Check for exact combination matches first
        if description in self.combinations:
            descriptors = self.combinations[description]
            return self._combine_descriptors(descriptors, synth_config)
        
        # Parse individual words and intensity modifiers
        words = description.split()
        descriptors = []
        intensity = 1.0
        
        for word in words:
            if word in self.intensity_modifiers:
                intensity *= self.intensity_modifiers[word]
            elif word in self.descriptors:
                descriptors.append(word)
        
        if not descriptors:
            # Default to a basic sound if no matches
            descriptors = ["warm"]
        
        return self._combine_descriptors(descriptors, synth_config, intensity)
    
    def _combine_descriptors(self, descriptors: List[str], synth_config: Dict, intensity: float = 1.0) -> Dict[str, Any]:
        """Combine multiple descriptors into parameter values"""
        combined_params = {}
        param_ranges = {}
        
        # Collect all parameter influences
        for descriptor in descriptors:
            if descriptor in self.descriptors:
                for param, value in self.descriptors[descriptor].items():
                    if param not in param_ranges:
                        param_ranges[param] = []
                    param_ranges[param].append(value)
        
        # Generate final parameter values
        for param, ranges in param_ranges.items():
            if param in synth_config:
                config = synth_config[param]
                
                if 'range' in config:
                    # For continuous parameters, average the ranges
                    min_vals = []
                    max_vals = []
                    
                    for r in ranges:
                        if isinstance(r, tuple) and len(r) == 2:
                            min_vals.append(r[0])
                            max_vals.append(r[1])
                        elif isinstance(r, (int, float)):
                            min_vals.append(r)
                            max_vals.append(r)
                    
                    if not min_vals:
                        continue
                        
                    avg_min = sum(min_vals) / len(min_vals)
                    avg_max = sum(max_vals) / len(max_vals)
                    
                    # Apply intensity modifier
                    if intensity != 1.0:
                        center = (avg_min + avg_max) / 2
                        range_size = (avg_max - avg_min) * intensity
                        avg_min = max(0, center - range_size / 2)
                        avg_max = min(127, center + range_size / 2)
                    
                    # Generate random value in the averaged range
                    combined_params[param] = random.uniform(avg_min, avg_max)
                
                elif 'set' in config:
                    # For discrete parameters, pick the most common choice
                    choices = []
                    for r in ranges:
                        if isinstance(r, list):
                            choices.extend(r)
                        else:
                            choices.append(r)
                    
                    if choices:
                        # Pick a random choice from the accumulated options
                        choice = random.choice(choices)
                        if choice in config['labels']:
                            idx = config['labels'].index(choice)
                            combined_params[param] = config['set'][idx]
                        else:
                            combined_params[param] = choice
        
        return combined_params

    def params_to_description(self, params: Dict[str, Any], synth_config: Dict) -> str:
        """Convert synthesizer parameters back to natural language description"""
        descriptions = []
        
        # Analyze parameter values against semantic ranges
        for descriptor, param_patterns in self.descriptors.items():
            match_score = 0
            total_params = 0
            
            for param, target_range in param_patterns.items():
                if param in params:
                    total_params += 1
                    param_value = params[param]
                    
                    if isinstance(target_range, tuple):
                        # Continuous parameter
                        if target_range[0] <= param_value <= target_range[1]:
                            match_score += 1
                    elif isinstance(target_range, list):
                        # Discrete parameter - check if current value matches
                        if param in synth_config and 'labels' in synth_config[param]:
                            current_label = self._value_to_label(param_value, synth_config[param])
                            if current_label in target_range:
                                match_score += 1
            
            # If most parameters match this descriptor, include it
            if total_params > 0 and (match_score / total_params) >= 0.6:
                descriptions.append(descriptor)
        
        return " ".join(descriptions) if descriptions else "neutral"
    
    def _value_to_label(self, value, param_config: Dict) -> str:
        """Convert parameter value to its label"""
        if 'set' in param_config and 'labels' in param_config:
            # If value is already a string/label, return it
            if isinstance(value, str) and value in param_config['labels']:
                return value
            
            # Find closest value in set (for numeric values)
            try:
                closest_idx = min(range(len(param_config['set'])), 
                                 key=lambda i: abs(param_config['set'][i] - float(value)))
                return param_config['labels'][closest_idx]
            except (ValueError, TypeError):
                # If conversion fails, return the value as string
                return str(value)
        return str(value)

# Musical context generator for training data
class TrainingDataGenerator:
    """Generates contextual training examples for SynthWhisperer"""
    
    def __init__(self, semantics: SynthSemantics):
        self.semantics = semantics
        
        # Musical contexts and situations
        self.contexts = [
            "for a deep house track",
            "in a synthwave composition", 
            "for ambient music",
            "in an electro song",
            "for a techno beat",
            "in a pop ballad",
            "for film scoring",
            "in a dance track",
            "for experimental music",
            "in a retro-style song",
        ]
        
        # User request patterns
        self.request_patterns = [
            "I need a {description} sound {context}",
            "Create a {description} patch {context}",
            "Make me something {description} {context}",
            "I want a {description} synth sound {context}",
            "Generate a {description} tone {context}",
            "Can you make a {description} sound {context}",
            "I'm looking for a {description} texture {context}",
        ]
        
    def generate_training_example(self, description: str, params: Dict[str, Any]) -> Dict[str, str]:
        """Generate a conversational training example"""
        context = random.choice(self.contexts)
        pattern = random.choice(self.request_patterns)
        
        user_request = pattern.format(description=description, context=context)
        
        # Format parameters as natural response
        param_text = self._format_params_as_text(params)
        assistant_response = f"Here's a {description} sound {context}:\n\n{param_text}"
        
        return {
            "user": user_request,
            "assistant": assistant_response,
            "metadata": {
                "description": description,
                "context": context,
                "parameters": params
            }
        }
    
    def _format_params_as_text(self, params: Dict[str, Any]) -> str:
        """Format parameters in a natural, educational way"""
        lines = []
        
        # Group parameters by function
        oscillator_params = {k: v for k, v in params.items() if 'osc' in k or 'mixer' in k}
        filter_params = {k: v for k, v in params.items() if 'filter' in k}
        envelope_params = {k: v for k, v in params.items() if 'amp_' in k or 'env_' in k}
        modulation_params = {k: v for k, v in params.items() if 'lfo' in k or 'mod_' in k}
        
        if oscillator_params:
            lines.append("**Oscillators:**")
            for param, value in oscillator_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    lines.append(f"- {friendly_name}: {value:.1f}")
                else:
                    lines.append(f"- {friendly_name}: {value}")
        
        if filter_params:
            lines.append("\n**Filter:**")
            for param, value in filter_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    lines.append(f"- {friendly_name}: {value:.1f}")
                else:
                    lines.append(f"- {friendly_name}: {value}")
        
        if envelope_params:
            lines.append("\n**Envelopes:**")
            for param, value in envelope_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    lines.append(f"- {friendly_name}: {value:.1f}")
                else:
                    lines.append(f"- {friendly_name}: {value}")
        
        if modulation_params:
            lines.append("\n**Modulation:**")
            for param, value in modulation_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, (int, float)):
                    lines.append(f"- {friendly_name}: {value:.1f}")
                else:
                    lines.append(f"- {friendly_name}: {value}")
        
        return "\n".join(lines)