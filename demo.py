"""
SynthWhisperer Demo
Interactive demo of the trained synthesizer AI assistant.
"""

import os
import sys
import json
from typing import Dict, Any

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_mapping import SynthSemantics
from model_trainer import SynthWhispererInference

class SynthWhispererDemo:
    """Interactive demo for SynthWhisperer"""
    
    def __init__(self, model_path: str = None):
        # Initialize semantic mapping (works without trained model)
        self.semantics = SynthSemantics()
        
        # Load synth configs
        self.synth_configs = self._load_synth_configs()
        
        # Try to load trained model if available
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = SynthWhispererInference(model_path)
                print("✅ Loaded trained SynthWhisperer model!")
            except Exception as e:
                print(f"⚠️  Could not load model: {e}")
                print("Using semantic mapping fallback...")
        else:
            print("🔧 No trained model found, using semantic mapping...")
    
    def _load_synth_configs(self) -> Dict[str, Dict]:
        """Load synthesizer configurations"""
        configs = {}
        
        # Try multiple possible locations for synth configs
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "synth-config"),  # Local to SynthWhisperer
            os.path.join(os.path.dirname(__file__), "..", "synth-config"),  # Parent directory
            "/home/swai/secret-sauce/synth-config"  # Original location (fallback)
        ]
        
        config_files = {
            "noisemaker": "noisemaker_params.json",
            "moog": "subphatty_params.json", 
            "guitar": "guitar-rig.json"
        }
        
        for config_path in possible_paths:
            if os.path.exists(config_path):
                for synth_name, filename in config_files.items():
                    filepath = os.path.join(config_path, filename)
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            configs[synth_name] = json.load(f)
                break
        
        return configs
    
    def get_response(self, user_input: str, synth_type: str = "noisemaker") -> str:
        """Get response from SynthWhisperer"""
        
        if self.model:
            # Use trained model
            return self.model.get_synth_advice(user_input)
        else:
            # Fallback to semantic mapping
            return self._semantic_fallback(user_input, synth_type)
    
    def _semantic_fallback(self, user_input: str, synth_type: str) -> str:
        """Fallback response using semantic mapping"""
        
        if synth_type not in self.synth_configs:
            return f"Sorry, I don't know about the {synth_type} synthesizer."
        
        synth_config = self.synth_configs[synth_type]
        
        # Extract description from user input
        description = self._extract_description(user_input)
        
        if not description:
            return "I'm not sure what kind of sound you're looking for. Try describing it with words like 'warm', 'bright', 'aggressive', etc."
        
        # Generate parameters
        params = self.semantics.describe_to_params(description, synth_config)
        
        if not params:
            return f"I couldn't create parameters for '{description}'. Try a different description."
        
        # Format response
        response = f"Here's a {description} sound on the {synth_type}:\n\n"
        
        # Group parameters nicely
        oscillator_params = {k: v for k, v in params.items() if 'osc' in k}
        filter_params = {k: v for k, v in params.items() if 'filter' in k}
        envelope_params = {k: v for k, v in params.items() if 'amp_' in k or 'env_' in k}
        modulation_params = {k: v for k, v in params.items() if 'lfo' in k or 'mod_' in k}
        
        if oscillator_params:
            response += "🎵 **Oscillators:**\n"
            for param, value in oscillator_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, float):
                    response += f"   - {friendly_name}: {value:.1f}\n"
                else:
                    response += f"   - {friendly_name}: {value}\n"
        
        if filter_params:
            response += "\n🔊 **Filter:**\n"
            for param, value in filter_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, float):
                    response += f"   - {friendly_name}: {value:.1f}\n"
                else:
                    response += f"   - {friendly_name}: {value}\n"
        
        if envelope_params:
            response += "\n📈 **Envelopes:**\n"
            for param, value in envelope_params.items():
                friendly_name = param.replace('_', ' ').title()
                response += f"   - {friendly_name}: {value:.1f}\n"
        
        if modulation_params:
            response += "\n🌊 **Modulation:**\n"
            for param, value in modulation_params.items():
                friendly_name = param.replace('_', ' ').title()
                if isinstance(value, float):
                    response += f"   - {friendly_name}: {value:.1f}\n"
                else:
                    response += f"   - {friendly_name}: {value}\n"
        
        return response
    
    def _extract_description(self, user_input: str) -> str:
        """Extract sound description from user input"""
        user_input = user_input.lower()
        
        # Common patterns
        patterns = [
            "i need a ",
            "create a ",
            "make me ",
            "i want a ",
            "generate a ",
            "i'm looking for a ",
            "give me a ",
        ]
        
        description = user_input
        for pattern in patterns:
            if pattern in description:
                description = description.split(pattern, 1)[1]
                break
        
        # Remove common endings
        endings = [" sound", " patch", " tone", " texture", " synth"]
        for ending in endings:
            if description.endswith(ending):
                description = description[:-len(ending)]
        
        # Remove context phrases
        context_phrases = [
            " for a house track",
            " for synthwave", 
            " for ambient music",
            " for techno",
            " for a song",
            " for my track",
        ]
        
        for phrase in context_phrases:
            if phrase in description:
                description = description.replace(phrase, "")
        
        return description.strip()
    
    def run_interactive_demo(self):
        """Run interactive demo"""
        print("🎹 Welcome to SynthWhisperer!")
        print("Ask me for synthesizer sounds and I'll help you create them.\n")
        print("Examples:")
        print("  • 'I need a warm bass sound'")
        print("  • 'Create a bright lead for synthwave'")
        print("  • 'Make me something aggressive and harsh'")
        print("  • 'What does filter cutoff do?'")
        print("\nType 'quit' to exit, 'synths' to see available synthesizers.\n")
        
        current_synth = "noisemaker"
        print(f"🎛️  Current synth: {current_synth}")
        
        while True:
            try:
                user_input = input("\n💭 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thanks for using SynthWhisperer!")
                    break
                
                if user_input.lower() == 'synths':
                    print("🎹 Available synthesizers:")
                    for synth in self.synth_configs.keys():
                        indicator = "📍" if synth == current_synth else "  "
                        print(f"   {indicator} {synth}")
                    continue
                
                if user_input.lower().startswith('use '):
                    synth_name = user_input[4:].strip()
                    if synth_name in self.synth_configs:
                        current_synth = synth_name
                        print(f"🎛️  Switched to {current_synth}")
                    else:
                        print(f"❌ Unknown synth: {synth_name}")
                    continue
                
                if not user_input:
                    continue
                
                # Get response
                response = self.get_response(user_input, current_synth)
                print(f"\n🤖 SynthWhisperer: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Thanks for using SynthWhisperer!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Run the demo"""
    # Look for model in current directory
    model_path = os.path.join(os.path.dirname(__file__), "synthwhisperer_model")
    if not os.path.exists(model_path):
        model_path = None
    
    demo = SynthWhispererDemo(model_path)
    demo.run_interactive_demo()

if __name__ == "__main__":
    main()