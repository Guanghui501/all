#!/usr/bin/env python
"""
Fine-Grained Cross-Modal Attention Analysis Demo

This script demonstrates how to use the fine-grained attention mechanism
to analyze atom-level and token-level interactions in material property prediction.

Usage:
    python demo_fine_grained_attention.py \
        --model_path /path/to/checkpoint.pt \
        --cif_path /path/to/structure.cif \
        --text "Material description..." \
        --save_dir ./results
"""

import argparse
import torch
from pathlib import Path

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.core.specie import chem_data, get_node_attributes
from transformers import BertTokenizer
import numpy as np

# Import from the same module used for training
from models.alignn import ALIGNN, ALIGNNConfig
from interpretability_enhanced import EnhancedInterpretabilityAnalyzer


def load_model_with_fine_grained_attention(checkpoint_path, device='cuda'):
    """Load model with fine-grained attention enabled."""

    # Load checkpoint (set weights_only=False for backward compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load config from checkpoint
    if 'config' in checkpoint:
        # Use saved config directly - it was created with the same module used for training
        config = checkpoint['config']
        print("✅ Loaded config from checkpoint")
        print(f"   - use_cross_modal_attention: {getattr(config, 'use_cross_modal_attention', False)}")
        print(f"   - use_middle_fusion: {getattr(config, 'use_middle_fusion', False)}")
        print(f"   - use_fine_grained_attention: {getattr(config, 'use_fine_grained_attention', False)}")
    else:
        # Fall back: infer config from checkpoint state_dict
        print("⚠️  Config not found in checkpoint, inferring from state_dict...")
        checkpoint_state = checkpoint.get('model', checkpoint)
        state_keys = list(checkpoint_state.keys())

        # Infer use_cross_modal_attention from fc1.weight shape
        use_cross_modal = False
        if 'fc1.weight' in checkpoint_state:
            fc1_shape = checkpoint_state['fc1.weight'].shape
            use_cross_modal = (fc1_shape[1] == 64)  # [64, 64] means cross-modal
            print(f"   - fc1.weight shape: {fc1_shape} -> use_cross_modal_attention={use_cross_modal}")

        # Infer other settings from state_dict keys
        use_middle_fusion = any('middle_fusion' in k for k in state_keys)
        use_fine_grained = any('fine_grained_attention' in k for k in state_keys)

        print(f"   - use_middle_fusion: {use_middle_fusion}")
        print(f"   - use_fine_grained_attention: {use_fine_grained}")

        config = ALIGNNConfig(
            name="alignn",
            alignn_layers=4,
            gcn_layers=4,
            atom_input_features=92,
            hidden_features=256,
            output_features=1,
            use_cross_modal_attention=use_cross_modal,
            cross_modal_hidden_dim=256,
            cross_modal_num_heads=4,
            use_middle_fusion=use_middle_fusion,
            use_fine_grained_attention=use_fine_grained,
        )

    # Create model
    model = ALIGNN(config)

    # Load weights
    checkpoint_state = checkpoint.get('model', checkpoint)
    model.load_state_dict(checkpoint_state, strict=False)

    model = model.to(device)
    model.eval()

    print("✅ Model loaded successfully")

    return model, config


def cif_to_graph(cif_path, cutoff=8.0, max_neighbors=12):
    """Convert CIF file to DGL graph."""

    # Read structure
    atoms = Atoms.from_cif(cif_path)

    # Create graph using JARVIS
    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="atomic_number",
        compute_line_graph=True,
        use_canonize=True
    )

    # Build feature lookup table (same as StructureDataset)
    max_z = max(v["Z"] for v in chem_data.values())
    template = get_node_attributes("C", atom_features="cgcnn")
    features = np.zeros((1 + max_z, len(template)))

    for element, v in chem_data.items():
        z = v["Z"]
        x = get_node_attributes(element, atom_features="cgcnn")
        if x is not None:
            features[z, :] = x

    # Convert atomic_number to cgcnn features
    z = g.ndata.pop("atom_features")
    g.ndata["atomic_number"] = z
    z = z.type(torch.LongTensor).squeeze()
    f = torch.tensor(features[z], dtype=torch.float32)
    g.ndata["atom_features"] = f

    return g, lg, atoms


def analyze_with_fine_grained_attention(
    model,
    g,
    lg,
    text,
    atoms_object,
    save_dir=None
):
    """Perform analysis with fine-grained attention."""

    device = next(model.parameters()).device
    g = g.to(device)
    lg = lg.to(device)

    # Get prediction with attention weights
    with torch.no_grad():
        output = model(
            [g, lg, [text]],
            return_features=True,
            return_attention=True
        )

    prediction = output['predictions'].cpu().item()

    # Extract fine-grained attention weights
    fg_attn = output.get('fine_grained_attention_weights', None)

    if fg_attn is None:
        print("❌ Fine-grained attention weights not found!")
        print("   Make sure the model has use_fine_grained_attention=True")
        return

    print(f"\n✅ Prediction: {prediction:.4f}")
    print(f"\n✅ Fine-grained attention extracted:")
    print(f"   - atom_to_text shape: {fg_attn['atom_to_text'].shape}")
    print(f"   - text_to_atom shape: {fg_attn['text_to_atom'].shape}")

    # Tokenize text to get token strings
    tokenizer = BertTokenizer.from_pretrained('m3rg-iitd/matscibert')
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']  # Add special tokens

    # Truncate or pad to match attention shape
    seq_len = fg_attn['atom_to_text'].shape[-1]
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))

    # Create analyzer for visualization
    analyzer = EnhancedInterpretabilityAnalyzer(model, device=device)

    # Visualize fine-grained attention
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / 'fine_grained_attention.png'
    else:
        save_path = 'fine_grained_attention.png'

    analysis = analyzer.visualize_fine_grained_attention(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text_tokens=tokens,
        save_path=save_path,
        top_k_atoms=10,
        top_k_words=15,
        show_all_heads=False
    )

    # Print analysis results
    print(f"\n{'='*80}")
    print("📊 Fine-Grained Attention Analysis")
    print(f"{'='*80}\n")

    if analysis and 'overall_top_words' in analysis:
        print("🔤 Top 10 Most Important Words (overall):")
        print(f"{'Rank':<6} {'Word':<20} {'Importance':<10}")
        print("-" * 40)
        for rank, (word, importance) in enumerate(analysis['overall_top_words'][:10], 1):
            print(f"{rank:<6} {word:<20} {importance:.4f}")

    print()

    if analysis and 'overall_top_atoms' in analysis:
        print("⚛️  Top 10 Most Important Atoms (overall):")
        print(f"{'Rank':<6} {'Atom':<20} {'Importance':<10}")
        print("-" * 40)
        for rank, (atom, importance) in enumerate(analysis['overall_top_atoms'][:10], 1):
            print(f"{rank:<6} {atom:<20} {importance:.4f}")

    print()

    # Print some specific atom-word pairs
    if analysis and 'atom_top_words' in analysis:
        print("🔍 Top Words for Each Atom:")
        print("-" * 60)
        for atom_id, top_words in list(analysis['atom_top_words'].items())[:5]:
            print(f"\n{atom_id}:")
            for word, importance in top_words[:5]:
                print(f"  - {word:<20} {importance:.4f}")

    print(f"\n{'='*80}\n")

    # Advanced analysis 1: Attention head specialization
    print("\n📊 Running advanced interpretability analysis...")
    print("   [1/4] Analyzing attention head specialization...")
    head_analysis = analyzer.analyze_attention_head_specialization(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text_tokens=tokens,
        save_path=save_dir / 'head_specialization.png' if save_dir else 'head_specialization.png'
    )

    if head_analysis:
        print(f"\n🎯 Attention Head Specialization:")
        print(f"   - Head diversity score: {head_analysis['head_diversity']:.3f}")
        print(f"   - Most important head: {max(head_analysis['head_importance'].items(), key=lambda x: x[1])[0]}")

        # Show top 3 most specialized heads
        print(f"\n   Top 3 Most Focused Heads:")
        sorted_heads = sorted(head_analysis['head_entropy'].items(), key=lambda x: x[1])[:3]
        for head_name, entropy in sorted_heads:
            pattern = head_analysis['head_patterns'][head_name]
            top_word = pattern['top_words'][0][0] if pattern['top_words'] else 'N/A'
            print(f"   - {head_name}: Entropy={entropy:.2f}, Top word='{top_word}'")

    # Advanced analysis 2: Key atom-word pairs
    print("\n   [2/4] Analyzing key atom-word pairs...")
    pairs_analysis = analyzer.analyze_key_atom_word_pairs(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text_tokens=tokens,
        top_k=20,
        save_path=save_dir / 'key_atom_word_pairs.png' if save_dir else 'key_atom_word_pairs.png'
    )

    if pairs_analysis:
        print(f"\n🔗 Top 10 Atom-Word Pairs:")
        for i, pair in enumerate(pairs_analysis['top_pairs'][:10], 1):
            print(f"   {i}. {pair['atom']} ←→ '{pair['word']}' (weight: {pair['weight']:.4f})")

        if pairs_analysis.get('category_stats'):
            print(f"\n   Semantic Category Distribution:")
            for cat, stats in pairs_analysis['category_stats'].items():
                cat_name = cat.replace('_', ' ').title()
                print(f"   - {cat_name}: {stats['count']} pairs ({stats['percentage']:.1f}%)")

    # Advanced analysis 3: Attention statistics
    print("\n   [3/4] Computing attention distribution statistics...")
    stats_analysis = analyzer.analyze_attention_statistics(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text_tokens=tokens,
        save_path=save_dir / 'attention_statistics.png' if save_dir else 'attention_statistics.png'
    )

    if stats_analysis:
        global_stats = stats_analysis['global_stats']
        print(f"\n📈 Attention Distribution Statistics:")
        print(f"   - Mean attention: {global_stats['mean']:.4f}")
        print(f"   - Entropy: {global_stats['entropy']:.2f}")
        print(f"   - Sparsity: {global_stats['sparsity']:.1f}%")
        print(f"   - Effective connections: {global_stats['effective_connections']:.1f}%")

        # Show most focused atoms
        print(f"\n   Most Focused Atoms (low entropy):")
        sorted_atoms = sorted(stats_analysis['per_atom_stats'].items(),
                            key=lambda x: x[1]['entropy'])[:3]
        for atom_name, atom_stats in sorted_atoms:
            print(f"   - {atom_name}: Entropy={atom_stats['entropy']:.2f}, Top word='{atom_stats['top_word']}'")

    # Advanced analysis 4: Text semantic regions
    print("\n   [4/4] Analyzing text semantic regions...")
    region_analysis = analyzer.analyze_text_semantic_regions(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text=text,
        text_tokens=tokens,
        save_path=save_dir / 'text_semantic_regions.png' if save_dir else 'text_semantic_regions.png'
    )

    if region_analysis and region_analysis.get('regions'):
        print(f"\n📝 Text Semantic Regions (Top 5 most important):")
        for i, region in enumerate(region_analysis['regions'][:5], 1):
            print(f"\n   Region {i} (ID: {region['region_id'] + 1}):")
            print(f"   Text: \"{region['text']}\"")
            print(f"   Importance: {region['avg_importance']:.4f} ({region['contribution']} contribution)")
            print(f"   Tokens: {region['num_tokens']}")

    print(f"\n{'='*80}")
    print(f"✅ All analyses complete! Results saved to: {save_dir}")
    print(f"\nGenerated visualizations:")
    if save_dir:
        print(f"   1. {save_dir / 'fine_grained_attention.png'} - Basic attention heatmaps")
        print(f"   2. {save_dir / 'head_specialization.png'} - Attention head analysis")
        print(f"   3. {save_dir / 'key_atom_word_pairs.png'} - Top atom-word pairs")
        print(f"   4. {save_dir / 'attention_statistics.png'} - Statistical analysis")
        print(f"   5. {save_dir / 'text_semantic_regions.png'} - Text region importance")
    print(f"{'='*80}\n")

    return {
        'basic_analysis': analysis,
        'head_analysis': head_analysis,
        'pairs_analysis': pairs_analysis,
        'stats_analysis': stats_analysis,
        'region_analysis': region_analysis
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Attention Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--cif_path', type=str, required=True,
                        help='Path to CIF file')
    parser.add_argument('--text', type=str, required=True,
                        help='Text description of the material')
    parser.add_argument('--save_dir', type=str, default='./fine_grained_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("🔬 Fine-Grained Cross-Modal Attention Analysis")
    print(f"{'='*80}\n")

    # Load model
    print("📦 Loading model...")
    model, config = load_model_with_fine_grained_attention(
        args.model_path,
        device=args.device
    )

    # Load structure
    print(f"\n📂 Loading structure from: {args.cif_path}")
    g, lg, atoms_object = cif_to_graph(args.cif_path)
    print(f"   - Number of atoms: {atoms_object.num_atoms}")
    print(f"   - Formula: {atoms_object.composition.reduced_formula}")

    # Analyze
    print(f"\n🔍 Analyzing with text:")
    print(f'   "{args.text[:100]}..."')

    analysis = analyze_with_fine_grained_attention(
        model=model,
        g=g,
        lg=lg,
        text=args.text,
        atoms_object=atoms_object,
        save_dir=args.save_dir
    )

    print(f"✅ Analysis complete! Results saved to: {args.save_dir}\n")


if __name__ == '__main__':
    main()
