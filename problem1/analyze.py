"""
Analysis and visualization of attention patterns.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from model import Seq2SeqTransformer
from dataset import create_dataloaders, get_vocab_size
from attention import create_causal_mask


def extract_attention_weights(model, dataloader, device, num_samples=100):
    """
    Extract attention weights from model for analysis.

    Args:
        model: Trained transformer model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to analyze

    Returns:
        Dictionary containing attention weights and sample data
    """
    model.eval()

    all_encoder_attentions = []
    all_decoder_self_attentions = []
    all_decoder_cross_attentions = []
    all_inputs = []
    all_targets = []

    samples_collected = 0

    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            batch_size = inputs.size(0)

            # Modify model forward pass to return attention weights
            # Here we use hooks to capture attention weights during forward pass

            encoder_attentions = []
            decoder_self_attentions = []
            decoder_cross_attentions = []

            # Register hooks to capture attention weights
            def make_hook(attention_list):
                def hook(module, input, output):
                    # output is (attention_output, attention_weights)
                    attn_weights = output[1].detach().cpu()
                    # remove batch dimension for visualization (take first example)
                    if attn_weights.ndim == 4:
                        attn_weights = attn_weights[0]
                    attention_list.append(attn_weights)
                return hook

            # Register hooks on attention layers
            encoder_hooks = [
                layer.self_attn.register_forward_hook(make_hook(encoder_attentions))
                for layer in model.encoder_layers
            ]
            decoder_hooks = [
                layer.self_attn.register_forward_hook(make_hook(decoder_self_attentions))
                for layer in model.decoder_layers
            ]
            cross_hooks = [
                layer.cross_attn.register_forward_hook(make_hook(decoder_cross_attentions))
                for layer in model.decoder_layers
            ]

            # Forward pass
            _ = model(inputs, targets)

            # Remove hooks
            for h in encoder_hooks + decoder_hooks + cross_hooks:
                h.remove()

            # Collect samples
            samples_to_take = min(batch_size, num_samples - samples_collected)
            all_inputs.extend(inputs[:samples_to_take].cpu().numpy())
            all_targets.extend(targets[:samples_to_take].cpu().numpy())

            # Collect attention weights from hooks
            if encoder_attentions:
                all_encoder_attentions.append(encoder_attentions)
            if decoder_self_attentions:
                all_decoder_self_attentions.append(decoder_self_attentions)
            if decoder_cross_attentions:
                all_decoder_cross_attentions.append(decoder_cross_attentions)

            samples_collected += samples_to_take

    return {
        'encoder_attention': all_encoder_attentions,
        'decoder_self_attention': all_decoder_self_attentions,
        'decoder_cross_attention': all_decoder_cross_attentions,
        'inputs': all_inputs,
        'targets': all_targets
    }



def visualize_attention_pattern(attention_weights, input_tokens, output_tokens,
                               title="Attention Pattern", save_path=None):
    """
    Visualize attention weights as heatmap.

    Args:
        attention_weights: Attention weights [num_heads, out_len, in_len]
        input_tokens: Input token labels
        output_tokens: Output token labels
        title: Plot title
        save_path: Path to save figure
    """
    num_heads = attention_weights.shape[0]

    # Create figure with subplots for each head
    fig, axes = plt.subplots(
        2, (num_heads + 1) // 2,
        figsize=(5 * ((num_heads + 1) // 2), 8)
    )
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Plot heatmap
        sns.heatmap(
            attention_weights[head_idx],
            ax=ax,
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Head {head_idx + 1}')
        ax.set_xlabel('Input Position')
        ax.set_ylabel('Output Position')

    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_head_specialization(attention_data, output_dir):
    """
    Analyze what each attention head specializes in.

    Args:
        attention_data: Dictionary with attention weights and samples
        output_dir: Directory to save analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing encoder self-attention patterns...")

    head_stats = {}
    encoder_attentions = attention_data['encoder_attention']
    inputs = attention_data['inputs']

    operator_token = 10  # '+' symbol in the dataset

    for layer_idx, layer_attn_list in enumerate(encoder_attentions):
        if len(layer_attn_list) == 0:
            continue

        # Each tensor in layer_attn_list: [num_heads, seq_len_q, seq_len_k]
        attn = layer_attn_list[0]
        num_heads = attn.size(0)
        seq_len_q = attn.size(-2)
        seq_len_k = attn.size(-1)

        for head_idx in range(num_heads):
            weights = attn[head_idx]  # [seq_len_q, seq_len_k]

            # Average attention to operator token (‘+’)
            operator_mask = torch.zeros_like(weights)
            for seq in inputs:
                if operator_token in seq:
                    plus_idx = list(seq).index(operator_token)
                    if plus_idx < seq_len_k:
                        operator_mask[:, plus_idx] = 1
            operator_attention = (weights * operator_mask).sum() / operator_mask.sum().clamp(min=1)

            # Average attention to same position (diagonal) 
            eye = torch.eye(min(seq_len_q, seq_len_k), device=weights.device)
            diag_mask = torch.zeros_like(weights)
            diag_mask[:eye.size(0), :eye.size(1)] = eye
            diagonal_attention = (weights * diag_mask).sum() / diag_mask.sum().clamp(min=1)

            # Average attention to carry positions (next digit to the right)
            carry_mask = torch.zeros_like(weights)
            q_len, k_len = weights.size(0), weights.size(1) 
            for i in range(q_len - 1):  
                if i + 1 < k_len:     
                    carry_mask[i, i + 1] = 1
            carry_attention = (weights * carry_mask).sum() / carry_mask.sum().clamp(min=1)

            # Entropy of attention distribution
            probs = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean()

            # Store statistics for this head
            head_stats[f'layer{layer_idx}_head{head_idx}'] = {
                'operator_attention': float(operator_attention.item()),
                'diagonal_attention': float(diagonal_attention.item()),
                'carry_attention': float(carry_attention.item()),
                'entropy': float(entropy.item())
            }

    # Save results
    with open(output_dir / 'head_analysis.json', 'w') as f:
        json.dump(head_stats, f, indent=2)

    print(f"Head analysis complete. Saved to {output_dir}/head_analysis.json")
    return head_stats




def ablation_study(model, dataloader, device, output_dir):
    """
    Perform head ablation study.

    Test model performance when individual heads are disabled.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run on
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running head ablation study...")

    # Get baseline accuracy
    baseline_acc = evaluate_model(model, dataloader, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    ablation_results = {'baseline': baseline_acc}

    for layer_idx, layer in enumerate(model.encoder_layers):
        num_heads = layer.self_attn.num_heads
        d_k = layer.self_attn.d_k

        # Backup original output projection weights
        original_weight = layer.self_attn.W_O.weight.data.clone()

        for head_idx in range(num_heads):
            start = head_idx * d_k
            end = (head_idx + 1) * d_k

            # Zero out this head's contribution
            layer.self_attn.W_O.weight.data[:, start:end] = 0

            # Evaluate accuracy after ablation
            acc = evaluate_model(model, dataloader, device)

            # Record result
            ablation_results[f'layer{layer_idx}_head{head_idx}'] = acc

            print(f"Layer {layer_idx} Head {head_idx}: {acc:.2%} ({(baseline_acc - acc):.2%})")

            # Restore weights
            layer.self_attn.W_O.weight.data.copy_(original_weight)

    # Save ablation results
    with open(output_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # Create visualization of head importance
    plot_head_importance(ablation_results, output_dir / 'head_importance.png')

    return ablation_results


def evaluate_model(model, dataloader, device):
    """
    Evaluate model accuracy.

    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device to run on

    Returns:
        Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            predictions = model.generate(inputs, max_len=targets.size(1))
            
            # Compare sequence-level equality (entire sequence must match)
            match = (predictions.cpu() == targets.cpu()).all(dim=1)

            correct += match.sum().item()
            total += targets.size(0)

    return correct / total


def plot_head_importance(ablation_results, save_path):
    """
    Visualize head importance from ablation study.

    Args:
        ablation_results: Dictionary of ablation results
        save_path: Path to save figure
    """
    # Extract performance drops for each head
    baseline = ablation_results['baseline']

    head_names = []
    drops = []
    for key, acc in ablation_results.items():
        if key == 'baseline':
            continue
        head_names.append(key)
        drops.append(baseline - acc)

    plt.figure(figsize=(12, 6))

    plt.bar(head_names, drops, color='steelblue')

    plt.xlabel('Head')
    plt.ylabel('Accuracy Drop')
    plt.title('Head Importance (Accuracy Drop When Removed)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_example_predictions(model, dataloader, device, output_dir, num_examples=5):
    """
    Visualize model predictions on example inputs.

    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
    """
    output_dir = Path(output_dir)
    (output_dir / 'examples').mkdir(parents=True, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_examples:
                break

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # Take first sample from batch
            input_seq = inputs[0:1]
            target_seq = targets[0]

            # Generate prediction
            prediction = model.generate(input_seq, max_len=target_seq.size(0))

            # Convert to strings for visualization
            input_str = ' '.join(map(str, input_seq[0].cpu().numpy()))
            target_str = ''.join(map(str, target_seq.cpu().numpy()))
            pred_str = ''.join(map(str, prediction[0].cpu().numpy()))

            print(f"\nExample {batch_idx + 1}:")
            print(f"  Input:  {input_str}")
            print(f"  Target: {target_str}")
            print(f"  Pred:   {pred_str}")
            print(f"  Correct: {target_str == pred_str}")

            # Save attention heatmaps to output_dir / 'examples' / f'example_{batch_idx}.png'
            encoder_attn, decoder_attn = [], []

            # Register hooks to collect encoder and decoder self-attention
            hooks = []

            def make_hook(attn_list):
                def hook(module, input, output):
                    attn_list.append(output[1].detach().cpu())
                return hook

            for layer in model.encoder_layers:
                hooks.append(layer.self_attn.register_forward_hook(make_hook(encoder_attn)))
            for layer in model.decoder_layers:
                hooks.append(layer.self_attn.register_forward_hook(make_hook(decoder_attn)))

            # Run a forward pass to collect attention maps
            decoder_input = target_seq.unsqueeze(0)[:, :-1].to(device)
            tgt_mask = create_causal_mask(decoder_input.size(1), device=device)
            _ = model(input_seq, decoder_input, tgt_mask=tgt_mask)

            # Remove hooks
            for h in hooks:
                h.remove()

            # Visualize first layer, first head as an example
            if len(encoder_attn) > 0:
                attn_map = encoder_attn[0][0]  # [num_heads, seq_len, seq_len]
                from matplotlib import pyplot as plt
                import seaborn as sns

                plt.figure(figsize=(6, 5))
                sns.heatmap(
                    attn_map[0],
                    cmap='Blues',
                    xticklabels=list(map(str, input_seq[0].cpu().numpy())),
                    yticklabels=list(map(str, input_seq[0].cpu().numpy())),
                    cbar=True,
                    square=True
                )
                plt.title(f'Example {batch_idx + 1} Encoder Head 1 Attention')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')

                save_path = output_dir / 'examples' / f'example_{batch_idx + 1}.png'
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                
def visualize_head_analysis(head_analysis_path, save_path):
    """
    Visualize head statistics from head_analysis.json.
    """
    with open(head_analysis_path, 'r') as f:
        data = json.load(f)

    heads = list(data.keys())
    operator_scores = [data[h]['operator_attention'] for h in heads]
    diagonal_scores = [data[h]['diagonal_attention'] for h in heads]
    carry_scores = [data[h]['carry_attention'] for h in heads]
    entropy_scores = [data[h]['entropy'] for h in heads]

    plt.figure(figsize=(12, 6))
    x = range(len(heads))
    plt.bar(x, carry_scores, label='Carry Attention', alpha=0.8)
    plt.bar(x, diagonal_scores, bottom=carry_scores, label='Diagonal Attention', alpha=0.6)
    plt.bar(x, operator_scores, bottom=np.array(carry_scores)+np.array(diagonal_scores), label='Operator Attention', alpha=0.6)
    plt.plot(x, entropy_scores, 'r--o', label='Entropy')

    plt.xticks(x, heads, rotation=45)
    plt.ylabel('Attention Scores / Entropy')
    plt.title('Head Attention Characteristics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Head analysis visualization saved to {save_path}")



def main():
    parser = argparse.ArgumentParser(description='Analyze attention patterns')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    vocab_size = get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    ).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")

    # Load data
    _, _, test_loader = create_dataloaders(args.data_dir, args.batch_size)

    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'attention_patterns').mkdir(parents=True, exist_ok=True)
    (output_dir / 'head_analysis').mkdir(parents=True, exist_ok=True)

    # Extract attention weights
    print("Extracting attention weights...")
    attention_data = extract_attention_weights(
        model, test_loader, args.device, args.num_samples
    )

    # Visualize a few attention patterns 
    print("Visualizing attention patterns for sample heads...")
    pattern_dir = output_dir / 'attention_patterns'
    pattern_dir.mkdir(parents=True, exist_ok=True)

    num_to_visualize = min(4, len(attention_data['encoder_attention']))
    for layer_idx in range(num_to_visualize):
        if len(attention_data['encoder_attention'][layer_idx]) == 0:
            continue

        attn = attention_data['encoder_attention'][layer_idx][0]

        attn_np = attn.detach().cpu().float().numpy()
        
        if attn_np.ndim != 3:
            print(f"[Warning] Unexpected attention shape {attn_np.shape} — skipping layer {layer_idx}")
            continue

        num_heads, seq_len_out, seq_len_in = attn_np.shape
        input_tokens = [str(tok) for tok in attention_data['inputs'][0][:seq_len_in]]
        output_tokens = [f"T{i}" for i in range(seq_len_out)]

        visualize_attention_pattern(
            attn_np,
            input_tokens,
            output_tokens,
            title=f"Encoder Layer {layer_idx} Multi-Head Attention",
            save_path=pattern_dir / f"encoder_layer{layer_idx}_attention.png"
        )
        print(f"Saved attention visualization: {pattern_dir / f'encoder_layer{layer_idx}_attention.png'}")

    # Analyze head specialization
    head_stats = analyze_head_specialization(
        attention_data, output_dir / 'head_analysis'
    )

    # Run ablation study
    ablation_results = ablation_study(
        model, test_loader, args.device, output_dir / 'head_analysis'
    )

    # Visualize head analysis results
    visualize_head_analysis(
        output_dir / 'head_analysis' / 'head_analysis.json',
        output_dir / 'head_analysis' / 'head_analysis_plot.png'
    )

    # Visualize example predictions
    visualize_example_predictions(
        model, test_loader, args.device, output_dir, num_examples=5
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")



if __name__ == '__main__':
    main()