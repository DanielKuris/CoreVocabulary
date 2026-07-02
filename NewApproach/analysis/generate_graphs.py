import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure stdout and stderr to use UTF-8 to prevent UnicodeEncodeError in Windows consoles
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

def main():
    # Resolve CSV path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "MASTER_BENCHMARK_MATRIX.csv")
    output_dir = os.path.join(current_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Error: Could not find benchmark matrix at {os.path.abspath(csv_path)}")
        return


    # Load dataset
    df = pd.read_csv(csv_path)

    # Clean whitespace
    for col in ['Dataset', 'Vocabulary', 'Exempt_Mode', 'Model_ID']:
        df[col] = df[col].astype(str).str.strip()

    # Extract vocabulary size from name (e.g. vocab_100.txt -> 100)
    def extract_vocab_size(name):
        try:
            if 'vocab_' in name:
                return int(name.split('_')[1].split('.')[0])
        except:
            pass
        return None

    df['Vocab_Size'] = df['Vocabulary'].apply(extract_vocab_size)

    # Filter out non-numeric vocabularies for plotting trends
    plot_df = df.dropna(subset=['Vocab_Size']).copy()
    plot_df = plot_df.sort_values(by='Vocab_Size')

    # Set aesthetic styling
    sns.set_theme(style="whitegrid", rc={"grid.color": ".9", "grid.linestyle": "--"})
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.titlesize': 16
    })

    # Graph 1: Model Scaling Benchmark
    # Compare MLM, EMB_SUB, T5, BYT5 SARI score on the ASSET dataset under english_stopwords
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    g1_df = plot_df[(plot_df['Dataset'] == 'asset') & (plot_df['Exempt_Mode'] == 'english_stopwords')]

    model_colors = {
        'MLM': '#1E88E5',      # Vivid Blue
        'EMB_SUB': '#004D40',  # Deep Teal
        'BYT5': '#FFC107',     # Warm Amber
        'T5': '#D81B60'        # Crimson Rose
    }

    for model in ['MLM', 'EMB_SUB', 'BYT5', 'T5']:
        model_data = g1_df[g1_df['Model_ID'] == model]
        if not model_data.empty:
            ax1.plot(
                model_data['Vocab_Size'],
                model_data['SARI_Avg'],
                marker='o',
                linewidth=2.5,
                markersize=5,
                color=model_colors[model],
                label=f"{model} (Avg SARI)"
            )

    ax1.set_title("Model Simplification Quality (SARI) vs. Vocabulary Size\n(Dataset: ASSET | Exemption Mode: english_stopwords)", pad=15)
    ax1.set_xlabel("Core Vocabulary Size (Allowed Word Count)")
    ax1.set_ylabel("SARI Score (Higher is Better)")
    ax1.set_ylim(15, 36)
    ax1.set_xticks(np.arange(100, 2001, 200))
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Graph 1 (Model scaling) to: {os.path.abspath(fig1_path)}")

    # Graph 1b: Model Simplification Quality (METEOR) vs. Vocabulary Size
    fig1b, ax1b = plt.subplots(figsize=(8, 5))
    for model in ['MLM', 'EMB_SUB', 'BYT5', 'T5']:
        model_data = g1_df[g1_df['Model_ID'] == model]
        if not model_data.empty:
            ax1b.plot(
                model_data['Vocab_Size'],
                model_data['METEOR_Avg'],
                marker='o',
                linewidth=2.5,
                markersize=5,
                color=model_colors[model],
                label=f"{model} (Avg METEOR)"
            )
    ax1b.set_title("Model Simplification Fluency (METEOR) vs. Vocabulary Size\n(Dataset: ASSET | Exemption Mode: english_stopwords)", pad=15)
    ax1b.set_xlabel("Core Vocabulary Size (Allowed Word Count)")
    ax1b.set_ylabel("METEOR Score (Higher is Better)")
    ax1b.set_ylim(0, 55)
    ax1b.set_xticks(np.arange(100, 2001, 200))
    ax1b.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    fig1b_path = os.path.join(output_dir, "model_comparison_meteor.png")
    plt.savefig(fig1b_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Graph 1b (Model scaling - METEOR) to: {os.path.abspath(fig1b_path)}")

    # Graph 1c: Model Meaning Preservation (Cosine Similarity) vs. Vocabulary Size
    fig1c, ax1c = plt.subplots(figsize=(8, 5))
    for model in ['MLM', 'EMB_SUB', 'BYT5', 'T5']:
        model_data = g1_df[g1_df['Model_ID'] == model]
        if not model_data.empty:
            ax1c.plot(
                model_data['Vocab_Size'],
                model_data['Cosine_Similarity_Avg'],
                marker='o',
                linewidth=2.5,
                markersize=5,
                color=model_colors[model],
                label=f"{model} (Avg Cosine Sim)"
            )
    ax1c.set_title("Model Meaning Preservation vs. Vocabulary Size", pad=15)
    ax1c.set_xlabel("Vocabulary Size")
    ax1c.set_ylabel("Cosine Similarity")
    ax1c.set_ylim(50, 95)
    ax1c.set_xticks(np.arange(100, 2001, 200))
    ax1c.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    fig1c_path = os.path.join(output_dir, "model_comparison_cosine.png")
    plt.savefig(fig1c_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Graph 1c (Model scaling - Cosine Sim) to: {os.path.abspath(fig1c_path)}")


    # Graph 2: The Stopword Exemption Impact
    # Compare english_stopwords vs none on BERTScore for MLM (the best model) on ASSET dataset
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    g2_df = plot_df[(plot_df['Dataset'] == 'asset') & (plot_df['Model_ID'] == 'MLM')]

    exemption_colors = {
        'english_stopwords': '#1E88E5', # Vivid Blue
        'none': '#D81B60'              # Crimson Rose
    }

    for exempt in ['english_stopwords', 'none']:
        exempt_data = g2_df[g2_df['Exempt_Mode'] == exempt]
        if not exempt_data.empty:
            ax2.plot(
                exempt_data['Vocab_Size'],
                exempt_data['BERTScore_Avg'],
                marker='s',
                linewidth=2.5,
                markersize=5,
                color=exemption_colors[exempt],
                label=f"MLM with {exempt} exempt" if exempt == 'english_stopwords' else "MLM with no exemptions"
            )

    ax2.set_title("Semantic Preservation (BERTScore) vs. Vocabulary Size\n(Model: MLM | Dataset: ASSET)", pad=15)
    ax2.set_xlabel("Core Vocabulary Size (Allowed Word Count)")
    ax2.set_ylabel("BERTScore F1 (Avg %)")
    ax2.set_ylim(60, 85)
    ax2.set_xticks(np.arange(100, 2001, 200))
    ax2.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "stopword_exemption.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Graph 2 (Stopword impact) to: {os.path.abspath(fig2_path)}")

    # Graph 3: Fluency vs. Meaning Preservation (Scatter Plot)
    # Compare architectures across meaning preservation (BERTScore) and fluency (METEOR)
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Calculate global averages per model
    summary = df.groupby('Model_ID')[['BERTScore_Avg', 'METEOR_Avg']].mean().reset_index()

    # Map model IDs to names for presentation
    model_mapping = {
        'EMB_SUB': 'Base Model (EMB_SUB)',
        'MLM': 'Masked Language Model (MLM)',
        'T5': 'Seq2Seq (T5)',
        'BYT5': 'Character-level (ByT5)'
    }
    summary['Model_Name'] = summary['Model_ID'].map(model_mapping)

    model_colors = {
        'EMB_SUB': '#004D40',  # Deep Teal
        'MLM': '#1E88E5',      # Vivid Blue
        'T5': '#D81B60',       # Crimson Rose
        'BYT5': '#FFC107'      # Warm Amber
    }

    model_markers = {
        'EMB_SUB': '^',        # Triangle
        'MLM': 'o',            # Circle
        'T5': 's',             # Square
        'BYT5': 'D'            # Diamond
    }

    for i, row in summary.iterrows():
        model = row['Model_ID']
        ax3.scatter(
            row['BERTScore_Avg'],
            row['METEOR_Avg'],
            color=model_colors[model],
            marker=model_markers[model],
            s=150,
            label=row['Model_Name'],
            edgecolor='black',
            linewidth=1.2,
            zorder=3
        )
        # Offset label slightly for readability
        ax3.text(
            row['BERTScore_Avg'] + 0.5,
            row['METEOR_Avg'] + 0.3,
            row['Model_ID'],
            fontsize=10,
            fontweight='bold',
            color='#333333'
        )

    ax3.set_title("Model Architecture Trade-offs", pad=15)
    ax3.set_xlabel("Meaning Preservation")
    ax3.set_ylabel("Grammatical Fluency")
    ax3.set_xlim(55, 83)
    ax3.set_ylim(0, 35)

    ax3.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "tradeoffs.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Graph 3 (Trade-offs scatter plot) to: {os.path.abspath(fig3_path)}")



if __name__ == "__main__":
    main()
