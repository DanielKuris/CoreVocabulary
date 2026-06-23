import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

def generate_benchmark_plots(csv_path="MASTER_BENCHMARK_MATRIX.csv", output_dir="plots"):
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found. Run your matrix first!")
        return
        
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
        
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].str.strip()
    df['Vocabulary'] = df['Vocabulary'].str.strip()
    df['Exempt_Mode'] = df['Exempt_Mode'].str.strip()
    df['Model_ID'] = df['Model_ID'].str.strip()
    
    def extract_vocab_size(name):
        try:
            if 'vocab_' in name:
                return int(name.split('_')[1].split('.')[0])
        except:
            pass
        return None

    df['Vocab_Size'] = df['Vocabulary'].apply(extract_vocab_size)
    plot_df = df.dropna(subset=['Vocab_Size']).copy()
    plot_df = plot_df.sort_values(by='Vocab_Size')
    
    datasets = plot_df['Dataset'].unique()
    models = plot_df['Model_ID'].unique()
    exempt_modes = plot_df['Exempt_Mode'].unique()
    
    accuracy_metrics = ["SARI", "BLEU", "METEOR", "BERTScore", "Jaccard_Similarity", "Cosine_Similarity"]
    
    for dataset in datasets:
        for model in models:
            for exempt in exempt_modes:
                subset = plot_df[
                    (plot_df['Dataset'] == dataset) & 
                    (plot_df['Model_ID'] == model) & 
                    (plot_df['Exempt_Mode'] == exempt)
                ]
                
                if subset.empty:
                    continue
                    
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [3, 1]})
                
                # Top Plot: Accuracy & Similarity Metrics
                for metric in accuracy_metrics:
                    sns.lineplot(
                        data=subset,
                        x="Vocab_Size",
                        y=metric,
                        marker="o",
                        linewidth=2.0,
                        markersize=6,
                        label=metric,
                        ax=ax1
                    )
                
                ax1.set_title(f"{dataset.upper()} | Model: {model} ({exempt.upper()} exemption)", pad=15)
                ax1.set_xlabel("")  # Shared with bottom plot
                ax1.set_ylabel("Score (0 - 100)")
                ax1.set_ylim(0, 100)
                ax1.set_yticks(range(0, 101, 10))
                ax1.set_xticks(subset['Vocab_Size'].unique())
                ax1.set_xticklabels([])  # Hide tick labels to avoid overlap
                ax1.legend(title="Evaluated Metrics", loc="upper left", bbox_to_anchor=(1.02, 1))
                
                # Bottom Plot: Compression Ratio
                sns.lineplot(
                    data=subset,
                    x="Vocab_Size",
                    y="Compression_Ratio",
                    marker="s",
                    color="#E63946",
                    linewidth=2.0,
                    markersize=6,
                    label="Compression Ratio",
                    ax=ax2
                )
                
                ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
                ax2.set_xlabel("Vocabulary Size")
                ax2.set_ylabel("Ratio")
                ax2.set_ylim(0, max(1.5, subset['Compression_Ratio'].max() * 1.2))
                ax2.set_xticks(subset['Vocab_Size'].unique())
                ax2.set_xticklabels(subset['Vocab_Size'].unique(), rotation=45)
                ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
                
                plt.tight_layout()
                
                safe_model_name = model.replace("/", "_").lower()
                filename = f"{output_dir}/{dataset}_{safe_model_name}_{exempt}_metrics.png"
                plt.savefig(filename, dpi=300)
                plt.close()
            
    print(f"🎉 Plot generation complete! All split vertical matrix views exported to '{output_dir}/'.")

if __name__ == "__main__":
    generate_benchmark_plots()