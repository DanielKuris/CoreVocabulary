import os
import sys
import io
import pandas as pd

# Configure stdout and stderr to use UTF-8 to prevent UnicodeEncodeError in Windows consoles
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

def main():
    # Resolve CSV path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "MASTER_BENCHMARK_MATRIX.csv")

    if not os.path.exists(csv_path):
        print(f"Error: Could not find benchmark matrix at {os.path.abspath(csv_path)}")
        return

    df = pd.read_csv(csv_path)

    # Clean whitespace
    for col in ['Dataset', 'Vocabulary', 'Exempt_Mode', 'Model_ID']:
        df[col] = df[col].astype(str).str.strip()

    print("==================================================================")
    print("LEXICAL-CONSTRAINED TEXT SIMPLIFICATION ANALYSIS REPORT")
    print("==================================================================")


    # 1. Global Model Performance Comparison
    print("\n1. GLOBAL PERFORMANCE BY MODEL (Averaged across all runs)")
    print("-" * 66)
    model_summary = df.groupby('Model_ID')[['SARI_Avg', 'BLEU_Avg', 'METEOR_Avg', 'BERTScore_Avg', 'Cosine_Similarity_Avg']].mean()
    print(model_summary.round(2).to_string())

    # 2. Custom Vocabulary Quality Comparison (Curated vs. Simple Frequency-based)
    print("\n2. CUSTOM VS. FREQUENCY-BASED VOCABULARIES (Model: MLM, Mode: english_stopwords)")
    print("-" * 66)
    target_vocabs = ['ThingExplainer1000.txt', 'vocab_1000.txt', 'OgdenBasicEnglish850.txt', 'vocab_800.txt', 'vocab_900.txt']
    
    for dataset in ['asset', 'turk_corpus']:
        print(f"\nDataset: {dataset.upper()}")
        subset = df[(df['Dataset'] == dataset) & (df['Model_ID'] == 'MLM') & (df['Exempt_Mode'] == 'english_stopwords')]
        res = subset[subset['Vocabulary'].isin(target_vocabs)].copy()
        # Sort manually to show custom right next to its baseline counterparts
        res['sort_order'] = res['Vocabulary'].map({
            'OgdenBasicEnglish850.txt': 1,
            'vocab_800.txt': 2,
            'vocab_900.txt': 3,
            'ThingExplainer1000.txt': 4,
            'vocab_1000.txt': 5
        })
        res = res.sort_values(by='sort_order')
        print(res[['Vocabulary', 'SARI_Avg', 'BLEU_Avg', 'BERTScore_Avg', 'Cosine_Similarity_Avg']].to_string(index=False))

    # 3. Peak Performance (Top combinations per dataset by SARI)
    print("\n3. TOP COMBINATION PER DATASET (Ranked by SARI)")
    print("-" * 66)

    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        top_sari = subset.sort_values(by='SARI_Avg', ascending=False).iloc[0]
        print(f"Dataset: {dataset.upper()}")
        print(f"  • Model:       {top_sari['Model_ID']}")
        print(f"  • Vocabulary:  {top_sari['Vocabulary']}")
        print(f"  • Exempt Mode: {top_sari['Exempt_Mode']}")
        print(f"  • SARI (Avg):  {top_sari['SARI_Avg']:.2f}")
        print(f"  • BERTScore:   {top_sari['BERTScore_Avg']:.2f}%")
        print(f"  • Cosine Sim:  {top_sari['Cosine_Similarity_Avg']:.2f}%")
        print()

if __name__ == "__main__":
    main()
