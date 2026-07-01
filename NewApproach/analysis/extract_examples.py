import os
import sys
import io
import pandas as pd

# Configure stdout and stderr to use UTF-8 to prevent UnicodeEncodeError in Windows consoles
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

def main():
    # Resolve samples comparison file path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_path = os.path.join(current_dir, "..", "samples", "turk_corpus_comparison.csv")

    if not os.path.exists(samples_path):
        print(f"Error: Could not find comparison samples at {os.path.abspath(samples_path)}")
        return

    df = pd.read_csv(samples_path)

    # Clean whitespace
    for col in ['Vocabulary', 'Exempt_Mode', 'Model_ID']:
        df[col] = df[col].astype(str).str.strip()

    # The 3 sentences we highlighted in the report
    highlight_indices = [52, 57, 327]

    print("==================================================================")
    print("SAMPLE SIMPLIFIED SENTENCE COMPARISONS")
    print("==================================================================")


    for idx in highlight_indices:
        sentence_data = df[df['Index'] == idx]
        if sentence_data.empty:
            continue
            
        original = sentence_data['Original_Sentence'].iloc[0]
        print(f"\n[Sentence ID: {idx}]")
        print(f"Original: {original}\n")

        # Define configurations to showcase
        configs = [
            {"model": "MLM", "vocab": "OgdenBasicEnglish850.txt", "exempt": "english_stopwords", "label": "MLM (Ogden 850, Stopwords)"},
            {"model": "MLM", "vocab": "ThingExplainer1000.txt", "exempt": "english_stopwords", "label": "MLM (ThingExplainer 1000, Stopwords)"},
            {"model": "EMB_SUB", "vocab": "ThingExplainer1000.txt", "exempt": "english_stopwords", "label": "EMB_SUB (ThingExplainer 1000)"},
            {"model": "T5", "vocab": "ThingExplainer1000.txt", "exempt": "english_stopwords", "label": "T5 (ThingExplainer 1000)"},
            {"model": "BYT5", "vocab": "ThingExplainer1000.txt", "exempt": "english_stopwords", "label": "BYT5 (ThingExplainer 1000)"}
        ]

        for config in configs:
            match = sentence_data[
                (sentence_data['Model_ID'] == config['model']) & 
                (sentence_data['Vocabulary'] == config['vocab']) & 
                (sentence_data['Exempt_Mode'] == config['exempt'])
            ]
            if not match.empty:
                transformed = match['Transformed_Sentence'].iloc[0]
                # Format long outputs cleanly
                if len(transformed) > 100:
                    transformed = transformed[:97] + "..."
                print(f"  • {config['label']:<40} -> \"{transformed}\"")
        print("-" * 66)

if __name__ == "__main__":
    main()
