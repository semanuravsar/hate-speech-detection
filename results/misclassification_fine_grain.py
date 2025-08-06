import torch
import pandas as pd
from transformers import AutoTokenizer
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # to access your model class if needed

from models.multitask_bert import MultiTaskBERT

# Configuration
MODEL_PATHS = {
    "aw_0.5": "results/aw_0.5.pt",
    "aw_1.0": "results/aw_1.0.pt", 
    "aw_2.0": "results/aw_2.0.pt",
    "baseline": "results/baseline.pt"
}

TASK = "main"  # for multi-task models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths - update these with your actual file paths
GENERAL_TEST_PATH = "datasets/latent_hatred_3class.csv"  # Update with your actual path
FINEGRAINED_TEST_PATH = "datasets/implicit_fine_labels.csv"  # Update with your actual path


def load_test_data():
    """Load test datasets"""
    print("📦 Loading test datasets...")
    
    # Load general test dataset (with implicit_hate, explicit_hate, not_hate labels)
    general_test_full = pd.read_csv(GENERAL_TEST_PATH)
    
    # Sample 15% of the general test dataset, stratified by label_id
    from sklearn.model_selection import train_test_split
    
    print(f"📊 Original dataset size: {len(general_test_full)}")
    print("Label distribution in original dataset:")
    print(general_test_full['label_id'].value_counts().sort_index())
    
    # Stratified sampling to get 15% of the data
    _, general_test = train_test_split(
        general_test_full, 
        test_size=0.15, 
        stratify=general_test_full['label_id'], 
        random_state=42
    )
    general_test = general_test.reset_index(drop=True)
    
    print(f"📊 Sampled dataset size: {len(general_test)} (15% of original)")
    print("Label distribution in sampled dataset:")
    print(general_test['label_id'].value_counts().sort_index())
    
    # Load fine-grained dataset (with implicit subcategories)
    finegrained_test = pd.read_csv(FINEGRAINED_TEST_PATH)
    
    return general_test, finegrained_test

def create_finegrain_mapping(finegrained_df):
    """Create mapping from label_id to category name and text to category"""
    # Create label_id to category mapping
    id_to_category = {}
    for _, row in finegrained_df.iterrows():
        id_to_category[row['label_id']] = row['label']
    
    # Create text to category mapping for direct lookup
    text_to_category = {}
    for _, row in finegrained_df.iterrows():
        clean_text = row['text'].strip()
        text_to_category[clean_text] = row['label']
    
    print("🏷️ Fine-grained categories found:")
    unique_categories = finegrained_df['label'].unique()
    for cat in unique_categories:
        print(f"  - {cat}")
    
    return id_to_category, text_to_category, unique_categories

def predict_batch(model, texts, tokenizer, task_name="main", batch_size=32):
    """Get predictions for a batch of texts"""
    model.eval()
    all_predictions = []
    
    print(f"🎯 Processing {len(texts)} texts in batches of {batch_size}...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            if task_name:  # Multi-task model
                logits = model(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"], 
                    task=task_name
                )
            
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
        
        if i % (batch_size * 10) == 0:
            print(f"  ✅ Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    return all_predictions

def analyze_model_misclassifications(model_name, model_path, general_test_df, text_to_finegrain, unique_categories):
    """Analyze misclassifications for a single model"""
    print(f"\n🧠 Loading model: {model_name}")
    
    # Load model
    model = MultiTaskBERT()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Get predictions
    predictions = predict_batch(
        model, 
        general_test_df['text'].tolist(), 
        tokenizer,
        task_name="main"
    )
    
    # Initialize counters
    implicit_to_nonhate_counts = Counter()
    total_implicit_counts = Counter()
    
    # Initialize all categories with 0 counts
    for category in unique_categories:
        implicit_to_nonhate_counts[category] = 0
        total_implicit_counts[category] = 0
    
    model_results = []
    
    # Analyze predictions
    for idx, row in general_test_df.iterrows():
        text = row['text'].strip()
        true_label = row['label_id']
        pred_label = predictions[idx]
        
        # If true label is implicit hate
        if true_label == 1:  # implicit_hate
            # Get fine-grained category
            finegrain_category = text_to_finegrain.get(text, 'unknown')
            
            if finegrain_category != 'unknown':
                total_implicit_counts[finegrain_category] += 1
                
                # If predicted as not_hate
                if pred_label == 0:  # not_hate
                    implicit_to_nonhate_counts[finegrain_category] += 1
                
                model_results.append({
                    'model': model_name,
                    'text': text,
                    'true_label': 'implicit_hate',
                    'pred_label': 'not_hate' if pred_label == 0 else ('implicit_hate' if pred_label == 1 else 'explicit_hate'),
                    'finegrain_category': finegrain_category,
                    'misclassified_as_nonhate': (pred_label == 0)
                })
    
    # Calculate confusion percentages
    confusion_percentages = {}
    for category in unique_categories:
        if total_implicit_counts[category] > 0:
            confusion_rate = (implicit_to_nonhate_counts[category] / 
                            total_implicit_counts[category]) * 100
            confusion_percentages[category] = confusion_rate
        else:
            confusion_percentages[category] = 0.0
    
    return {
        'model_name': model_name,
        'confusion_percentages': confusion_percentages,
        'implicit_to_nonhate_counts': implicit_to_nonhate_counts,
        'total_implicit_counts': total_implicit_counts,
        'detailed_results': model_results
    }

def print_model_results(model_results):
    """Print analysis results for a single model"""
    model_name = model_results['model_name']
    confusion_percentages = model_results['confusion_percentages']
    implicit_to_nonhate_counts = model_results['implicit_to_nonhate_counts']
    total_implicit_counts = model_results['total_implicit_counts']
    
    print(f"\n🔍 RESULTS FOR {model_name.upper()}")
    print("=" * 60)
    print(f"{'Category':<20} {'Confused %':<12} {'Count':<8} {'Total':<8}")
    print("-" * 60)
    
    # Sort by confusion rate (percentage)
    sorted_categories = sorted(confusion_percentages.items(), 
                             key=lambda x: x[1], reverse=True)
    
    for category, percentage in sorted_categories:
        confused_count = implicit_to_nonhate_counts[category]
        total_count = total_implicit_counts[category]
        print(f"{category:<20} {percentage:<12.1f} {confused_count:<8} {total_count:<8}")
    
    # Create summary statement
    print(f"\n📊 SUMMARY FOR {model_name}:")
    print("-" * 40)
    
    summary_parts = []
    for category, percentage in sorted_categories[:6]:  # Top 6 categories
        if percentage > 0:  # Only include categories with actual confusion
            summary_parts.append(f"{category.title()} ({percentage:.1f}%)")
    
    if summary_parts:
        summary = (f"For {model_name}, the implicit category most confused with non-hate was " + 
                  summary_parts[0])
        if len(summary_parts) > 1:
            if len(summary_parts) == 2:
                summary += " and " + summary_parts[1]
            else:
                summary += ", followed by " + ", ".join(summary_parts[1:-1])
                summary += ", and " + summary_parts[-1]
        summary += "."
        print(summary)
    else:
        print(f"No implicit hate categories were confused with non-hate for {model_name}.")


def save_combined_results(all_model_results):
    """Save all results to CSV files"""
    # Combine all detailed results
    all_detailed_results = []
    for result in all_model_results:
        all_detailed_results.extend(result['detailed_results'])
    
    # Save detailed results
    detailed_df = pd.DataFrame(all_detailed_results)
    detailed_df.to_csv('all_models_detailed_misclassifications.csv', index=False)
    print("💾 Detailed results saved to 'all_models_detailed_misclassifications.csv'")
    
    # Create summary table
    summary_data = []
    for result in all_model_results:
        model_name = result['model_name']
        confusion_percentages = result['confusion_percentages']
        
        for category, percentage in confusion_percentages.items():
            summary_data.append({
                'Model': model_name,
                'Category': category,
                'Confusion_Rate_Percent': percentage,
                'Misclassified_Count': result['implicit_to_nonhate_counts'][category],
                'Total_Count': result['total_implicit_counts'][category]
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('all_models_confusion_summary.csv', index=False)
    print("💾 Summary results saved to 'all_models_confusion_summary.csv'")
    
    return detailed_df, summary_df

def analyze_per_category_classification_rates(model_name, model_path, general_test_df, text_to_finegrain, unique_categories):
    """
    Analyze classification rates for each fine-grained category.
    For each category, calculate:
    - Percentage correctly classified as hate (implicit or explicit)
    - Percentage incorrectly classified as non-hate
    """
    print(f"\n🎯 Analyzing per-category classification rates for: {model_name}")
    
    # Load model
    model = MultiTaskBERT()
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Get predictions
    predictions = predict_batch(
        model, 
        general_test_df['text'].tolist(), 
        tokenizer,
        task_name="main"
    )
    
    # Initialize counters for each category
    category_stats = {}
    for category in unique_categories:
        category_stats[category] = {
            'total_samples': 0,
            'correctly_classified_as_hate': 0,  # predicted as implicit or explicit hate
            'incorrectly_classified_as_nonhate': 0,  # predicted as non-hate
            'correct_implicit': 0,  # correctly predicted as implicit hate
            'correct_explicit': 0,  # correctly predicted as explicit hate
            'misclassified_as_nonhate': 0,  # predicted as non-hate when should be hate
            'misclassified_as_wrong_hate': 0  # predicted as explicit when should be implicit, or vice versa
        }
    
    # Analyze predictions
    for idx, row in general_test_df.iterrows():
        text = row['text'].strip()
        true_label = row['label_id']  # 0: not_hate, 1: implicit_hate, 2: explicit_hate
        pred_label = predictions[idx]
        
        # Only analyze hate samples (implicit or explicit)
        if true_label in [1, 2]:  # implicit_hate or explicit_hate
            # Get fine-grained category
            finegrain_category = text_to_finegrain.get(text, 'unknown')
            
            if finegrain_category != 'unknown':
                stats = category_stats[finegrain_category]
                stats['total_samples'] += 1
                
                if pred_label == 0:  # Predicted as not_hate
                    stats['incorrectly_classified_as_nonhate'] += 1
                    stats['misclassified_as_nonhate'] += 1
                else:  # Predicted as some form of hate
                    stats['correctly_classified_as_hate'] += 1
                    
                    if pred_label == true_label:  # Exact match
                        if true_label == 1:  # implicit hate
                            stats['correct_implicit'] += 1
                        else:  # explicit hate
                            stats['correct_explicit'] += 1
                    else:  # Wrong type of hate
                        stats['misclassified_as_wrong_hate'] += 1
    
    # Calculate percentages
    category_results = {}
    for category, stats in category_stats.items():
        if stats['total_samples'] > 0:
            category_results[category] = {
                'total_samples': stats['total_samples'],
                'hate_classification_accuracy': (stats['correctly_classified_as_hate'] / stats['total_samples']) * 100,
                'nonhate_misclassification_rate': (stats['incorrectly_classified_as_nonhate'] / stats['total_samples']) * 100,
                'exact_classification_accuracy': ((stats['correct_implicit'] + stats['correct_explicit']) / stats['total_samples']) * 100,
                'wrong_hate_type_rate': (stats['misclassified_as_wrong_hate'] / stats['total_samples']) * 100,
                'raw_counts': stats
            }
        else:
            category_results[category] = {
                'total_samples': 0,
                'hate_classification_accuracy': 0.0,
                'nonhate_misclassification_rate': 0.0,
                'exact_classification_accuracy': 0.0,
                'wrong_hate_type_rate': 0.0,
                'raw_counts': stats
            }
    
    return {
        'model_name': model_name,
        'category_results': category_results
    }

def print_per_category_results(category_analysis):
    """Print per-category classification results"""
    model_name = category_analysis['model_name']
    category_results = category_analysis['category_results']
    
    print(f"\n📊 PER-CATEGORY CLASSIFICATION RATES FOR {model_name.upper()}")
    print("=" * 80)
    print(f"{'Category':<20} {'Samples':<8} {'Hate Acc%':<10} {'NonHate Err%':<12} {'Exact Acc%':<11}")
    print("-" * 80)
    
    # Sort by hate classification accuracy (descending)
    sorted_categories = sorted(category_results.items(), 
                             key=lambda x: x[1]['hate_classification_accuracy'], 
                             reverse=True)
    
    for category, results in sorted_categories:
        if results['total_samples'] > 0:
            print(f"{category:<20} {results['total_samples']:<8} "
                  f"{results['hate_classification_accuracy']:<10.1f} "
                  f"{results['nonhate_misclassification_rate']:<12.1f} "
                  f"{results['exact_classification_accuracy']:<11.1f}")
    
    print(f"\nColumn explanations:")
    print(f"- Hate Acc%: % correctly identified as hate (implicit OR explicit)")
    print(f"- NonHate Err%: % incorrectly classified as non-hate")
    print(f"- Exact Acc%: % with exact label match (implicit vs explicit)")

def analyze_all_models_per_category(general_test_df, text_to_finegrain, unique_categories):
    """Analyze per-category classification rates for all models"""
    print(f"\n🔬 ANALYZING PER-CATEGORY CLASSIFICATION RATES FOR ALL MODELS")
    print("=" * 80)
    
    all_category_analyses = []
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            category_analysis = analyze_per_category_classification_rates(
                model_name, model_path, general_test_df, text_to_finegrain, unique_categories
            )
            all_category_analyses.append(category_analysis)
            print_per_category_results(category_analysis)
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {str(e)}")
            continue
    
    return all_category_analyses

def save_per_category_results(all_category_analyses):
    """Save per-category analysis results to CSV"""
    # Combine all results into a comprehensive table
    combined_data = []
    
    for analysis in all_category_analyses:
        model_name = analysis['model_name']
        category_results = analysis['category_results']
        
        for category, results in category_results.items():
            if results['total_samples'] > 0:  # Only include categories with samples
                combined_data.append({
                    'Model': model_name,
                    'Category': category,
                    'Total_Samples': results['total_samples'],
                    'Hate_Classification_Accuracy_Percent': results['hate_classification_accuracy'],
                    'NonHate_Misclassification_Rate_Percent': results['nonhate_misclassification_rate'],
                    'Exact_Classification_Accuracy_Percent': results['exact_classification_accuracy'],
                    'Wrong_Hate_Type_Rate_Percent': results['wrong_hate_type_rate'],
                    'Correctly_Classified_As_Hate_Count': results['raw_counts']['correctly_classified_as_hate'],
                    'Misclassified_As_NonHate_Count': results['raw_counts']['misclassified_as_nonhate'],
                    'Correct_Implicit_Count': results['raw_counts']['correct_implicit'],
                    'Correct_Explicit_Count': results['raw_counts']['correct_explicit']
                })
    
    # Save to CSV
    per_category_df = pd.DataFrame(combined_data)
    per_category_df.to_csv('per_category_classification_rates.csv', index=False)
    print("💾 Per-category results saved to 'per_category_classification_rates.csv'")
    
    return per_category_df

# Modified main function to include the new analysis
def main_with_per_category_analysis():
    """Main execution function with additional per-category analysis"""
    print("🚀 Starting Enhanced 4-Model Implicit Hate Analysis")
    print("=" * 70)
    
    # Load datasets
    general_test, finegrained_test = load_test_data()
    
    # Create mappings
    id_to_category, text_to_finegrain, unique_categories = create_finegrain_mapping(finegrained_test)
    
    print(f"\n📊 Dataset Info:")
    print(f"  - General test samples: {len(general_test)}")
    print(f"  - Fine-grained samples: {len(finegrained_test)}")
    print(f"  - Unique fine-grained categories: {len(unique_categories)}")
    
    # Original analysis (misclassification distribution)
    print(f"\n" + "="*70)
    print("PART 1: ORIGINAL ANALYSIS - Misclassification Distribution")
    print("="*70)
    
    all_model_results = []
    for model_name, model_path in MODEL_PATHS.items():
        try:
            model_results = analyze_model_misclassifications(
                model_name, model_path, general_test, text_to_finegrain, unique_categories
            )
            all_model_results.append(model_results)
            print_model_results(model_results)
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {str(e)}")
            continue
    
    # New analysis (per-category classification rates)
    print(f"\n" + "="*70)
    print("PART 2: NEW ANALYSIS - Per-Category Classification Rates")
    print("="*70)
    
    all_category_analyses = analyze_all_models_per_category(
        general_test, text_to_finegrain, unique_categories
    )
    
    # Save all results
    if all_model_results and all_category_analyses:
        print(f"\n💾 Saving all results...")
        
        # Save original results
        detailed_df, summary_df = save_combined_results(all_model_results)
        
        # Save new per-category results
        per_category_df = save_per_category_results(all_category_analyses)
        
        print(f"\n✅ Enhanced analysis complete!")
        print(f"📁 Files created:")
        print(f"  - all_models_detailed_misclassifications.csv (original)")
        print(f"  - all_models_confusion_summary.csv (original)")
        print(f"  - per_category_classification_rates.csv (new)")
        
        return all_model_results, all_category_analyses, detailed_df, summary_df, per_category_df
    else:
        print("❌ Analysis failed.")
        return None

def main():
    """Main execution function"""
    print("🚀 Starting 4-Model Implicit Hate Misclassification Analysis")
    print("=" * 70)
    
    # Load datasets
    general_test, finegrained_test = load_test_data()
    
    # Create mappings
    id_to_category, text_to_finegrain, unique_categories = create_finegrain_mapping(finegrained_test)
    
    print(f"\n📊 Dataset Info:")
    print(f"  - General test samples: {len(general_test)}")
    print(f"  - Fine-grained samples: {len(finegrained_test)}")
    print(f"  - Unique fine-grained categories: {len(unique_categories)}")
    
    # Analyze each model
    all_model_results = []
    
    for model_name, model_path in MODEL_PATHS.items():
        try:
            model_results = analyze_model_misclassifications(
                model_name, model_path, general_test, text_to_finegrain, unique_categories
            )
            all_model_results.append(model_results)
            print_model_results(model_results)
        except Exception as e:
            print(f"❌ Error analyzing {model_name}: {str(e)}")
            continue
    
    if all_model_results:
        
        print(f"\n💾 Saving results...")
        detailed_df, summary_df = save_combined_results(all_model_results)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Files created:")
        print(f"  - all_models_detailed_misclassifications.csv")
        print(f"  - all_models_confusion_summary.csv") 
        print(f"  - model_comparison_heatmap.png")
        print(f"  - individual_model_confusion.png")
        
        return all_model_results, detailed_df, summary_df
    else:
        print("❌ No models were successfully analyzed.")
        return None, None, None

if __name__ == "__main__":
    # Update these paths with your actual data file locations
    GENERAL_TEST_PATH = "datasets/latent_hatred_3class.csv"  # UPDATE THIS
    FINEGRAINED_TEST_PATH = "datasets/implicit_fine_labels.csv"  # UPDATE THIS
    
    # Run the analysis
    all_results, detailed_results, summary_results = main_with_per_category_analysis()