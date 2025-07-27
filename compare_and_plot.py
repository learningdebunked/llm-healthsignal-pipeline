#!/usr/bin/env python3
"""
Healthcare Signal Analysis - Comparison and Plotting Tool
Compares different model configurations and visualizes performance metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean the ablation results data"""
    df = pd.read_csv('ablation_results.csv')
    
    # Clean the data
    df['Config'] = df['Config'].str.strip()
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    df['F1 Score'] = pd.to_numeric(df['F1 Score'], errors='coerce')
    df['LLM Helpfulness'] = pd.to_numeric(df['LLM Helpfulness'], errors='coerce')
    
    return df

def create_performance_comparison(df):
    """Create performance comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Healthcare AI Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy Comparison
    axes[0, 0].bar(df['Config'], df['Accuracy'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(df['Accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. F1 Score Comparison
    axes[0, 1].bar(df['Config'], df['F1 Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[0, 1].set_title('F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(df['F1 Score']):
        axes[0, 1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. LLM Helpfulness (only for configs with data)
    llm_data = df.dropna(subset=['LLM Helpfulness'])
    axes[1, 0].bar(llm_data['Config'], llm_data['LLM Helpfulness'], color=['#45B7D1', '#96CEB4'])
    axes[1, 0].set_title('LLM Helpfulness Rating')
    axes[1, 0].set_ylabel('Helpfulness Score (1-5)')
    axes[1, 0].set_ylim(0, 5)
    for i, v in enumerate(llm_data['LLM Helpfulness']):
        axes[1, 0].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Combined Performance Radar Chart
    metrics = ['Accuracy', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    axes[1, 1].remove()  # Remove the subplot
    ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (_, row) in enumerate(df.iterrows()):
        values = [row['Accuracy'], row['F1 Score']]
        values += values[:1]  # Complete the circle
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=row['Config'], color=colors[i])
        ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Performance Radar Chart')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_analysis(df):
    """Create detailed analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance Improvement Analysis
    baseline_acc = df.iloc[0]['Accuracy']  # LSTM baseline
    baseline_f1 = df.iloc[0]['F1 Score']
    
    acc_improvement = ((df['Accuracy'] - baseline_acc) / baseline_acc * 100).round(1)
    f1_improvement = ((df['F1 Score'] - baseline_f1) / baseline_f1 * 100).round(1)
    
    x = np.arange(len(df))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, acc_improvement, width, label='Accuracy Improvement', color='#FF6B6B')
    axes[0, 0].bar(x + width/2, f1_improvement, width, label='F1 Score Improvement', color='#4ECDC4')
    axes[0, 0].set_title('Performance Improvement vs Baseline (LSTM)')
    axes[0, 0].set_ylabel('Improvement (%)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df['Config'])
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 2. Model Complexity vs Performance
    complexity_scores = [1, 2, 4, 3]  # Relative complexity scores
    axes[0, 1].scatter(complexity_scores, df['Accuracy'], s=100, color='#45B7D1', label='Accuracy')
    axes[0, 1].scatter(complexity_scores, df['F1 Score'], s=100, color='#96CEB4', label='F1 Score')
    
    for i, config in enumerate(df['Config']):
        axes[0, 1].annotate(config, (complexity_scores[i], df.iloc[i]['Accuracy']), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[0, 1].set_title('Model Complexity vs Performance')
    axes[0, 1].set_xlabel('Relative Complexity')
    axes[0, 1].set_ylabel('Performance Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Performance Distribution
    metrics_data = []
    for _, row in df.iterrows():
        metrics_data.extend([
            {'Config': row['Config'], 'Metric': 'Accuracy', 'Value': row['Accuracy']},
            {'Config': row['Config'], 'Metric': 'F1 Score', 'Value': row['F1 Score']}
        ])
    
    metrics_df = pd.DataFrame(metrics_data)
    sns.boxplot(data=metrics_df, x='Metric', y='Value', ax=axes[1, 0])
    axes[1, 0].set_title('Performance Distribution Across Models')
    axes[1, 0].set_ylabel('Score')
    
    # 4. Model Ranking
    df['Combined_Score'] = (df['Accuracy'] + df['F1 Score']) / 2
    df_sorted = df.sort_values('Combined_Score', ascending=True)
    
    axes[1, 1].barh(df_sorted['Config'], df_sorted['Combined_Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 1].set_title('Overall Model Ranking')
    axes[1, 1].set_xlabel('Combined Performance Score')
    
    for i, v in enumerate(df_sorted['Combined_Score']):
        axes[1, 1].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detailed_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """Generate a summary report"""
    print("="*60)
    print("🩺 HEALTHCARE AI MODEL PERFORMANCE REPORT")
    print("="*60)
    
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 40)
    for _, row in df.iterrows():
        print(f"{row['Config']:<20} | Acc: {row['Accuracy']:.3f} | F1: {row['F1 Score']:.3f}")
    
    print("\n🏆 BEST PERFORMING MODELS:")
    print("-" * 40)
    best_acc = df.loc[df['Accuracy'].idxmax()]
    best_f1 = df.loc[df['F1 Score'].idxmax()]
    
    print(f"Highest Accuracy:    {best_acc['Config']} ({best_acc['Accuracy']:.3f})")
    print(f"Highest F1 Score:    {best_f1['Config']} ({best_f1['F1 Score']:.3f})")
    
    print("\n📈 IMPROVEMENT ANALYSIS:")
    print("-" * 40)
    baseline = df.iloc[0]  # LSTM baseline
    for _, row in df.iterrows():
        if row['Config'] != baseline['Config']:
            acc_imp = ((row['Accuracy'] - baseline['Accuracy']) / baseline['Accuracy'] * 100)
            f1_imp = ((row['F1 Score'] - baseline['F1 Score']) / baseline['F1 Score'] * 100)
            print(f"{row['Config']:<20} | +{acc_imp:+.1f}% Acc | +{f1_imp:+.1f}% F1")
    
    print("\n🤖 LLM INTEGRATION ANALYSIS:")
    print("-" * 40)
    llm_configs = df.dropna(subset=['LLM Helpfulness'])
    if not llm_configs.empty:
        for _, row in llm_configs.iterrows():
            print(f"{row['Config']:<20} | Helpfulness: {row['LLM Helpfulness']:.1f}/5.0")
    
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    best_overall = df.loc[df['Accuracy'].idxmax()]
    print(f"• Best Overall Model: {best_overall['Config']}")
    print(f"• Recommended for Production: {best_overall['Config']}")
    print(f"• Performance Gain: +{((best_overall['Accuracy'] - df.iloc[0]['Accuracy']) / df.iloc[0]['Accuracy'] * 100):.1f}% vs baseline")
    
    print("\n" + "="*60)

def main():
    """Main execution function"""
    print("🚀 Starting Healthcare AI Model Comparison and Analysis...")
    
    # Load data
    df = load_and_clean_data()
    print(f"✅ Loaded data for {len(df)} model configurations")
    
    # Generate summary report
    generate_summary_report(df)
    
    # Create visualizations
    print("\n📊 Generating performance comparison plots...")
    create_performance_comparison(df)
    
    print("📈 Generating detailed analysis plots...")
    create_detailed_analysis(df)
    
    print("\n✅ Analysis complete! Check the generated PNG files:")
    print("   - model_performance_comparison.png")
    print("   - detailed_model_analysis.png")

if __name__ == "__main__":
    main()