"""
Student Performance & Academic Analytics System
Visualization Module (visualization.py)

This module generates professional, polished academic charts using
Matplotlib and Seaborn. It provides separate functions for each chart
and a master function to batch generate all required analytical plots.
"""

import os
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set global aesthetic styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_dept_final_scores(df: pd.DataFrame, save_path: str) -> str:
    """
    1. Average final score by department — Bar Chart.
    """
    plt.figure(figsize=(9, 5.5))
    dept_avg = df.groupby('Department')['Final_Score'].mean().reset_index()
    dept_avg = dept_avg.sort_values(by='Final_Score', ascending=False)
    
    palette = sns.color_palette("crest", n_colors=len(dept_avg))
    ax = sns.barplot(
        data=dept_avg, 
        x='Department', 
        y='Final_Score',
        hue='Department',
        legend=False,
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}",
            (p.get_x() + p.get_width() / 2.0, height / 2.0),
            ha='center', va='center',
            fontsize=11, color='white', weight='bold'
        )
        
    plt.title("Average Final Exam Score by Department", weight='bold', pad=15)
    plt.xlabel("Academic Department", weight='semibold', labelpad=10)
    plt.ylabel("Average Final Score (0 - 100)", weight='semibold', labelpad=10)
    plt.ylim(0, 105)
    plt.axhline(df['Final_Score'].mean(), color='red', linestyle='--', label=f"Cohort Mean ({df['Final_Score'].mean():.1f})")
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_study_hours_vs_final_score(df: pd.DataFrame, save_path: str) -> str:
    """
    2. Study hours vs final score — Scatter Plot with linear regression trendline.
    """
    plt.figure(figsize=(9, 5.5))
    
    palette = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Average': '#f39c12', 'At Risk': '#e74c3c'}
    
    sns.scatterplot(
        data=df,
        x='Study_Hours',
        y='Final_Score',
        hue='Performance_Category',
        palette=palette,
        s=80,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add trendline
    sns.regplot(
        data=df,
        x='Study_Hours',
        y='Final_Score',
        scatter=False,
        color='#2c3e50',
        line_kws={'linestyle': '--', 'linewidth': 1.8, 'label': 'Trendline'}
    )
    
    corr = np.corrcoef(df['Study_Hours'], df['Final_Score'])[0, 1]
    
    plt.title(f"Study Hours vs Final Score (Correlation r = {corr:.2f})", weight='bold', pad=15)
    plt.xlabel("Weekly Study Hours (hrs/week)", weight='semibold', labelpad=10)
    plt.ylabel("Final Exam Score (0 - 100)", weight='semibold', labelpad=10)
    plt.legend(title="Performance Tier", frameon=True, loc='lower right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_attendance_vs_final_score(df: pd.DataFrame, save_path: str) -> str:
    """
    3. Attendance vs final score — Scatter Plot with regression trendline.
    """
    plt.figure(figsize=(9, 5.5))
    
    palette = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Average': '#f39c12', 'At Risk': '#e74c3c'}
    
    sns.scatterplot(
        data=df,
        x='Attendance',
        y='Final_Score',
        hue='Performance_Category',
        palette=palette,
        s=80,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5
    )
    
    sns.regplot(
        data=df,
        x='Attendance',
        y='Final_Score',
        scatter=False,
        color='#8e44ad',
        line_kws={'linestyle': '-.', 'linewidth': 1.8, 'label': 'Trendline'}
    )
    
    corr = np.corrcoef(df['Attendance'], df['Final_Score'])[0, 1]
    
    plt.title(f"Class Attendance Rate vs Final Score (Correlation r = {corr:.2f})", weight='bold', pad=15)
    plt.xlabel("Attendance Percentage (%)", weight='semibold', labelpad=10)
    plt.ylabel("Final Exam Score (0 - 100)", weight='semibold', labelpad=10)
    plt.legend(title="Performance Tier", frameon=True, loc='lower right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_final_score_distribution(df: pd.DataFrame, save_path: str) -> str:
    """
    4. Final score distribution — Histogram with KDE curve.
    """
    plt.figure(figsize=(9, 5.5))
    
    sns.histplot(
        df['Final_Score'],
        bins=15,
        kde=True,
        color='#3498db',
        edgecolor='black',
        alpha=0.6,
        line_kws={'linewidth': 2.2, 'color': '#1b4f72'}
    )
    
    mean_val = df['Final_Score'].mean()
    median_val = df['Final_Score'].median()
    
    plt.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1.8, label=f"Mean ({mean_val:.1f})")
    plt.axvline(median_val, color='#27ae60', linestyle=':', linewidth=2.0, label=f"Median ({median_val:.1f})")
    
    plt.title("Distribution of Student Final Exam Scores", weight='bold', pad=15)
    plt.xlabel("Final Exam Score", weight='semibold', labelpad=10)
    plt.ylabel("Student Count (Frequency)", weight='semibold', labelpad=10)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_category_distribution(df: pd.DataFrame, save_path: str) -> str:
    """
    5. Performance category distribution — Count Plot.
    """
    plt.figure(figsize=(8.5, 5.5))
    
    order = ['Excellent', 'Good', 'Average', 'At Risk']
    palette = {'Excellent': '#2ecc71', 'Good': '#3498db', 'Average': '#f39c12', 'At Risk': '#e74c3c'}
    
    ax = sns.countplot(
        data=df,
        x='Performance_Category',
        hue='Performance_Category',
        legend=False,
        order=order,
        palette=palette,
        edgecolor='black',
        linewidth=0.8
    )
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.annotate(
            f"{int(height)} ({percentage:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height + 1),
            ha='center', va='bottom',
            fontsize=11, weight='bold'
        )
        
    plt.title("Distribution of Student Academic Performance Tiers", weight='bold', pad=15)
    plt.xlabel("Performance Category", weight='bold', labelpad=10)
    plt.ylabel("Number of Students", weight='bold', labelpad=10)
    plt.ylim(0, max(df['Performance_Category'].value_counts()) + 10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_correlation_matrix(df: pd.DataFrame, save_path: str) -> str:
    """
    6. Correlation matrix — Heatmap of numeric academic attributes.
    """
    plt.figure(figsize=(8.5, 6.5))
    
    numeric_cols = [
        'Attendance', 'Study_Hours', 'Assignment_Score', 
        'Midterm_Score', 'Final_Score', 'Previous_GPA', 'Weighted_Score'
    ]
    
    corr_matrix = df[numeric_cols].corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    
    plt.title("Academic Attributes Correlation Matrix", weight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', weight='bold')
    plt.yticks(rotation=0, weight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_dept_score_distribution(df: pd.DataFrame, save_path: str) -> str:
    """
    7. Score distribution by department — Box Plot.
    """
    plt.figure(figsize=(9, 5.5))
    
    num_depts = df['Department'].nunique()
    palette = sns.color_palette("Set2", n_colors=num_depts)
    
    sns.boxplot(
        data=df,
        x='Department',
        y='Final_Score',
        hue='Department',
        legend=False,
        palette=palette,
        linewidth=1.2,
        boxprops=dict(alpha=0.85),
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": "6"}
    )
    
    # Add stripplot overlay to show individual student data points
    sns.stripplot(
        data=df,
        x='Department',
        y='Final_Score',
        color='black',
        alpha=0.35,
        jitter=0.2,
        size=5
    )
    
    plt.title("Final Exam Score Dispersion by Department (Red Diamond = Mean)", weight='bold', pad=15)
    plt.xlabel("Department", weight='bold', labelpad=10)
    plt.ylabel("Final Exam Score (0 - 100)", weight='bold', labelpad=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def generate_all_visualizations(df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Batch generate all 7 required academic visualizations and save to output_dir.
    
    Args:
        df (pd.DataFrame): Processed student dataframe
        output_dir (str): Folder where chart PNG files will be saved
        
    Returns:
        List[str]: List of paths to generated image files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = [
        plot_dept_final_scores(df, os.path.join(output_dir, "01_avg_final_score_by_dept.png")),
        plot_study_hours_vs_final_score(df, os.path.join(output_dir, "02_study_hours_vs_final_score.png")),
        plot_attendance_vs_final_score(df, os.path.join(output_dir, "03_attendance_vs_final_score.png")),
        plot_final_score_distribution(df, os.path.join(output_dir, "04_final_score_distribution.png")),
        plot_category_distribution(df, os.path.join(output_dir, "05_performance_category_distribution.png")),
        plot_correlation_matrix(df, os.path.join(output_dir, "06_correlation_matrix_heatmap.png")),
        plot_dept_score_distribution(df, os.path.join(output_dir, "07_score_distribution_by_department.png")),
    ]
    
    return saved_files
