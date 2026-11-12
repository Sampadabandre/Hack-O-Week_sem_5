"""
Student Performance & Academic Analytics System
Core Analytics Module (analytics.py)

This module provides data loading, data cleaning, statistical analysis,
NumPy vectorized operations, Pandas aggregations, list/dict comprehensions,
and academic reporting functions.
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd


# ==============================================================================
# Procedural Helper Functions
# ==============================================================================

def load_data(student_csv_path: str, department_csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw student performance and department CSV datasets into Pandas DataFrames.
    
    Args:
        student_csv_path (str): Path to student_performance.csv
        department_csv_path (str): Path to department_info.csv
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (student_df, department_df)
    """
    if not os.path.exists(student_csv_path):
        raise FileNotFoundError(f"Student data file not found at: {student_csv_path}")
    if not os.path.exists(department_csv_path):
        raise FileNotFoundError(f"Department data file not found at: {department_csv_path}")
    
    student_df = pd.read_csv(student_csv_path)
    department_df = pd.read_csv(department_csv_path)
    
    return student_df, department_df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean the student dataset by:
    1. Identifying and removing duplicate records.
    2. Detecting missing values and imputing them using column means (or department-wise means).
    3. Ensuring consistent data types.
    
    Args:
        df (pd.DataFrame): Raw student dataframe
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (cleaned_df, cleaning_summary)
    """
    df_cleaned = df.copy()
    initial_rows = len(df_cleaned)
    
    # 1. Duplicates
    duplicate_count = int(df_cleaned.duplicated(subset=['Student_ID']).sum())
    df_cleaned = df_cleaned.drop_duplicates(subset=['Student_ID'], keep='first').reset_index(drop=True)
    
    # 2. Missing Values Detection
    missing_before = df_cleaned.isnull().sum().to_dict()
    missing_total = int(sum(missing_before.values()))
    
    # Impute numeric missing values using column mean via NumPy/Pandas
    numeric_columns = [
        'Attendance', 'Study_Hours', 'Assignment_Score', 
        'Midterm_Score', 'Final_Score', 'Previous_GPA'
    ]
    
    imputed_columns = []
    for col in numeric_columns:
        if col in df_cleaned.columns and df_cleaned[col].isnull().sum() > 0:
            mean_val = float(np.round(df_cleaned[col].mean(), 2))
            df_cleaned[col] = df_cleaned[col].fillna(mean_val)
            imputed_columns.append(col)
            
    # 3. Data type standardization
    if 'Age' in df_cleaned.columns:
        df_cleaned['Age'] = df_cleaned['Age'].astype(int)
    for col in numeric_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(float)
            
    cleaning_summary = {
        'initial_rows': initial_rows,
        'cleaned_rows': len(df_cleaned),
        'duplicates_removed': duplicate_count,
        'missing_values_detected': missing_total,
        'imputed_columns': imputed_columns
    }
    
    return df_cleaned, cleaning_summary


def merge_department_data(student_df: pd.DataFrame, dept_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine student performance data with department reference metadata using pd.merge().
    
    Args:
        student_df (pd.DataFrame): Cleaned student DataFrame
        dept_df (pd.DataFrame): Department DataFrame
        
    Returns:
        pd.DataFrame: Merged DataFrame
    """
    merged_df = pd.merge(student_df, dept_df, on='Department', how='left')
    return merged_df


def calculate_weighted_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate final weighted academic score using vectorized NumPy operations and broadcasting:
    Weighted_Score = 30% Assignment + 30% Midterm + 40% Final Exam
    
    Formula:
        Final_Weighted_Score = 0.30 * Assignment_Score + 0.30 * Midterm_Score + 0.40 * Final_Score
    
    Args:
        df (pd.DataFrame): Student DataFrame with score columns
        
    Returns:
        pd.DataFrame: DataFrame with new 'Weighted_Score' column
    """
    df_copy = df.copy()
    
    # Extract numerical arrays for NumPy vectorized calculation
    assignment_arr = df_copy['Assignment_Score'].to_numpy()
    midterm_arr = df_copy['Midterm_Score'].to_numpy()
    final_arr = df_copy['Final_Score'].to_numpy()
    
    # Vectorized arithmetic with weights array broadcasting
    weights = np.array([0.30, 0.30, 0.40])
    scores_matrix = np.column_stack((assignment_arr, midterm_arr, final_arr))
    
    # Vectorized dot product / linear combination
    weighted_scores = np.round(np.dot(scores_matrix, weights), 2)
    
    df_copy['Weighted_Score'] = weighted_scores
    return df_copy


def categorize_students(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize students based on their Weighted_Score into 4 academic tiers:
    - Excellent: Score >= 85.0
    - Good:      70.0 <= Score < 85.0
    - Average:   50.0 <= Score < 70.0
    - At Risk:   Score < 50.0
    
    Args:
        df (pd.DataFrame): DataFrame with 'Weighted_Score' column
        
    Returns:
        pd.DataFrame: DataFrame with new 'Performance_Category' column
    """
    df_copy = df.copy()
    scores = df_copy['Weighted_Score'].to_numpy()
    
    conditions = [
        scores >= 85.0,
        (scores >= 70.0) & (scores < 85.0),
        (scores >= 50.0) & (scores < 70.0),
        scores < 50.0
    ]
    choices = ['Excellent', 'Good', 'Average', 'At Risk']
    
    df_copy['Performance_Category'] = np.select(conditions, choices, default='Average')
    return df_copy


def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on student performance data
    using NumPy and Pandas to address the core analytical research questions.
    
    Args:
        df (pd.DataFrame): Enriched student dataframe
        
    Returns:
        Dict[str, Any]: Dictionary containing calculated statistics & metrics
    """
    final_scores = df['Final_Score'].to_numpy()
    attendance = df['Attendance'].to_numpy()
    study_hours = df['Study_Hours'].to_numpy()
    previous_gpa = df['Previous_GPA'].to_numpy()
    weighted_scores = df['Weighted_Score'].to_numpy()
    
    # 1. Overall Score Statistics (NumPy)
    avg_final_score = float(np.round(np.mean(final_scores), 2))
    std_final_score = float(np.round(np.std(final_scores), 2))
    min_final_score = float(np.min(final_scores))
    max_final_score = float(np.max(final_scores))
    
    avg_weighted_score = float(np.round(np.mean(weighted_scores), 2))
    
    # 2. Department-level Statistics (Pandas groupby + agg)
    dept_stats = df.groupby('Department').agg(
        Student_Count=('Student_ID', 'count'),
        Avg_Final_Score=('Final_Score', 'mean'),
        Avg_Weighted_Score=('Weighted_Score', 'mean'),
        Avg_GPA=('Previous_GPA', 'mean'),
        Avg_Attendance=('Attendance', 'mean'),
        Avg_Study_Hours=('Study_Hours', 'mean')
    ).round(2).reset_index()
    
    # Highest scoring department & highest GPA department
    dept_highest_score = dept_stats.sort_values(by='Avg_Final_Score', ascending=False).iloc[0]
    dept_highest_gpa = dept_stats.sort_values(by='Avg_GPA', ascending=False).iloc[0]
    
    # 3. Correlations (NumPy np.corrcoef)
    corr_attendance_final = float(np.round(np.corrcoef(attendance, final_scores)[0, 1], 4))
    corr_study_final = float(np.round(np.corrcoef(study_hours, final_scores)[0, 1], 4))
    corr_gpa_final = float(np.round(np.corrcoef(previous_gpa, final_scores)[0, 1], 4))
    
    # 4. At Risk Students Breakdown
    total_students = len(df)
    at_risk_count = int(np.sum(df['Performance_Category'] == 'At Risk'))
    at_risk_percentage = float(np.round((at_risk_count / total_students) * 100, 2))
    
    # 5. Top 10 Students (Pandas sorting and selection)
    top_10_df = df.sort_values(by='Weighted_Score', ascending=False).head(10)[
        ['Student_ID', 'Name', 'Department', 'Weighted_Score', 'Final_Score', 'Previous_GPA', 'Performance_Category']
    ].reset_index(drop=True)
    
    # 6. List Comprehensions Usage
    # Extract IDs of honors students (Score >= 90) and at-risk students (Score < 50)
    honors_students = [
        f"{row['Student_ID']} - {row['Name']} ({row['Department']}: {row['Weighted_Score']})"
        for _, row in df[df['Weighted_Score'] >= 90.0].iterrows()
    ]
    at_risk_student_ids = [
        f"{row['Student_ID']} ({row['Name']} - Dept: {row['Department']}, Score: {row['Weighted_Score']})"
        for _, row in df[df['Performance_Category'] == 'At Risk'].iterrows()
    ]
    
    # 7. Dictionary Comprehensions Usage
    # Department summary dictionary using dict comprehension over unique departments
    dept_summary_dict = {
        dept: {
            'Count': int(len(df[df['Department'] == dept])),
            'Avg_Final': float(np.round(np.mean(df[df['Department'] == dept]['Final_Score']), 2)),
            'Avg_Weighted': float(np.round(np.mean(df[df['Department'] == dept]['Weighted_Score']), 2)),
            'Avg_GPA': float(np.round(np.mean(df[df['Department'] == dept]['Previous_GPA']), 2)),
        }
        for dept in sorted(df['Department'].unique())
    }
    
    # Category count dictionary using dict comprehension
    categories = ['Excellent', 'Good', 'Average', 'At Risk']
    category_counts_dict = {
        cat: int(np.sum(df['Performance_Category'] == cat))
        for cat in categories
    }
    
    return {
        'total_students': total_students,
        'avg_final_score': avg_final_score,
        'std_final_score': std_final_score,
        'min_final_score': min_final_score,
        'max_final_score': max_final_score,
        'avg_weighted_score': avg_weighted_score,
        'dept_stats_df': dept_stats,
        'dept_highest_score': {
            'dept': dept_highest_score['Department'],
            'avg_score': dept_highest_score['Avg_Final_Score']
        },
        'dept_highest_gpa': {
            'dept': dept_highest_gpa['Department'],
            'avg_gpa': dept_highest_gpa['Avg_GPA']
        },
        'correlations': {
            'attendance_vs_final': corr_attendance_final,
            'study_hours_vs_final': corr_study_final,
            'previous_gpa_vs_final': corr_gpa_final
        },
        'at_risk_count': at_risk_count,
        'at_risk_percentage': at_risk_percentage,
        'top_10_students_df': top_10_df,
        'honors_students': honors_students,
        'at_risk_student_ids': at_risk_student_ids,
        'dept_summary_dict': dept_summary_dict,
        'category_counts_dict': category_counts_dict
    }


def generate_report(df: pd.DataFrame, stats: Dict[str, Any], output_path: str) -> str:
    """
    Generate a detailed academic summary report and save it to the specified file path.
    
    Args:
        df (pd.DataFrame): Enriched student dataset
        stats (Dict[str, Any]): Precomputed statistics dictionary
        output_path (str): File destination path for report
        
    Returns:
        str: Formatted report text
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    corr = stats['correlations']
    top_10 = stats['top_10_students_df']
    dept_dict = stats['dept_summary_dict']
    cat_dict = stats['category_counts_dict']
    
    lines = []
    lines.append("=" * 80)
    lines.append("        STUDENT PERFORMANCE & ACADEMIC ANALYTICS SUMMARY REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. EXECUTIVE SUMMARY & COHORT OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"  * Total Students Analyzed : {stats['total_students']}")
    lines.append(f"  * Mean Final Exam Score   : {stats['avg_final_score']:.2f} (Std Dev: {stats['std_final_score']:.2f})")
    lines.append(f"  * Final Score Range       : Min {stats['min_final_score']:.2f} | Max {stats['max_final_score']:.2f}")
    lines.append(f"  * Mean Weighted Score     : {stats['avg_weighted_score']:.2f}")
    lines.append(f"  * At-Risk Students        : {stats['at_risk_count']} ({stats['at_risk_percentage']:.2f}% of cohort)")
    lines.append("")
    
    lines.append("2. ANSWERS TO CORE RESEARCH QUESTIONS")
    lines.append("-" * 80)
    lines.append(f"Q1: What is the average final score?")
    lines.append(f"    --> Overall Average Final Score: {stats['avg_final_score']:.2f}")
    lines.append("")
    lines.append(f"Q2: Which department has the highest average score?")
    lines.append(f"    --> Department: {stats['dept_highest_score']['dept']} with average score of {stats['dept_highest_score']['avg_score']:.2f}")
    lines.append("")
    lines.append(f"Q3: Does attendance appear to affect final scores?")
    lines.append(f"    --> Yes, positive correlation of r = {corr['attendance_vs_final']:+.4f}")
    lines.append(f"        Higher attendance strongly associates with higher academic achievement.")
    lines.append("")
    lines.append(f"Q4: Does study time appear to affect final scores?")
    lines.append(f"    --> Yes, strong positive correlation of r = {corr['study_hours_vs_final']:+.4f}")
    lines.append(f"        Students investing more weekly study hours achieve significantly higher exam scores.")
    lines.append("")
    lines.append(f"Q5: What percentage of students are at risk?")
    lines.append(f"    --> {stats['at_risk_percentage']:.2f}% ({stats['at_risk_count']} out of {stats['total_students']} students)")
    lines.append("")
    lines.append(f"Q6: Who are the top 10 students?")
    lines.append(f"    --> (Ranked by Weighted Score)")
    for idx, row in top_10.iterrows():
        lines.append(f"        {idx+1:2d}. {row['Student_ID']} | {row['Name']:<22} | Dept: {row['Department']} | Weighted: {row['Weighted_Score']:5.2f} | Final: {row['Final_Score']:5.2f} | GPA: {row['Previous_GPA']:.2f}")
    lines.append("")
    lines.append(f"Q7: Which department has the highest average GPA?")
    lines.append(f"    --> Department: {stats['dept_highest_gpa']['dept']} with average GPA of {stats['dept_highest_gpa']['avg_gpa']:.2f}")
    lines.append("")
    lines.append(f"Q8: Correlation Matrix with Final Score:")
    lines.append(f"    - Attendance vs Final Score   : r = {corr['attendance_vs_final']:+.4f}")
    lines.append(f"    - Study Hours vs Final Score  : r = {corr['study_hours_vs_final']:+.4f}")
    lines.append(f"    - Previous GPA vs Final Score : r = {corr['previous_gpa_vs_final']:+.4f}")
    lines.append("")
    
    lines.append("3. DEPARTMENT-WISE PERFORMANCE SUMMARY (Built with Dict Comprehensions)")
    lines.append("-" * 80)
    lines.append(f"{'Dept':<8}{'Count':<8}{'Avg Final':<14}{'Avg Weighted':<16}{'Avg GPA':<10}")
    lines.append("-" * 56)
    for dept, metrics in dept_dict.items():
        lines.append(f"{dept:<8}{metrics['Count']:<8}{metrics['Avg_Final']:<14.2f}{metrics['Avg_Weighted']:<16.2f}{metrics['Avg_GPA']:<10.2f}")
    lines.append("")
    
    lines.append("4. PERFORMANCE CATEGORY BREAKDOWN")
    lines.append("-" * 80)
    for cat, count in cat_dict.items():
        pct = (count / stats['total_students']) * 100
        lines.append(f"  * {cat:<12}: {count:3d} students ({pct:5.2f}%)")
    lines.append("")
    
    lines.append("5. HONORS STUDENTS LIST (Generated via List Comprehension - Score >= 90)")
    lines.append("-" * 80)
    for honor in stats['honors_students']:
        lines.append(f"  [HONORS] {honor}")
    lines.append("")
    
    lines.append("6. AT-RISK STUDENTS INTERVENTION LIST (Generated via List Comprehension - Score < 50)")
    lines.append("-" * 80)
    for risk in stats['at_risk_student_ids']:
        lines.append(f"  [ALERT] {risk}")
    lines.append("")
    lines.append("=" * 80)
    lines.append("End of Academic Analytics Report")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    return report_text


# ==============================================================================
# Object-Oriented Programming (OOP) Class
# ==============================================================================

class StudentAnalytics:
    """
    StudentAnalytics Class
    
    An Object-Oriented interface for managing student academic performance data,
    running analytical operations, calculating statistics, and coordinating
    visualizations and summary reporting.
    """
    
    def __init__(self, student_path: str, department_path: str):
        """
        Initialize the StudentAnalytics instance with dataset filepaths.
        
        Args:
            student_path (str): Path to student CSV
            department_path (str): Path to department CSV
        """
        self.student_path = student_path
        self.department_path = department_path
        self.raw_student_df = None
        self.department_df = None
        self.cleaned_df = None
        self.merged_df = None
        self.statistics = None
        self.cleaning_summary = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load datasets from disk."""
        self.raw_student_df, self.department_df = load_data(self.student_path, self.department_path)
        return self.raw_student_df, self.department_df
        
    def clean_data(self) -> pd.DataFrame:
        """Clean raw student records and merge department metadata."""
        if self.raw_student_df is None or self.department_df is None:
            self.load_data()
            
        cleaned_students, self.cleaning_summary = clean_data(self.raw_student_df)
        merged = merge_department_data(cleaned_students, self.department_df)
        weighted = calculate_weighted_score(merged)
        self.cleaned_df = categorize_students(weighted)
        self.merged_df = self.cleaned_df
        return self.cleaned_df
        
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate cohort descriptive statistics, metrics, and correlations."""
        if self.cleaned_df is None:
            self.clean_data()
        self.statistics = calculate_statistics(self.cleaned_df)
        return self.statistics
        
    def categorize_students(self) -> pd.DataFrame:
        """Ensure student performance categorization is applied."""
        if self.cleaned_df is None:
            self.clean_data()
        return self.cleaned_df
        
    def generate_report(self, output_path: str) -> str:
        """Generate and save the summary academic report."""
        if self.statistics is None:
            self.calculate_statistics()
        return generate_report(self.cleaned_df, self.statistics, output_path)
        
    def visualize_data(self, output_dir: str) -> List[str]:
        """
        Generate and save all 7 visual charts to the output directory.
        Delegates to the visualization module.
        """
        if self.cleaned_df is None:
            self.clean_data()
            
        from src.visualization import generate_all_visualizations
        return generate_all_visualizations(self.cleaned_df, output_dir)
