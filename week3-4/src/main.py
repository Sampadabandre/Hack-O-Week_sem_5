"""
Student Performance & Academic Analytics System
Main Application Entry Point (main.py)

Orchestrates data loading, cleaning, merging, statistical analysis,
NumPy vectorization, Pandas aggregations, visualization generation,
and academic reporting.
"""

import os
import sys

# Ensure src/ and project root are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analytics import StudentAnalytics


def print_banner():
    """Display an attractive CLI banner."""
    print("=" * 80)
    print("        STUDENT PERFORMANCE & ACADEMIC ANALYTICS SYSTEM")
    print("        Week 3-4 Python * NumPy * Pandas * Matplotlib * Seaborn")
    print("=" * 80)


def main():
    """Main execution workflow."""
    print_banner()
    
    # 1. Define filepaths
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")
    charts_dir = os.path.join(output_dir, "charts")
    reports_dir = os.path.join(output_dir, "reports")
    
    student_csv = os.path.join(data_dir, "student_performance.csv")
    dept_csv = os.path.join(data_dir, "department_info.csv")
    report_file = os.path.join(reports_dir, "academic_analytics_summary.txt")
    
    print(f"\n[1/6] Initializing StudentAnalytics Engine...")
    analytics = StudentAnalytics(student_path=student_csv, department_path=dept_csv)
    
    # 2. Load Raw Datasets
    print(f"[2/6] Loading CSV Datasets...")
    raw_students, dept_info = analytics.load_data()
    print(f"      [+] Raw Student Records Loaded : {len(raw_students)} rows, {len(raw_students.columns)} columns")
    print(f"      [+] Department Metadata Loaded : {len(dept_info)} departments")
    
    # 3. Clean and Transform Data
    print(f"\n[3/6] Cleaning Data & Imputing Missing Values...")
    cleaned_df = analytics.clean_data()
    cleaning = analytics.cleaning_summary
    print(f"      [+] Duplicate Rows Removed     : {cleaning['duplicates_removed']}")
    print(f"      [+] Missing Values Imputed     : {cleaning['missing_values_detected']} (Columns: {', '.join(cleaning['imputed_columns'])})")
    print(f"      [+] Cleaned Cohort Size        : {cleaning['cleaned_rows']} students")
    print(f"      [+] Vectorized Weighted Scores : 30% Assignment + 30% Midterm + 40% Final Exam")
    print(f"      [+] Performance Categorization : Excellent, Good, Average, At Risk")
    
    # 4. Statistical Analysis & Research Questions
    print(f"\n[4/6] Computing Statistical Insights & Addressing Research Questions...")
    stats = analytics.calculate_statistics()
    corr = stats['correlations']
    top_10 = stats['top_10_students_df']
    
    print("\n" + "-" * 80)
    print("                       KEY ANALYTICAL FINDINGS")
    print("-" * 80)
    print(f"  1. Average Final Score      : {stats['avg_final_score']:.2f} (Cohort Range: {stats['min_final_score']:.1f} - {stats['max_final_score']:.1f})")
    print(f"  2. Top Department by Score  : {stats['dept_highest_score']['dept']} ({stats['dept_highest_score']['avg_score']:.2f} avg score)")
    print(f"  3. Attendance Impact        : Positive correlation (r = {corr['attendance_vs_final']:+.4f})")
    print(f"  4. Study Time Impact        : Strong positive correlation (r = {corr['study_hours_vs_final']:+.4f})")
    print(f"  5. At-Risk Students Rate    : {stats['at_risk_percentage']:.2f}% ({stats['at_risk_count']} students)")
    print(f"  6. Top Performing Student   : {top_10.iloc[0]['Name']} ({top_10.iloc[0]['Department']} - Weighted Score: {top_10.iloc[0]['Weighted_Score']:.2f})")
    print(f"  7. Top Department by GPA    : {stats['dept_highest_gpa']['dept']} ({stats['dept_highest_gpa']['avg_gpa']:.2f} avg GPA)")
    print(f"  8. Correlation Summary      :")
    print(f"     - Attendance vs Final Score   : r = {corr['attendance_vs_final']:+.4f}")
    print(f"     - Study Hours vs Final Score  : r = {corr['study_hours_vs_final']:+.4f}")
    print(f"     - Previous GPA vs Final Score : r = {corr['previous_gpa_vs_final']:+.4f}")
    print("-" * 80)
    
    # Department Breakdown Table
    print("\n[Department-Wise Summary Table]")
    print(f"{'Dept':<6} {'Students':<10} {'Avg Final':<12} {'Avg Weighted':<14} {'Avg GPA':<10}")
    print("-" * 52)
    for dept, data in stats['dept_summary_dict'].items():
        print(f"{dept:<6} {data['Count']:<10} {data['Avg_Final']:<12.2f} {data['Avg_Weighted']:<14.2f} {data['Avg_GPA']:<10.2f}")
        
    # Top 5 Students Preview
    print("\n[Top 5 Students Preview]")
    for idx, row in top_10.head(5).iterrows():
        print(f"  #{idx+1}: {row['Student_ID']} - {row['Name']:<20} ({row['Department']}) | Weighted: {row['Weighted_Score']:.2f} | Final: {row['Final_Score']:.2f} | GPA: {row['Previous_GPA']:.2f}")

    # 5. Generate Visualizations
    print(f"\n[5/6] Generating Matplotlib & Seaborn Visualizations...")
    chart_paths = analytics.visualize_data(charts_dir)
    for path in chart_paths:
        filename = os.path.basename(path)
        print(f"      [+] Generated Chart : output/charts/{filename}")
        
    # 6. Generate Summary Report
    print(f"\n[6/6] Writing Summary Report...")
    analytics.generate_report(report_file)
    print(f"      [+] Report Saved    : output/reports/academic_analytics_summary.txt")
    
    print("\n" + "=" * 80)
    print("  [SUCCESS] Pipeline Execution Finished! All outputs are ready.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
