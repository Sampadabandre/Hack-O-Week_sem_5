# 🎓 Student Performance & Academic Analytics System

> **A comprehensive, beginner-friendly Python data analytics system demonstrating Object-Oriented Programming (OOP), modular functions, list/dictionary comprehensions, NumPy vectorization, Pandas data transformation, and Matplotlib/Seaborn statistical visualizations.**

---

## 📌 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Objective](#-objective)
3. [Technologies Used](#-technologies-used)
4. [Key Features](#-key-features)
5. [Dataset Description](#-dataset-description)
6. [Week 3–4 Python & Data Science Concepts](#-week-34-python--data-science-concepts)
   - [Core Python & OOP](#1-core-python--oop)
   - [NumPy Concepts](#2-numpy-concepts)
   - [Pandas Concepts](#3-pandas-concepts)
   - [Visualization Techniques](#4-visualization-techniques)
7. [Project Structure](#-project-structure)
8. [Installation & Setup](#-installation--setup)
9. [How to Run](#-how-to-run)
10. [Analytical Findings & Research Answers](#-analytical-findings--research-answers)
11. [Visualizations Gallery](#-visualizations-gallery)
12. [Future Improvements](#-future-improvements)

---

## 📖 Problem Statement

Academic institutions often struggle to identify students needing timely academic interventions, analyze department-level academic trends, and understand the core behavioral drivers of student achievement (such as study hours and attendance rates). This project develops an automated, reproducible analytics engine that ingests raw student performance data, handles real-world data imperfections (missing values and duplicate records), computes weighted scores, segments students into actionable performance tiers, and generates academic reports and statistical visualizations.

---

## 🎯 Objective

- Build a clean, modular, and beginner-friendly Python analytics pipeline.
- Demonstrate mastery of **Week 3–4 Python Data Science concepts** (Functions, Classes, List/Dict Comprehensions, NumPy arrays/broadcasting, Pandas transformations/merging/aggregations, and Seaborn/Matplotlib charts).
- Answer 8 critical academic research questions to support institutional decision-making.
- Output clean visualizations to `output/charts/` and formatted reports to `output/reports/`.

---

## 💻 Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python (>=3.9)** | Core programming language, procedural functions, OOP, comprehensions |
| **NumPy (>=1.24)** | Array operations, vectorized calculations, broadcasting, linear combinations, correlation matrix |
| **Pandas (>=2.0)** | Data ingestion, missing value imputation, deduplication, merging, `groupby()`, `agg()` |
| **Matplotlib (>=3.7)** | Underlying plotting engine, figure formatting, titles, axis labels, grids |
| **Seaborn (>=0.12)** | Statistical data visualizations, heatmaps, box plots, count plots, regression lines |
| **Jupyter Notebook** | Interactive exploratory data analysis (EDA) walkthrough |

---

## ✨ Key Features

- **Automated Data Cleaning Pipeline**: Detects duplicate entries and imputes missing numeric values using column averages.
- **Relational Data Merging**: Joins raw student exam metrics with institutional department metadata (`pd.merge`).
- **Vectorized Performance Calculations**: Computes composite weighted scores ($30\%$ Assignment $+ 30\%$ Midterm $+ 40\%$ Final Exam) using vectorized matrix operations in NumPy.
- **Academic Tier Classification**: Classifies students into `Excellent`, `Good`, `Average`, and `At Risk` performance categories.
- **Answers to 8 Core Research Questions**: Provides answers to institutional academic performance questions.
- **7 Publication-Ready Visualizations**: Automatically generated and saved to `output/charts/`.
- **Comprehensive Academic Report**: Generated and saved to `output/reports/academic_analytics_summary.txt`.

---

## 📊 Dataset Description

The project utilizes two relational datasets located inside `data/`:

### 1. `student_performance.csv` (150 unique student records)
| Column Name | Data Type | Description | Sample Range |
| :--- | :--- | :--- | :--- |
| `Student_ID` | String | Unique student identifier | `STU1001` – `STU1150` |
| `Name` | String | Full name of student | e.g., Aarav Sharma |
| `Gender` | String | Student gender | Male, Female |
| `Age` | Integer | Student age | 18 – 24 |
| `Department` | String | Academic department code | `CS`, `DS`, `EE`, `ME`, `BT` |
| `Attendance` | Float | Class attendance rate | 56.0% – 98.0% |
| `Study_Hours`| Float | Weekly self-study hours | 3.0 – 27.0 hrs/week |
| `Assignment_Score` | Float | Continuous assessment score | 40.0 – 98.0 |
| `Midterm_Score` | Float | Midterm exam score | 36.0 – 96.0 |
| `Final_Score` | Float | Final exam score | 35.0 – 99.0 |
| `Previous_GPA` | Float | Cumulative GPA scale | 2.15 – 4.00 |

*Note: Includes intentional missing values across numeric attributes and duplicate records to test data cleaning.*

### 2. `department_info.csv` (5 academic departments)
| Department | Department_Name | Faculty | Specialization |
| :--- | :--- | :--- | :--- |
| `CS` | Computer Science & Engineering | Faculty of Engineering | Artificial Intelligence & Software Systems |
| `DS` | Data Science & Analytics | Faculty of Computing | Big Data & Machine Learning |
| `EE` | Electrical Engineering | Faculty of Engineering | Power Systems & Embedded Electronics |
| `ME` | Mechanical Engineering | Faculty of Engineering | Robotics & Thermal Engineering |
| `BT` | Biotechnology | Faculty of Applied Sciences | Bioinformatics & Genetic Engineering |

---

## 🧠 Week 3–4 Python & Data Science Concepts

This project directly demonstrates the learning outcomes of Week 3–4:

### 1. Core Python & OOP
- **Functions (`src/analytics.py`, `src/visualization.py`)**: Distinct single-responsibility functions for loading, cleaning, transforming, aggregating, and visualizing.
- **Object-Oriented Programming (OOP) (`StudentAnalytics` class in `src/analytics.py`)**:
  - `load_data()`: Reads datasets.
  - `clean_data()`: Imputes missing values, removes duplicates, merges departments, calculates weighted scores.
  - `calculate_statistics()`: Computes all cohort statistics and metrics.
  - `categorize_students()`: Segments cohort into performance tiers.
  - `generate_report(output_path)`: Writes text report to disk.
  - `visualize_data(output_dir)`: Delegates to visualization functions.
- **List Comprehensions**:
  ```python
  # Honors students list comprehension (Score >= 90)
  honors_students = [
      f"{row['Student_ID']} - {row['Name']} ({row['Department']}: {row['Weighted_Score']})"
      for _, row in df[df['Weighted_Score'] >= 90.0].iterrows()
  ]
  ```
- **Dictionary Comprehensions**:
  ```python
  # Department summary dictionary comprehension
  dept_summary_dict = {
      dept: {
          'Count': int(len(df[df['Department'] == dept])),
          'Avg_Final': float(np.round(np.mean(df[df['Department'] == dept]['Final_Score']), 2)),
          'Avg_Weighted': float(np.round(np.mean(df[df['Department'] == dept]['Weighted_Score']), 2)),
          'Avg_GPA': float(np.round(np.mean(df[df['Department'] == dept]['Previous_GPA']), 2)),
      }
      for dept in sorted(df['Department'].unique())
  }
  ```

### 2. NumPy Concepts
- **Vectorized Arithmetic & Broadcasting**:
  ```python
  weights = np.array([0.30, 0.30, 0.40])
  scores_matrix = np.column_stack((assignment_arr, midterm_arr, final_arr))
  weighted_scores = np.round(np.dot(scores_matrix, weights), 2)
  ```
- **NumPy Statistical Functions**: `np.mean()`, `np.std()`, `np.min()`, `np.max()`, and `np.corrcoef()`.
- **Vectorized Conditional Selection**: `np.select(conditions, choices, default='Average')` for student tiering.

### 3. Pandas Concepts
- **Data Ingestion & Inspection**: `pd.read_csv()`, `head()`, `shape`, `isnull().sum()`, `duplicated()`.
- **Data Cleaning & Imputation**: `df.drop_duplicates()`, `df[col].fillna(mean_val)`.
- **Relational Merging**: `pd.merge(student_df, dept_df, on='Department', how='left')`.
- **Grouping & Aggregation**: `df.groupby('Department').agg(...)`.
- **Sorting & Filtering**: Filtering at-risk students and sorting top 10 academic achievers.

### 4. Visualization Techniques
- **Matplotlib & Seaborn integration**: High-resolution 300 DPI image outputs, color palettes (`crest`, `Set2`, `coolwarm`), customized labels, titles, legends, value annotations, and grids.

---

## 📁 Project Structure

```text
student-performance-analytics/
│
├── data/
│   ├── student_performance.csv       # Raw student exam records with duplicates & missing values
│   └── department_info.csv           # Department metadata
│
├── src/
│   ├── analytics.py                  # Core analytics functions & StudentAnalytics OOP class
│   ├── visualization.py              # 7 Matplotlib & Seaborn plotting functions
│   └── main.py                       # CLI application entry point and orchestrator
│
├── notebooks/
│   └── student_analysis.ipynb        # Interactive step-by-step Jupyter Notebook
│
├── output/
│   ├── charts/                       # Auto-generated 300 DPI chart images
│   │   ├── 01_avg_final_score_by_dept.png
│   │   ├── 02_study_hours_vs_final_score.png
│   │   ├── 03_attendance_vs_final_score.png
│   │   ├── 04_final_score_distribution.png
│   │   ├── 05_performance_category_distribution.png
│   │   ├── 06_correlation_matrix_heatmap.png
│   │   └── 07_score_distribution_by_department.png
│   │
│   └── reports/                      # Summary text reports
│       └── academic_analytics_summary.txt
│
├── README.md                         # Project documentation
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Workspace
Navigate to your project root folder:
```bash
cd week3-4
```

### 2. Install Required Dependencies
Install the necessary packages using `pip`:
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Run the Main Pipeline
Execute `main.py` from the project root:
```bash
python src/main.py
```

### Run the Interactive Jupyter Notebook
Launch Jupyter Notebook and open `notebooks/student_analysis.ipynb`:
```bash
jupyter notebook notebooks/student_analysis.ipynb
```

---

## 📋 Analytical Findings & Research Answers

When running the system, it automatically calculates and reports answers to the **8 core research questions**:

| Question | Answer / Insight |
| :--- | :--- |
| **Q1: What is the average final score?** | **75.79 / 100** (Cohort Range: 35.0 – 99.0, Std Dev: 17.49) |
| **Q2: Which department has highest average score?** | **Computer Science & Engineering (CS)** with an average final score of **89.13** |
| **Q3: Does attendance appear to affect final scores?** | **Yes, strong positive correlation ($r = +0.9726$)**. Higher attendance directly correlates with higher scores. |
| **Q4: Does study time appear to affect final scores?** | **Yes, strong positive correlation ($r = +0.9582$)**. Higher study hours lead to higher exam scores. |
| **Q5: What percentage of students are at risk?** | **11.33%** (17 out of 150 students scored $< 50.0$ weighted score) |
| **Q6: Who are the top 10 students?** | 1. **Shweta Phadke** (DS - 97.80)<br>2. **Devika Marathe** (CS - 97.20)<br>3. **Nandini Dutta** (CS - 97.10)<br>4. **Simran Kaur** (DS - 96.90)<br>5. **Maitreyi Phatak** (DS - 96.20)<br>6. **Divya Menon** (CS - 95.90)<br>7. **Radhika Chiplunkar** (CS - 95.20)<br>8. **Garima Sehgal** (DS - 95.20)<br>9. **Shalini Venkatesh** (BT - 94.20)<br>10. **Ananya Patel** (DS - 94.20) |
| **Q7: Which department has highest average GPA?** | **Computer Science & Engineering (CS)** with average GPA of **3.68 / 4.00** |
| **Q8: Correlation with Final Score** | • **Attendance vs Final**: $r = +0.9726$<br>• **Study Hours vs Final**: $r = +0.9582$<br>• **Previous GPA vs Final**: $r = +0.9906$ |

### Department-Wise Summary Table
| Department | Student Count | Avg Final Exam Score | Avg Weighted Score | Avg Previous GPA |
| :--- | :--- | :--- | :--- | :--- |
| **Biotechnology (BT)** | 30 | 76.10 | 75.51 | 3.31 |
| **Computer Science (CS)** | 30 | 89.13 | 87.22 | 3.68 |
| **Data Science (DS)** | 30 | 84.37 | 82.64 | 3.52 |
| **Electrical Engineering (EE)** | 30 | 64.57 | 64.36 | 2.97 |
| **Mechanical Engineering (ME)** | 30 | 64.80 | 64.43 | 2.98 |

### Performance Category Breakdown
- **Excellent ($\ge 85.0$)**: 48 students ($32.0\%$)
- **Good ($70.0 - 84.99$)**: 54 students ($36.0\%$)
- **Average ($50.0 - 69.99$)**: 31 students ($20.67\%$)
- **At Risk ($< 50.0$)**: 17 students ($11.33\%$)

---

## 📈 Visualizations Gallery

The following 7 charts are automatically generated inside `output/charts/`:

1. **`01_avg_final_score_by_dept.png`**: Bar chart comparing department final exam averages against the cohort mean line.
2. **`02_study_hours_vs_final_score.png`**: Scatter plot showing relationship between weekly study hours and final score with linear trendline.
3. **`03_attendance_vs_final_score.png`**: Scatter plot illustrating attendance percentage impact on final exam performance.
4. **`04_final_score_distribution.png`**: Histogram with Kernel Density Estimation (KDE), cohort mean, and median lines.
5. **`05_performance_category_distribution.png`**: Count plot showing student distribution across performance tiers (`Excellent`, `Good`, `Average`, `At Risk`).
6. **`06_correlation_matrix_heatmap.png`**: Annotated heatmap displaying Pearson correlations among all academic metrics.
7. **`07_score_distribution_by_department.png`**: Box plot showing score distributions, quartiles, and outliers by department with individual student data points.

---

## 🔮 Future Improvements

- **Predictive Early Warning System**: Implement machine learning classifiers (e.g., Logistic Regression or Decision Trees) to predict at-risk students before midterm exams.
- **Interactive Web Dashboard**: Build an interactive Streamlit or Next.js dashboard for real-time exploratory filtering by instructors.
- **Automated Email Alerts**: Integrate an automated notification system to send personalized academic advisory emails to students in the "At Risk" category.
- **Longitudinal Trend Tracking**: Extend the schema to track student performance across multiple semesters and course modules.

---

## 📄 License & Attribution
Developed for the **Hack-O-Week (Sem V)** Python Data Science Curriculum. Beginner-friendly, modular, and open for educational use.
