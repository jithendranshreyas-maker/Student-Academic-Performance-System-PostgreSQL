# 🎓 Student Academic Performance System

A Python-based academic performance management system built with **Streamlit, PostgreSQL, and Machine Learning**. The project combines academic record management with student performance analysis and marks prediction.

## Overview

The Student Academic Performance System is designed to organize student academic data, analyze performance across semesters, and support marks prediction through a trained machine learning pipeline.

The project uses PostgreSQL for persistent data storage and Streamlit to provide an interactive web interface.

## Features

- **Student Dashboard** — access student-related academic information.
- **Academic Performance Analysis** — work with marks, grades, attendance, and semester results.
- **Marks Prediction** — use a trained machine learning pipeline to estimate academic marks.
- **Backlog Tracking** — identify and review subjects with backlogs.
- **Authentication** — student and administrative account functionality.
- **PostgreSQL Integration** — store and retrieve application data using a relational database.
- **Data Import Utilities** — populate the database from CSV datasets.

*Available functionality depends on the current application configuration and database setup.*

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and data processing |
| Streamlit | Interactive web interface |
| PostgreSQL | Relational database |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning pipeline |
| Joblib / Pickle | Loading saved model artifacts |

## Project Structure

```text
PROJECT_2_2/
├── app2_2.py
├── app2_2_postgres.py
├── auth2.py
├── create_admin.py
├── import_csv_data.py
├── schema_postgresql.sql
├── requirements2.txt
├── PROJECT2.py
├── best_marks_prediction_pipeline.pkl
├── preprocessor_linear.pkl
├── model_metrics.pkl
├── residual_std.pkl
├── prediction_error.pkl
├── error_by_mark_range.csv
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jithendranshreyas-maker/Student-Academic-Performance-System-PostgreSQL.git
cd Student-Academic-Performance-System-PostgreSQL
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements2.txt
```

### 4. Configure PostgreSQL

Install and start PostgreSQL, then create a database for the project.

For example:

```sql
CREATE DATABASE project_2_postgres;
```

Configure the connection settings using the environment variables expected by `auth2.py` and the application. Keep credentials in a local environment file or an appropriate secrets manager.

**Never commit database passwords, API keys, or secret configuration files to GitHub.**

### 5. Initialize the database

Use the provided schema file to create the database tables:

```bash
psql -U YOUR_DB_USER -d project_2_postgres -f schema_postgresql.sql
```

Replace `YOUR_DB_USER` with your PostgreSQL username.

### 6. Import sample data

Place the required CSV datasets in the project directory and follow the configuration expected by `import_csv_data.py`.

Run the importer when the database and connection settings are ready:

```bash
python import_csv_data.py
```

### 7. Launch the application

```bash
streamlit run app2_2.py
```

Open the local URL displayed in the terminal to access the application.

## Machine Learning

The project includes saved machine learning artifacts for academic marks prediction, including a trained prediction pipeline and preprocessing components.

Model evaluation artifacts are also included. Numerical performance claims should be based on the actual evaluation results produced by the project.

## Security

- Keep environment files and database credentials private.
- Use appropriate authentication and database permissions.
- Do not publish real student records or personally identifiable information.
- Use synthetic or anonymized data when sharing demonstrations publicly.

## Future Improvements

- Deploy the application for public demonstration.
- Add automated model evaluation and monitoring.
- Improve academic performance visualizations.
- Add more comprehensive testing and validation.
- Document model performance and limitations.

## Author

**Shreyas Jithendran**

GitHub: [@jithendranshreyas-maker](https://github.com/jithendranshreyas-maker)

---

*Developed as a machine learning and database application project focused on academic performance analysis and prediction.*

