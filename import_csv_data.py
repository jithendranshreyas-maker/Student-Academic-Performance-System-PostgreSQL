"""Import the five PROJECT_2_2 CSV datasets into the new PostgreSQL database.

Run from the PROJECT_2_2 folder after applying schema_postgresql.sql.
Requires: pandas, psycopg2-binary, python-dotenv.
"""
from pathlib import Path
import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env2", override=False)

FILES = {
    "students": BASE / "students_5_years.csv",
    "curriculum": BASE / "curriculum_5_years.csv",
    "academic_records": BASE / "academic_records_5_years.csv",
    "semester_results": BASE / "semester_results_5_years.csv",
}

def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD") or None,
        dbname=os.getenv("DB_NAME", "project_2_postgres"),
        connect_timeout=10,
    )

def records(df, columns):
    df = df[columns].astype(object).where(pd.notna(df[columns]), None)
    return list(df.itertuples(index=False, name=None))

def main():
    for key, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Put the supplied CSV files beside this script "
                "and ensure their filenames match the expected names."
            )

    students = pd.read_csv(FILES["students"], dtype={"Student_ID": str})
    curriculum = pd.read_csv(FILES["curriculum"])
    academic = pd.read_csv(FILES["academic_records"], dtype={"Student_ID": str})
    semesters = pd.read_csv(FILES["semester_results"], dtype={"Student_ID": str})

    # Normalize IDs before inserting so they match between datasets.
    for frame in (students, academic, semesters):
        frame["Student_ID"] = frame["Student_ID"].astype(str)

    # The source students CSV has no current_year/current_semester columns.
    # Derive each student's latest recorded term from academic_records, then
    # map semester 1-2 -> year 1, 3-4 -> year 2, etc. Do not confuse the
    # calendar Academic_Year (e.g. 2025) with the student's study year.
    latest_term = (
        academic.sort_values(["Academic_Year", "Semester"])
        .groupby("Student_ID", as_index=False)
        .tail(1)[["Student_ID", "Academic_Year", "Semester"]]
        .rename(columns={"Academic_Year": "Latest_Academic_Year", "Semester": "Latest_Semester"})
    )
    students = students.merge(latest_term, on="Student_ID", how="left")
    students["current_semester"] = pd.to_numeric(students["Latest_Semester"], errors="coerce")
    students["current_year"] = ((students["current_semester"] + 1) // 2).where(
        students["current_semester"].notna(), None
    )

    con = connect()
    try:
        with con.cursor() as cur:
            # Student profile rows are upserted without overwriting account-related fields.
            execute_values(cur, """
                INSERT INTO students
                    (student_id, name, email, section, branch, academic_year, current_year, current_semester)
                VALUES %s
                ON CONFLICT (student_id) DO UPDATE SET
                    name = EXCLUDED.name, email = EXCLUDED.email,
                    section = EXCLUDED.section, branch = EXCLUDED.branch,
                    academic_year = EXCLUDED.academic_year,
                    current_year = EXCLUDED.current_year,
                    current_semester = EXCLUDED.current_semester
            """, records(students, [
                "Student_ID", "Name", "Email", "Section", "Branch", "Academic_Year",
                "current_year", "current_semester"
            ]))

            execute_values(cur, """
                INSERT INTO curriculum
                    (branch, academic_year, semester, subject_code, subject_name, credits, maximum_marks)
                VALUES %s
                ON CONFLICT (branch, academic_year, semester, subject_code) DO UPDATE SET
                    subject_name = EXCLUDED.subject_name,
                    credits = EXCLUDED.credits,
                    maximum_marks = EXCLUDED.maximum_marks
            """, records(curriculum, ["Branch", "Academic_Year", "Semester", "Subject_Code", "Subject_Name", "Credits", "Maximum_Marks"]))

            # Resolve each academic record to the matching curriculum row using its composite key.
            cur.execute("SELECT subject_id, branch, academic_year, semester, subject_code FROM curriculum")
            curriculum_ids = {
                (str(branch), int(year), int(semester), str(code)): int(subject_id)
                for subject_id, branch, year, semester, code in cur.fetchall()
            }
            academic["subject_id"] = academic.apply(
                lambda r: curriculum_ids.get((str(r["Branch"]), int(r["Academic_Year"]), int(r["Semester"]), str(r["Subject_Code"]))),
                axis=1,
            )
            academic_columns = [
                "Student_ID", "Name", "Email", "Section", "Branch", "Academic_Year", "Semester",
                "subject_id", "Subject_Code", "Subject_Name", "Credits", "Quiz1", "Quiz2", "Midsem",
                "Assignment", "Endsem", "Classes_Attended", "Classes_Held", "Attendance_Percentage",
                "Marks_Obtained", "Maximum_Marks", "Grade", "Grade_Point", "Backlog",
            ]
            academic_records = records(academic, academic_columns)
            execute_values(cur, """
                INSERT INTO academic_records (
                    student_id, name, email, section, branch, academic_year, semester,
                    subject_id, subject_code, subject_name, credits, quiz1, quiz2, midsem,
                    assignment, endsem, classes_attended, classes_held, attendance_percentage,
                    marks_obtained, maximum_marks, grade, grade_point, backlog
                ) VALUES %s
                ON CONFLICT (student_id, academic_year, semester, subject_code) DO UPDATE SET
                    name = EXCLUDED.name, email = EXCLUDED.email, section = EXCLUDED.section,
                    branch = EXCLUDED.branch, subject_id = EXCLUDED.subject_id,
                    subject_name = EXCLUDED.subject_name, credits = EXCLUDED.credits,
                    quiz1 = EXCLUDED.quiz1, quiz2 = EXCLUDED.quiz2, midsem = EXCLUDED.midsem,
                    assignment = EXCLUDED.assignment, endsem = EXCLUDED.endsem,
                    classes_attended = EXCLUDED.classes_attended, classes_held = EXCLUDED.classes_held,
                    attendance_percentage = EXCLUDED.attendance_percentage,
                    marks_obtained = EXCLUDED.marks_obtained, maximum_marks = EXCLUDED.maximum_marks,
                    grade = EXCLUDED.grade, grade_point = EXCLUDED.grade_point, backlog = EXCLUDED.backlog
            """, academic_records, page_size=1000)

            semester_records = records(semesters, [
                "Student_ID", "Academic_Year", "Semester", "Semester_Percentage", "SGA", "Total_Backlogs", "Total_Credits"
            ])
            execute_values(cur, """
                INSERT INTO semester_results
                    (student_id, academic_year, semester, semester_percentage, sga, total_backlogs, total_credits, result_type)
                VALUES %s
                ON CONFLICT (student_id, academic_year, semester, result_type) DO UPDATE SET
                    semester_percentage = EXCLUDED.semester_percentage,
                    sga = EXCLUDED.sga,
                    total_backlogs = EXCLUDED.total_backlogs,
                    total_credits = EXCLUDED.total_credits
            """, [row + ("Actual",) for row in semester_records], page_size=1000)

        con.commit()
        with con.cursor() as cur:
            for table in ("students", "curriculum", "academic_records", "semester_results"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                print(f"{table}: {cur.fetchone()[0]:,} rows")
        print("CSV import completed successfully.")
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

if __name__ == "__main__":
    main()
