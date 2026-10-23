import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import bcrypt
import psycopg2
from auth2 import login, signup_student, get_connection
import os

from dotenv import load_dotenv

load_dotenv(".env2", override=False)


subject_data_query = """
SELECT
    a.quiz1,
    a.quiz2,
    a.midsem,
    a.assignment,
    a.attendance_percentage,
    sr.sga AS prev_sga,
    sr.semester_percentage AS prev_sem_percentage,
    sr.total_backlogs AS prev_total_backlogs
FROM academic_records a
LEFT JOIN semester_results sr
    ON sr.student_id = a.student_id
    AND (
        sr.academic_year < a.academic_year
        OR (
            sr.academic_year = a.academic_year
            AND sr.semester < a.semester
        )
    )
WHERE a.student_id = %s
AND a.subject_code = %s
AND a.academic_year = %s
AND a.semester = %s
ORDER BY sr.academic_year DESC, sr.semester DESC
LIMIT 1
"""
def predict_subject(
    subject_code,
    quiz1,
    quiz2,
    midsem,
    assignment,
    attendance_percentage,
    prev_sga,
    prev_sem_percentage,
    prev_total_backlogs,
    branch,
    semester,
    return_explanation=False
):
    feature_columns = [
        "subject_code",
        "quiz1",
        "quiz2",
        "midsem",
        "assignment",
        "attendance_percentage",
        "prev_sga",
        "prev_sem_percentage",
        "prev_total_backlogs",
        "branch",
        "semester"
    ]

    input_data = {
        "subject_code": subject_code,
        "quiz1": quiz1,
        "quiz2": quiz2,
        "midsem": midsem,
        "assignment": assignment,
        "attendance_percentage": attendance_percentage,
        "prev_sga": prev_sga,
        "prev_sem_percentage": prev_sem_percentage,
        "prev_total_backlogs": prev_total_backlogs,
        "branch": branch,
        "semester": semester
    }

    prediction_input = pd.DataFrame([input_data])[feature_columns]

    predicted_marks = float(model.predict(prediction_input)[0])

    if not return_explanation:
        return predicted_marks

    # Use the final pipeline step as the estimator instead of assuming its
    # step name is exactly "model".
    if hasattr(model, "steps") and model.steps:
        estimator = model.steps[-1][1]
        preprocessing_pipeline = model[:-1]
    else:
        estimator = model
        preprocessing_pipeline = preprocessor

    if not hasattr(estimator, "coef_"):
        raise ValueError(
            "Factor explanations require a linear model with coefficients."
        )

    transformed_input = preprocessing_pipeline.transform(prediction_input)
    feature_names = preprocessing_pipeline.get_feature_names_out()

    if hasattr(transformed_input, "toarray"):
        transformed_values = transformed_input.toarray().reshape(-1)
    else:
        transformed_values = np.asarray(transformed_input).reshape(-1)

    coefficients = np.asarray(estimator.coef_).reshape(-1)

    if not (
        len(feature_names)
        == len(transformed_values)
        == len(coefficients)
    ):
        raise ValueError(
            "Feature names, transformed inputs, and coefficients "
            "have different dimensions."
        )

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "input_value": transformed_values,
        "coefficient": coefficients,
        "contribution": transformed_values * coefficients
    })

    return predicted_marks, explanation_df

st.set_page_config(
    page_title="Student Academic Performance System",
    page_icon="🎓",
    layout="wide"
)


model = joblib.load("best_marks_prediction_pipeline.pkl")
preprocessor = joblib.load("preprocessor_linear.pkl")
residual_std=joblib.load("residual_std.pkl")
model_metrics = joblib.load("model_metrics.pkl")
prediction_error=joblib.load("prediction_error.pkl")
error_by_range = pd.read_csv("error_by_mark_range.csv",index_col=0)

st.markdown(
    """
    <style>

    header[data-testid="stHeader"] {
    }

    .top-nav {
        position: relative;
        width: 100%;
        min-height: 65px;
        z-index: 1;
    }

    .top-nav-title {
        font-size: 22px;
        font-weight: bold;
        margin-right: 40px;
    }

    .main .block-container {
        padding-top: 90px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="top-nav">
        <div class="top-nav-title">
            🎓 Student Academic Performance & Risk Analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

pages = [
    "Student Dashboard",
    "Prediction Dashboard",
    "Student Performance",
    "Overall Academic Prediction",
    "Individual Subject Prediction",
    "Historical Comparison",
    "Student Risk Analysis",
    "Backlog Tracker",
    "Prediction Validation"
]

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    display: block !important;
    min-width: 260px !important;
    width: 260px !important;
    transform: none !important;
}

button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarExpandButton"] {
    display: none !important;
}

div[data-testid="stHorizontalBlock"] {
    gap: 0.75rem;
}

.top-navigation {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--background-color);
    padding: 0.5rem 0;
}

.fixed-left-panel {
    position: fixed;
    top: 150px;
    left: 2rem;
    bottom: 1rem;
    width: 240px;
    overflow-y: auto;
    padding: 0.75rem;
    background: var(--background-color);
    z-index: 100;
}

/* Keep the top navigation visible while scrolling */
div[data-testid="stVerticalBlock"]:has(> div.top-navigation) {
    position: sticky;
    top: 0;
    z-index: 999;
    background: var(--background-color);
}

/* Prevent the page content from becoming too narrow */
div[data-testid="stMainBlockContainer"] {
    max-width: 100%;
}

/* Keep the left panel's contents available while scrolling */
div[data-testid="stHorizontalBlock"]:has(
    div[data-testid="stColumn"]:first-child
) {
    align-items: flex-start;
}


.main-content-panel {
    margin-left: 270px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-navigation">', unsafe_allow_html=True)


# ---------------- AUTHENTICATION ----------------

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"] or "user" not in st.session_state:
    login_tab, signup_tab = st.tabs(
        ["Student / Admin Login", "Student Sign Up"]
    )

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            submitted = st.form_submit_button("Login")

        if submitted:
            if login(username, password):
                st.rerun()
            else:
                st.error("Invalid username or password, or account is inactive.")


    with signup_tab:
        with st.form("student_signup_form"):
            st.subheader("Create Student Account")

            name = st.text_input("Full Name")
            new_username = st.text_input("Choose a Username")
            new_email = st.text_input("Email Address")
            phone = st.text_input("Phone Number")

            branch = st.selectbox(
                "Branch",
                ["CS", "IT", "EEE", "AI","Mech","Civil","Chem","Metallurgy","ECE","CDS"]
            )

            section = st.text_input("Section (optional)")
            current_year = st.selectbox(
                "Current Year", [1, 2, 3, 4]
            )
            current_semester = st.selectbox(
                "Current Semester", [1, 2, 3, 4, 5, 6, 7, 8]
            )
            academic_year = st.number_input(
                "Academic Year",
                min_value=2020,
                max_value=2100,
                value=2026
            )

            new_password = st.text_input(
                "Create Password", type="password"
            )
            confirm_password = st.text_input(
                "Confirm Password", type="password"
            )

            signup_submitted = st.form_submit_button(
                "Create Student Account"
            )

        if signup_submitted:
            success, message = signup_student(
                name=name,
                username=new_username,
                email=new_email,
                password=new_password,
                confirm_password=confirm_password,
                phone=phone,
                section=section,
                branch=branch,
                current_year=current_year,
                current_semester=current_semester,
                academic_year=academic_year
            )

            if success:
                st.success(message)
                st.info("You can now log in using your username and password.")
            else:
                st.error(message)


    # Stop here until the user has logged in.
    st.stop()

# This section runs only after successful login.
user = st.session_state["user"]
role = user["role"]
student_id = user["student_id"]

with st.sidebar:
    st.write(f"Logged in as: {user['username']}")
    st.write(f"Role: {role.capitalize()}")

    if st.button("Logout", width="stretch"):
        st.session_state["authenticated"] = False
        st.session_state.pop("user", None)
        st.session_state.pop("page", None)
        st.rerun()
        st.error(message)

# ---------------- NAVIGATION ----------------

admin_pages = [
    "Admin Dashboard",
    "Student Dashboard",
    "Prediction Dashboard",
    "Student Performance",
    "Overall Academic Prediction",
    "Individual Subject Prediction",
    "Historical Comparison",
    "Student Risk Analysis",
    "Backlog Tracker",
    "Prediction Validation",
    "Student Data Review"
]

student_pages = [
    "Student Dashboard",
    "Prediction Dashboard",
    "Student Academic Details",
    "Individual Subject Prediction",
    "Overall Academic Prediction",
    "Prediction Validation"
]

pages = admin_pages if role == "admin" else student_pages

if "page" not in st.session_state:
    st.session_state.page = (
        "Admin Dashboard" if role == "admin"
        else "Student Dashboard"
    )

if st.session_state.page not in pages:
    st.session_state.page = "Student Dashboard"

nav_columns = st.columns(5)

for index, name in enumerate(pages):
    with nav_columns[index % 5]:
        if st.button(
            name,
            key=f"nav_{name}",
            width="stretch",
            type=(
                "primary"
                if st.session_state.page == name
                else "secondary"
            )
        ):
            st.session_state.page = name
            st.rerun()

page = st.session_state.page

# ---------------------------------------------------------
# RESET INDIVIDUAL PREDICTION WHEN NAVIGATING
# ---------------------------------------------------------

previous_page = st.session_state.get("previous_page")

if previous_page != page:

    if (
        previous_page == "Individual Subject Prediction"
        or page == "Individual Subject Prediction"
    ):
        keys_to_clear = [
        "individual_prediction_student",
        "individual_prediction_subject",
        "prediction_semester_{selected_student}",
        "individual_quiz1",
        "individual_quiz2",
        "individual_midsem",
        "individual_assignment",
        "individual_attendance",
        "individual_prev_sga",
        "individual_prev_sem_percentage",
        "individual_prev_total_backlogs",
        "load_marks_predict",
        "save_subject_prediction",
        "new_individual_prediction",
        "individual_factor_contributions",
        "historical_marks_distribution",
        "historical_grade_distribution",
        "predicted_marks_historical_grade_ranges",
        "individual_prediction_input",
        "individual_prediction_result",
        "individual_prediction_page_active",
        ]


        for key in keys_to_clear:
            st.session_state.pop(key, None)

        # Clear any cached semester selection for this page.
        for key in list(st.session_state.keys()):
            if key.startswith("prediction_semester_"):
                st.session_state.pop(key, None)

st.session_state["previous_page"] = page


# ---------------- PAGE ACCESS GUARD ----------------

admin_only_pages = {
    "Student Performance",
    "Historical Comparison",
    "Student Risk Analysis",
    "Backlog Tracker"
}

if role != "admin" and page in admin_only_pages:
    st.error("You don't have permission to access this page.")
    st.stop()

# ---------------- SIDEBAR ----------------

with st.sidebar:

    con = get_connection()

    cursor = con.cursor()

  
    if page == "Student Dashboard":
        st.markdown("#### Dashboard Sections")

        show_overview = st.checkbox(
            "Overview", value=True, key="sd_overview"
        )
        show_student_info = st.checkbox(
            "Student Information", value=False, key="sd_student_info"
        )
        show_academic = st.checkbox(
            "Academic Performance", value=False, key="sd_academic"
        )
        show_semester = st.checkbox(
            "Semester Performance", value=False, key="sd_semester"
        )
        show_sga = st.checkbox(
            "SGA Trend", value=False, key="sd_sga"
        )
        show_attendance = st.checkbox(
            "Attendance", value=False, key="sd_attendance"
        )



    elif page == "Prediction Dashboard":
        st.markdown("### Student Overview")

        show_pd_overview = st.checkbox(
            "Overview", value=True, key="pd_overview"
        )
        show_pd_student_info = st.checkbox(
            "Student Information", value=False, key="pd_student_info"
        )
        show_pd_academic = st.checkbox(
            "Academic Performance", value=False, key="pd_academic"
        )
        show_pd_semester = st.checkbox(
            "Semester Performance", value=False, key="pd_semester"
        )
        show_pd_sga = st.checkbox(
            "SGA Trend", value=False, key="pd_sga"
        )
        show_pd_attendance = st.checkbox(
            "Attendance", value=False, key="pd_attendance"
        )

        st.divider()
        st.markdown("### Prediction Options")

        show_overall_predictions = st.checkbox(
            "Overall Academic Predictions",
            value=False,
            key="pd_overall_predictions"
        )
        show_subject_predictions = st.checkbox(
            "Individual Subject Predictions",
            value=True,
            key="pd_subject_predictions"
        )
        show_prediction_history = st.checkbox(
            "Prediction History",
            value=True,
            key="pd_prediction_history"
        )



    elif page == "Student Performance":
        st.markdown("### Performance Options")

        show_subject_details = st.checkbox(
            "Subject Details", value=True, key="sp_details"
        )
        show_grade_points = st.checkbox(
            "Grade Points", value=False, key="sp_grade_points"
        )
        show_subject_marks = st.checkbox(
            "Subject Marks", value=True, key="sp_marks"
        )
        show_grade_distribution = st.checkbox(
            "Grade Distribution", value=False, key="sp_grade_distribution"
        )
        show_backlog_status = st.checkbox(
            "Backlog Status", value=False, key="sp_backlogs"
        )
        show_semester_performance = st.checkbox(
            "Semester Performance", value=False, key="sp_semester"
        )

    elif page == "Overall Academic Prediction":
        st.markdown("### Prediction Settings")
        st.markdown("#### Student & Inputs")

        show_student_details = st.checkbox(
            "Student Details", value=True, key="oap_student_details"
        )
        show_input_values = st.checkbox(
            "Input Values", value=True, key="oap_input_values"
        )

        st.divider()
        st.markdown("#### Prediction Results")

        show_prediction_summary = st.checkbox(
            "Semester Prediction Summary", value=True, key="oap_summary"
        )
        show_predicted_subjects = st.checkbox(
            "Predicted Subject Results & Feedback",
            value=False, key="oap_subjects"
        )
        show_prediction_explanation = st.checkbox(
            "Prediction Explanation", value=False,
            key="oap_prediction_explanation"
        )
        show_attendance_tracker = st.checkbox(
            "Attendance Tracker", value=False, key="oap_attendance"
        )
        show_historical_comparison = st.checkbox(
            "Historical Performance Comparison",
            value=False, key="oap_history"
        )
        show_subject_advice = st.checkbox(
            "Subject-Specific Advice", value=False, key="oap_advice"
        )
        show_academic_risk = st.checkbox(
            "Overall Academic Risk & Recommended Actions",
            value=False, key="oap_risk"
        )
        show_action_priorities = st.checkbox(
            "Subject Action Priorities", value=False, key="oap_priorities"
        )

    elif page == "Individual Subject Prediction":
        st.markdown("### Prediction Options")

        show_prediction_input = st.checkbox(
            "Prediction Input", value=True, key="isp_prediction_input"
        )
        show_prediction_confidence = st.checkbox(
            "Prediction Confidence", value=True,
            key="isp_prediction_confidence"
        )
        show_estimated_grade = st.checkbox(
            "Estimated Grade & Reliability", value=True,
            key="isp_estimated_grade"
        )
        show_factor_explanation = st.checkbox(
            "Factor Contribution Explanation", value=False,
            key="isp_factor_explanation"
        )
        show_historical_metrics = st.checkbox(
            "Historical Performance", value=False,
            key="isp_historical_metrics"
        )
        show_grade_cutoffs = st.checkbox(
            "Subject-wise Grade Cutoffs", value=False,
            key="isp_grade_cutoffs"
        )
        show_grade_ranges = st.checkbox(
            "Predicted Marks vs Historical Grade Ranges",
            value=False, key="isp_grade_ranges"
        )
        show_prediction_summary = st.checkbox(
            "Prediction Summary", value=False,
            key="isp_prediction_summary"
        )
        show_uncertainty = st.checkbox(
            "Prediction Uncertainty", value=False,
            key="isp_uncertainty"
        )
        show_model_validation = st.checkbox(
            "Model Validation Summary", value=True,
            key="isp_model_validation"
        )

    elif page == "Historical Comparison":
        st.markdown("#### Comparison Options")

        show_sga_trend = st.checkbox(
            "SGA Trend", value=True, key="hc_sga"
        )
        show_cga_tracker = st.checkbox(
            "CGA Tracker", value=True, key="hc_cga"
        )
        show_backlog_history = st.checkbox(
            "Backlog History", value=True, key="hc_backlogs"
        )
        show_semester_table = st.checkbox(
            "Semester Details", value=False, key="hc_semester_table"
        )

    elif page == "Student Risk Analysis":
        st.markdown("#### Risk Options")

        show_risk_summary = st.checkbox(
            "Student Risk Summary", value=True, key="risk_summary"
        )
        show_semester_indicators = st.checkbox(
            "Semester Risk Indicators", value=True,
            key="semester_risk_indicators"
        )

    elif page == "Backlog Tracker":
        st.markdown("#### Details of Students who have backlogs.")

    elif page == "Prediction Validation":
        st.markdown("#### Validation Options")

        show_validation_filters = st.checkbox(
            "Validation Status Filter", value=True,
            key="show_validation_filter"
        )
        show_validation_form = st.checkbox(
            "Prediction Validation Form", value=True,
            key="show_validation_form"
        )
        show_validation_results = st.checkbox(
            "Prediction Validation Results", value=True,
            key="show_validation_results"
        )
    


# ---------------- DATABASE CONNECTION ----------------


con = get_connection()

cursor = con.cursor()

# Student-submitted academic data is kept separate from official records
# until an administrator approves it.
cursor.execute("""
CREATE TABLE IF NOT EXISTS student_academic_submissions (
    submission_id INTEGER GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    student_id VARCHAR(20) NOT NULL,
    student_name VARCHAR(255),
    branch VARCHAR(100),
    subject_code VARCHAR(100) NOT NULL,
    subject_name VARCHAR(255),
    academic_year INT NOT NULL,
    semester INT NOT NULL,
    quiz1 DECIMAL(7,2),
    quiz2 DECIMAL(7,2),
    midsem DECIMAL(7,2),
    assignment DECIMAL(7,2),
    classes_attended INT,
    classes_held INT,
    attendance_percentage DECIMAL(6,2),
    prev_sga DECIMAL(5,2),
    prev_sem_percentage DECIMAL(6,2),
    prev_total_backlogs INT,
    status VARCHAR(30) NOT NULL DEFAULT 'Pending',
    admin_notes TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP NULL DEFAULT NULL,
    reviewed_by VARCHAR(255)
)
""")
con.commit()

query = """
SELECT
    student_id,
    name,
    branch,
    current_year,
    current_semester
FROM students
"""



if role == "student":
    query += " WHERE student_id = %s"
    cursor.execute(query, (student_id,))
    students_df = pd.DataFrame(
        cursor.fetchall(),
        columns=[column[0] for column in cursor.description]
    )
else:
    cursor.execute(query)
    students_df = pd.DataFrame(
        cursor.fetchall(),
        columns=[column[0] for column in cursor.description]
    )

if page == "Admin Dashboard" and role == "admin":
    st.title("Admin Dashboard")
    st.write("Welcome to the Student Academic Performance System.")

    col1, col2, col3, col4 = st.columns(4)

    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]
    col1.metric("Total Students", total_students)

    try:
        cursor.execute("""
            SELECT COUNT(*) FROM user_accounts
            WHERE is_active = TRUE
        """)
        active_accounts = cursor.fetchone()[0]
        col2.metric("Active Accounts", active_accounts)

        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        col3.metric("Subject Predictions", total_predictions)

        cursor.execute(
            "SELECT COUNT(*) FROM overall_prediction_history"
        )
        total_overall = cursor.fetchone()[0]
        col4.metric("Overall Predictions", total_overall)

    except Exception as e:
        st.error(f"Could not load dashboard statistics: {e}")

    st.subheader("Registered Students")

    cursor.execute("""
        SELECT student_id, name, branch, current_year, current_semester
        FROM students
        ORDER BY student_id
    """)

    admin_students_df = pd.DataFrame(
        cursor.fetchall(),
        columns=[column[0] for column in cursor.description]
    )

    if not admin_students_df.empty:
        st.dataframe(
            admin_students_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No students are registered yet.")


if page == "Student Dashboard":
    st.title("Student Dashboard")

    cursor.execute(
        """
        SELECT
            p.prediction_id,
            p.student_id,
            c.subject_code,
            c.subject_name,
            pf.actual_marks,
            c.maximum_marks,
            pf.actual_grade,
            pf.feedback_date
        FROM prediction_feedback pf
        JOIN predictions p
            ON pf.prediction_id = p.prediction_id
        LEFT JOIN curriculum c
            ON p.subject_id = c.subject_id
        WHERE p.student_id = %s
        ORDER BY pf.feedback_date DESC
        """,
        (student_id,)
    )

    validated_results = cursor.fetchall()
    
    if role == "student":
        selected_student = student_id
    else:
        student_ids = students_df["student_id"].astype(str).tolist()

        if not student_ids:
            st.warning("No students are available.")
            st.stop()

        selected_student = st.selectbox(
            "Select Student",
            student_ids,
            key="dashboard_student_select"
        )

    validated_query = """
        SELECT
            p.prediction_id,
            p.student_id,
            c.subject_code,
            c.subject_name,
            pf.actual_marks,
            c.maximum_marks,
            pf.actual_grade,
            pf.feedback_date AS validation_date
        FROM prediction_feedback pf
        INNER JOIN predictions p
            ON pf.prediction_id = p.prediction_id
        LEFT JOIN curriculum c
            ON p.subject_id = c.subject_id
        WHERE p.student_id = %s
        ORDER BY pf.feedback_date DESC
    """

    validated_results = pd.read_sql(
        validated_query,
        con,
        params=(str(selected_student),)
    )

    st.subheader("Validated Actual Marks")

    if not validated_results.empty:
        st.dataframe(
            validated_results.drop(
                columns=["prediction_id", "student_id"]
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No validated actual marks are available for this student yet.")


    student_row = students_df.loc[
        students_df["student_id"].astype(str) == str(selected_student)
    ]

    if student_row.empty:
        st.error("Selected student was not found.")
        st.stop()

    current_year_value = student_row.iloc[0]["current_year"]

    student_year = (
        int(current_year_value)
        if pd.notna(current_year_value)
        else None
    )
    semester_value = student_row.iloc[0]["current_semester"]

    if pd.isna(semester_value) or str(semester_value).strip() == "":
        current_semester = 8
    else:
        current_semester = int(semester_value)


    query = """
        SELECT
            a.record_id,
            a.subject_code,
            COALESCE(c.subject_name, a.subject_code) AS subject_name,
            a.academic_year,
            a.semester,
            a.marks_obtained,
            a.maximum_marks,
            a.grade,
            a.grade_point,
            a.attendance_percentage,
            a.backlog
        FROM academic_records a
        LEFT JOIN curriculum c
            ON a.subject_id = c.subject_id
        WHERE a.student_id = %s
        ORDER BY a.academic_year, a.semester, a.subject_code
    """

    academic_df = pd.read_sql(
        query,
        con,
        params=(str(selected_student),)
    )


    st.subheader("Saved Subject Predictions")

    saved_predictions_query = """
        SELECT
            p.prediction_id,
            c.subject_code,
            c.subject_name,
            p.prediction_type,
            p.predicted_marks,
            p.predicted_grade,
            p.created_at
        FROM predictions p
        LEFT JOIN curriculum c
            ON p.subject_id = c.subject_id
        WHERE p.student_id = %s
        ORDER BY p.created_at DESC, p.prediction_id DESC
    """

    saved_predictions_df = pd.read_sql(
        saved_predictions_query,
        con,
        params=(str(selected_student),)
    )

    if saved_predictions_df.empty:
        st.info("No saved predictions are available for this student yet.")
    else:
        st.dataframe(
            saved_predictions_df.drop(columns=["prediction_id"]),
            use_container_width=True,
            hide_index=True
        )

    if show_overview:
        st.subheader("Overview")

        if not academic_df.empty:
            avg_marks = academic_df["marks_obtained"].mean()
            avg_gp = academic_df["grade_point"].mean()
            total_subjects = academic_df["subject_code"].nunique()
            total_backlogs = academic_df["backlog"].fillna(0).sum()
            avg_attendance = academic_df["attendance_percentage"].mean()

            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "Average Marks",
                f"{avg_marks:.2f}" if pd.notna(avg_marks) else "N/A"
            )
            col2.metric(
                "Average Grade Point",
                f"{avg_gp:.2f}" if pd.notna(avg_gp) else "N/A"
            )
            col3.metric("Subjects", total_subjects)
            col4.metric("Backlogs", int(total_backlogs))
            col5.metric(
                "Average Attendance",
                f"{avg_attendance:.2f}%"
                if pd.notna(avg_attendance) else "N/A"
            )
        else:
            st.info("No academic records are available for this student.")

    student_info = student_row.iloc[0]

    if show_student_info:
        st.subheader("Student Information")
        st.dataframe(student_info, width="stretch")

    if show_academic:
        st.subheader("Academic Performance")

        if not academic_df.empty:
            display_df = academic_df.drop(columns=["record_id"])
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No academic records are available for this student.")

    query = """
    SELECT
        academic_year,
        semester,
        sga,
        semester_percentage,
        total_backlogs,
        semester_status
    FROM semester_results
    WHERE student_id = %s
    ORDER BY academic_year, semester
    """

    semester_df = pd.read_sql(
        query, con, params=(selected_student,)
    )
    semester_df = semester_df.fillna(0)

    if show_semester:
        st.subheader("Semester Performance")
        st.dataframe(semester_df, width="stretch")

    if show_sga:
        st.subheader("SGA Trend")

        if not semester_df.empty:
            chart_df = semester_df.copy()
            chart_df["Semester_Label"] = (
                chart_df["academic_year"].astype(str)
                + " - Sem "
                + chart_df["semester"].astype(str)
            )

            fig = px.line(
                chart_df,
                x="Semester_Label",
                y="sga",
                markers=True,
                labels={
                    "Semester_Label": "Semester",
                    "sga": "SGA"
                }
            )

            fig.update_yaxes(range=[0, 10], dtick=1)
            st.plotly_chart(fig, width="stretch")

            latest_semester = semester_df.iloc[-1]
            latest_sga = latest_semester["sga"]

            previous_sga = (
                semester_df.iloc[-2]["sga"]
                if len(semester_df) > 1 else None
            )

            sga_change = (
                latest_sga - previous_sga
                if previous_sga is not None else None
            )

            col1, col2 = st.columns(2)
            col1.metric(
                "Latest SGA",
                f"{latest_sga:.2f}",
                f"{sga_change:+.2f}" if sga_change is not None else None
            )
            col2.metric(
                "Latest Semester",
                f"Semester {int(latest_semester['semester'])}"
            )
        else:
            st.info("No semester results are available for this student.")

    if show_attendance:
        st.subheader("Attendance")

        if not academic_df.empty:
            attendance = academic_df["attendance_percentage"].mean()
            attendance_status = (
                "Eligible" if attendance >= 75
                else "Attendance Shortage"
            )

            col1, col2 = st.columns(2)
            col1.metric("Average Attendance", f"{attendance:.2f}%")
            col2.metric("Status", attendance_status)
        else:
            st.info("No attendance records are available.")

    st.divider()
    st.subheader("Download Evaluation Report")

    report_csv = error_by_range.to_csv().encode("utf-8")

    st.download_button(
        label="Download Mark-Range Evaluation (CSV)",
        data=report_csv,
        file_name="error_by_mark_range.csv",
        mime="text/csv"
    )

    st.subheader("Prediction Error Across Mark Ranges")
    st.bar_chart(error_by_range["MAE"])

    st.caption(
        "Mean absolute prediction error by predicted-mark range. "
        "Lower values indicate smaller errors on average."
    )

    st.subheader("Model Evaluation Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"{model_metrics['test mae']:.2f} marks")
    col2.metric("Root Mean Squared Error", f"{model_metrics['test rmse']:.2f} marks")
    col3.metric("R² Score", f"{model_metrics['test r2']:.4f}")

    st.caption(
        "Evaluation results are based on the generated synthetic dataset "
        "and may not represent performance on real student records."
    )





elif page == "Prediction Dashboard":
    st.title("Prediction Dashboard")

    st.caption(
        "Review saved individual-subject predictions and overall academic predictions. "
        "Generating predictions remains available on the existing prediction pages."
    )
    st.divider()

    prediction_student_ids = students_df["student_id"].tolist()
    if not prediction_student_ids:
        st.info("No students are available to display predictions.")
        st.stop()

    if role == "student":
        prediction_student = student_id
        st.text_input(
            "Student ID",
            value=str(prediction_student),
            disabled=True,
            key="prediction_dashboard_student_id"
        )
    else:
        prediction_student = st.selectbox(
            "Select Student",
            prediction_student_ids,
            key="prediction_dashboard_student"
        )



    # ---------------- PREDICTION-BASED STUDENT OVERVIEW ----------------
    prediction_student_info = students_df[
        students_df["student_id"] == prediction_student
    ]

    overall_history_query = """
        SELECT
            prediction_id,
            academic_year,
            semester,
            predicted_sga,
            estimated_cg,
            predicted_backlog_count,
            backlog_risk,
            performance_category,
            overall_academic_risk
        FROM overall_prediction_history
        WHERE student_id = %s
        ORDER BY prediction_id DESC
    """

    prediction_history_df = pd.read_sql(
        overall_history_query,
        con,
        params=(prediction_student,)
    )

    latest_overall = (
        prediction_history_df.iloc[0]
        if not prediction_history_df.empty
        else None
    )

    prediction_subjects_df = pd.DataFrame()

    if latest_overall is not None:
        prediction_subjects_df = pd.read_sql(
            """
            SELECT
                subject_code,
                subject_name,
                credits,
                predicted_marks,
                estimated_grade,
                attendance_percentage,
                backlog_risk,
                historical_average,
                performance_vs_history
            FROM overall_prediction_subjects
            WHERE prediction_id = %s
            ORDER BY subject_code
            """,
            con,
            params=(int(latest_overall["prediction_id"]),)
        )

    if show_pd_overview:
        st.subheader("Latest Prediction Overview")

        if latest_overall is None:
            st.info(
                "No saved overall prediction exists for this student yet. "
                "Generate an overall academic prediction to view this overview."
            )
        else:
            col1, col2, col3, col4, col5 = st.columns(5)

            col1.metric(
                "Predicted SGA",
                f"{float(latest_overall['predicted_sga'] or 0):.2f}"
            )
            col2.metric(
                "Estimated CG",
                f"{float(latest_overall['estimated_cg'] or 0):.2f}"
            )
            col3.metric(
                "Predicted Backlogs",
                int(latest_overall["predicted_backlog_count"] or 0)
            )
            col4.metric(
                "Backlog Risk",
                str(latest_overall["backlog_risk"] or "N/A")
            )
            col5.metric(
                "Predicted Subjects",
                len(prediction_subjects_df)
            )

            st.caption(
                f"Latest prediction: Academic Year "
                f"{latest_overall['academic_year']}, "
                f"Semester {latest_overall['semester']}"
            )

    if show_pd_student_info:
        st.subheader("Student Information")
        if not prediction_student_info.empty:
            st.dataframe(
                prediction_student_info,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Student information is unavailable.")

    if show_pd_academic:
        st.subheader("Predicted Academic Performance")
        if not prediction_subjects_df.empty:
            st.dataframe(
                prediction_subjects_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No predicted subject results are available. "
                "Generate an overall academic prediction first."
            )

    if show_pd_semester:
        st.subheader("Predicted Semester Performance")
        if latest_overall is not None:
            semester_display = pd.DataFrame([{
                "academic_year": latest_overall["academic_year"],
                "semester": latest_overall["semester"],
                "predicted_sga": latest_overall["predicted_sga"],
                "estimated_cg": latest_overall["estimated_cg"],
                "predicted_backlog_count":
                    latest_overall["predicted_backlog_count"],
                "backlog_risk": latest_overall["backlog_risk"],
                "performance_category":
                    latest_overall["performance_category"],
                "overall_academic_risk":
                    latest_overall["overall_academic_risk"]
            }])

            st.dataframe(
                semester_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Generate an overall prediction to view predicted semester performance.")

    if show_pd_sga:
        st.subheader("Predicted SGA Trend")

        if not prediction_history_df.empty:
            sga_chart_df = prediction_history_df.dropna(
                subset=["predicted_sga"]
            ).copy()

            sga_chart_df = sga_chart_df.sort_values("prediction_id")

            if not sga_chart_df.empty:
                sga_chart_df["Semester_Label"] = (
                    sga_chart_df["academic_year"].astype(str)
                    + " - Sem "
                    + sga_chart_df["semester"].astype(str)
                )

                fig = px.line(
                    sga_chart_df,
                    x="Semester_Label",
                    y="predicted_sga",
                    markers=True,
                    labels={
                        "Semester_Label": "Semester",
                        "predicted_sga": "Predicted SGA"
                    }
                )
                fig.update_yaxes(range=[0, 10], dtick=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predicted SGA values are available.")
        else:
            st.info("Generate an overall prediction to view the predicted SGA trend.")

    if show_pd_attendance:
        st.subheader("Predicted Subject Attendance")

        if not prediction_subjects_df.empty:
            attendance_df = prediction_subjects_df[
                ["subject_code", "subject_name", "attendance_percentage"]
            ].copy()

            st.dataframe(
                attendance_df,
                use_container_width=True,
                hide_index=True
            )

            valid_attendance = pd.to_numeric(
                attendance_df["attendance_percentage"],
                errors="coerce"
            ).dropna()

            if not valid_attendance.empty:
                avg_attendance = valid_attendance.mean()
                col1, col2 = st.columns(2)
                col1.metric(
                    "Average Attendance in Prediction",
                    f"{avg_attendance:.2f}%"
                )
                col2.metric(
                    "Attendance Threshold",
                    "75%"
                )

            st.caption(
                "Attendance values are those stored with the prediction; "
                "they may be historical inputs rather than forecasted attendance."
            )
        else:
            st.info(
                "No predicted subject results are available. "
                "Generate an overall academic prediction first."
            )


    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button(
            "Generate Overall Academic Prediction",
            key="prediction_dashboard_go_overall",
            use_container_width=True
        ):
            st.session_state.page = "Overall Academic Prediction"
            st.rerun()
    with action_col2:
        if st.button(
            "Generate Individual Subject Prediction",
            key="prediction_dashboard_go_individual",
            use_container_width=True
        ):
            st.session_state.page = "Individual Subject Prediction"
            st.rerun()

    if show_overall_predictions:
        st.subheader("Saved Overall Academic Predictions")
        try:
            overall_history_df = pd.read_sql(
                """
                SELECT
                    prediction_id,
                    academic_year,
                    semester,
                    predicted_sga,
                    estimated_cg,
                    predicted_backlog_count,
                    backlog_risk,
                    performance_category,
                    overall_academic_risk
                FROM overall_prediction_history
                WHERE student_id = %s
                ORDER BY prediction_id DESC
                """,
                con,
                params=(prediction_student,)
            )

            if overall_history_df.empty:
                st.info("No saved overall academic predictions found for this student.")
            else:
                latest_overall = overall_history_df.iloc[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted SGA", f"{float(latest_overall['predicted_sga'] or 0):.2f}")
                m2.metric("Estimated CG", f"{float(latest_overall['estimated_cg'] or 0):.2f}")
                m3.metric("Predicted Backlogs", int(latest_overall["predicted_backlog_count"] or 0))
                m4.metric("Backlog Risk", str(latest_overall["backlog_risk"] or "N/A"))

                st.write(
                    f"**Latest prediction:** Academic Year {latest_overall['academic_year']}, "
                    f"Semester {latest_overall['semester']}"
                )
                st.write(f"**Performance Category:** {latest_overall['performance_category'] or 'N/A'}")
                st.write(f"**Overall Academic Risk:** {latest_overall['overall_academic_risk'] or 'N/A'}")

                if show_prediction_history:
                    st.markdown("#### Overall Prediction History")
                    st.dataframe(
                        overall_history_df.drop(columns=["prediction_id"]),
                        width="stretch",
                        hide_index=True
                    )

                st.markdown("#### Predicted Subject Results")
                latest_prediction_id = int(latest_overall["prediction_id"])
                overall_subjects_df = pd.read_sql(
                    """
                    SELECT
                        subject_code,
                        subject_name,
                        credits,
                        predicted_marks,
                        estimated_grade,
                        attendance_percentage,
                        backlog_risk,
                        historical_average,
                        performance_vs_history
                    FROM overall_prediction_subjects
                    WHERE prediction_id = %s
                    ORDER BY subject_code
                    """,
                    con,
                    params=(latest_prediction_id,)
                )
                if overall_subjects_df.empty:
                    st.info("No subject-level results were saved for the latest overall prediction.")
                else:
                    st.dataframe(overall_subjects_df, width="stretch", hide_index=True)

        except Exception as overall_dashboard_error:
            st.warning(
                "Overall academic prediction history could not be loaded: "
                f"{overall_dashboard_error}"
            )

    if show_subject_predictions:
        st.divider()
        st.subheader("Saved Individual Subject Predictions")
        try:
            saved_predictions_df = pd.read_sql(
                """
                SELECT
                    p.prediction_id,
                    c.subject_code,
                    c.subject_name,
                    p.prediction_type,
                    p.predicted_marks,
                    p.predicted_grade,
                    p.lower_bound,
                    p.upper_bound,
                    p.prediction_confidence,
                    p.created_at
                FROM predictions p
                LEFT JOIN curriculum c ON c.subject_id = p.subject_id
                WHERE p.student_id = %s
                ORDER BY p.created_at DESC, p.prediction_id DESC
                """,
                con,
                params=(prediction_student,)
            )

            if saved_predictions_df.empty:
                st.info("No saved individual subject predictions found for this student.")
            else:
                st.dataframe(
                    saved_predictions_df.drop(columns=["prediction_id"]),
                    width="stretch",
                    hide_index=True
                )
                latest_subject_prediction = saved_predictions_df.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Latest Predicted Marks",
                    f"{float(latest_subject_prediction['predicted_marks'] or 0):.2f}"
                )
                m2.metric(
                    "Latest Predicted Grade",
                    str(latest_subject_prediction["predicted_grade"] or "N/A")
                )
                confidence = latest_subject_prediction["prediction_confidence"]
                m3.metric(
                    "Prediction Confidence",
                    f"{float(confidence):.1f}%" if pd.notna(confidence) else "N/A"
                )
        except Exception as subject_dashboard_error:
            st.warning(
                "Individual subject prediction history could not be loaded: "
                f"{subject_dashboard_error}"
            )


if page == "Student Performance":
    st.title("Student Performance")
    st.caption(
        "Explore subject-wise results, attendance, grades, and semester outcomes."
    )
    st.divider()

    
    student_ids = students_df["student_id"].astype(str).tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_ids,
        key="performance_student"
    )

    student_row = students_df.loc[
        students_df["student_id"].astype(str) == str(selected_student)
    ]

    if student_row.empty:
        st.error("Selected student was not found.")
        st.stop()

    student_info = student_row.iloc[0]

    semester_value = student_row.iloc[0]["current_semester"]

    if pd.isna(semester_value) or str(semester_value).strip() == "":
        current_semester = 8
    else:
        current_semester = int(semester_value)

    student_semester = st.selectbox(
        "Select Semester",
        options=list(range(1, 9)),
        index=current_semester - 1,
        key=f"performance_semester_{selected_student}"
    )

    query = """
        SELECT
            a.subject_code,
            c.subject_name,
            a.marks_obtained,
            a.maximum_marks,
            a.grade,
            a.grade_point,
            a.attendance_percentage,
            a.backlog
        FROM academic_records a
        JOIN curriculum c
            ON a.subject_id = c.subject_id
        WHERE a.student_id = %s
          AND a.academic_year = %s
          AND a.semester = %s
        ORDER BY a.subject_code
    """

    academic_year_df = pd.read_sql(
        """
        SELECT academic_year
        FROM academic_records
        WHERE student_id = %s
          AND semester = %s
        ORDER BY academic_year DESC
        LIMIT 1
        """,
        con,
        params=(str(selected_student), int(student_semester))
    )

    if academic_year_df.empty:
        st.warning(
            "No actual performance records were found for this semester."
        )
        st.stop()

    curriculum_year = int(academic_year_df.iloc[0]["academic_year"])

    academic_df = pd.read_sql(
        query,
        con,
        params=(
            str(selected_student),
            curriculum_year,
            int(student_semester)
        )
    )

    if academic_df.empty:
        st.info("No performance records are available for this semester.")
        st.stop()


    academic_df = pd.DataFrame(
        cursor.fetchall(),
        columns=[
            "subject_code",
            "subject_name",
            "marks_obtained",
            "maximum_marks",
            "grade",
            "grade_point",
            "attendance_percentage",
            "backlog"
        ]
    )

    if not academic_df.empty:
        actual_marks = pd.to_numeric(
            academic_df["marks_obtained"],
            errors="coerce"
        )

        actual_maximum = pd.to_numeric(
            academic_df["maximum_marks"],
            errors="coerce"
        )

        valid_rows = (
            actual_marks.notna()
            & actual_maximum.notna()
            & (actual_maximum > 0)
        )

        actual_percentage = (
            (
                actual_marks[valid_rows].sum()
                / actual_maximum[valid_rows].sum()
            ) * 100
            if valid_rows.any()
            else None
        )

        actual_total_backlogs = int(
            pd.to_numeric(
                academic_df["backlog"],
                errors="coerce"
            ).fillna(0).sum()
        )
    else:
        actual_percentage = None
        actual_total_backlogs = None

    query = """
        SELECT
            a.subject_code,
            c.subject_name,
            a.marks_obtained,
            a.maximum_marks,
            a.grade,
            a.grade_point,
            a.attendance_percentage,
            a.backlog
        FROM academic_records a
        JOIN curriculum c
            ON a.subject_id = c.subject_id
        WHERE a.student_id = %s
        AND a.semester = %s
        ORDER BY a.subject_code
    """



    with con.cursor() as cursor:
        cursor.execute(query, (selected_student, student_semester))
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    performance_df = pd.DataFrame(rows, columns=columns)

    performance_df["attendance_status"] = performance_df[
        "attendance_percentage"
    ].apply(lambda x: "Safe" if x >= 75 else "Shortage")

    performance_df["performance_category"] = performance_df[
        "marks_obtained"
    ].apply(
        lambda x:
            "Amazing" if x >= 85
            else "Decent" if x >= 70
            else "Average" if x >= 55
            else "Below Par" if x >= 40
            else "Very Bad"
    )

    styled_df = performance_df.style.map(
        lambda x:
            "background-color: #00b050; color: black; font-weight: bold;"
            if x == "Amazing"
            else "background-color: #70ad47; color: black; font-weight: bold;"
            if x == "Decent"
            else "background-color: #a9d18e; color: black; font-weight: bold;"
            if x == "Average"
            else "background-color: #ff9999; color: black; font-weight: bold;"
            if x == "Below Par"
            else "background-color: #ff0000; color: black; font-weight: bold;"
            if x == "Very Bad"
            else "",
        subset=["performance_category"]
    )

    styled_df = styled_df.map(
        lambda x:
            "background-color: #ff0000; color: black; font-weight: bold;"
            if isinstance(x, (int, float)) and x < 75
            else "",
        subset=["attendance_percentage"]
    )

    styled_df = styled_df.map(
        lambda x:
            "background-color: #ff0000; color: black; font-weight: bold;"
            if x == "Shortage"
            else "",
        subset=["attendance_status"]
    )

    st.markdown("### Performance Overview")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Subjects", len(performance_df))

        col2.metric(
            "Average Marks",
            f"{performance_df['marks_obtained'].mean():.1f}/100"
        )

        col3.metric(
            "Average Attendance",
            f"{performance_df['attendance_percentage'].mean():.1f}%"
        )

        backlog_count = performance_df["backlog"].sum()

        col4.metric("Backlogs", int(backlog_count))

    if show_subject_details:
        st.markdown("### Subject Details")

        with st.container(border=True):
            st.dataframe(styled_df, width="stretch")

    if not performance_df.empty:
        performance_df["display_grade_point"] = performance_df.apply(
            lambda row: 10
            if str(row["grade"]).strip().upper() in ["FF", "FA"]
            else row["grade_point"],
            axis=1
        )

        bar_colors = [
            "#FA6363"
            if str(grade).strip().upper() in ["FF", "FA"]
            else "#636EFA"
            for grade in performance_df["grade"]
        ]

        if show_grade_points:
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=performance_df["subject_code"],
                    y=performance_df["display_grade_point"],
                    marker_color=bar_colors,
                    customdata=performance_df[
                        ["subject_name", "grade", "grade_point"]
                    ],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Subject: %{customdata[0]}<br>"
                        "Grade: %{customdata[1]}<br>"
                        "Grade Point: %{customdata[2]}<extra></extra>"
                    )
                )
            )

            fig.update_yaxes(
                range=[0, 10],
                dtick=1,
                title="Grade Point"
            )

            fig.update_xaxes(title="Subject")
            fig.update_layout(showlegend=False)

            st.markdown("### Subject-wise Grade Point")
            st.caption(
                "Compare grade points across subjects. "
                "Failed subjects are highlighted in red."
            )
            st.plotly_chart(fig, width="stretch")

        if show_subject_marks:
            fig = px.bar(
                performance_df,
                x="subject_code",
                y="marks_obtained",
                hover_data=["subject_name", "grade", "grade_point"],
                labels={
                    "subject_code": "Subject",
                    "marks_obtained": "Marks"
                }
            )

            fig.update_layout(
                xaxis_title="Subject",
                yaxis_title="Marks out of 100",
                yaxis=dict(range=[0, 100], dtick=10),
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.markdown("### Subject-wise Marks")
            st.caption("Compare marks obtained in each subject.")
            st.plotly_chart(fig, width="stretch")

        if show_grade_distribution:
            grade_counts = (
                performance_df["grade"]
                .value_counts()
                .reset_index()
            )

            grade_counts.columns = ["grade", "count"]

            fig = px.pie(
                grade_counts,
                names="grade",
                values="count",
                title="Grade Distribution"
            )

            st.plotly_chart(fig, width="stretch")

        if show_backlog_status:
            if backlog_count > 0:
                st.warning(
                    f"This student currently had {int(backlog_count)} backlog(s)."
                )
            else:
                st.success("No backlogs recorded.")

        query = """
            SELECT
                academic_year,
                semester,
                sga,
                semester_percentage,
                total_backlogs,
                semester_status
            FROM semester_results
            WHERE student_id = %s
            AND result_type = 'Actual'
            ORDER BY academic_year, semester
        """

        semester_df = pd.read_sql(
            query,
            con,
            params=(str(selected_student),)
        )

        if show_semester:
            st.subheader("Semester Performance")

            if not semester_df.empty:
                st.dataframe(
                    semester_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    f"No actual semester results found for student {selected_student}."
                )


if "show_overall_dashboard" not in st.session_state:
    st.session_state["show_overall_dashboard"] = False

if "saved_overall_prediction" not in st.session_state:
    st.session_state["saved_overall_prediction"] = None

if "overall_prediction_generated" not in st.session_state:
    st.session_state["overall_prediction_generated"] = False


if page == "Overall Academic Prediction":

    grade_points = {
        "AA": 10,
        "AB": 9,
        "BB": 8,
        "BC": 7,
        "CC": 6,
        "CD": 5,
        "DD": 4,
        "FF": 0,
        "FA": 0
    }

    st.title("SGA Prediction")

    student_ids = students_df["student_id"].astype(str).tolist()

    selected_student = st.selectbox(
        "Select Student",
        student_ids
    )

    student_row = students_df.loc[
        students_df["student_id"].astype(str) == str(selected_student)
    ]

    if not student_row.empty:
        student_info = student_row.iloc[0]
        student_year = int(student_info["current_year"])
        student_branch = student_info["branch"]

        current_semester = int(student_info["current_semester"])
        student_semester = st.selectbox(
            "Select Semester",
            options=list(range(1, 9)),
            index=max(0, min(current_semester - 1, 7)),
            key=f"overall_prediction_semester_{selected_student}"
        )

        student_year = ((student_semester - 1) // 2) + 1

    else:
        st.error("Selected student was not found in the student list.")
        st.stop()

    query = """
        SELECT
            a.subject_code,
            c.subject_name,
            a.marks_obtained,
            a.maximum_marks,
            a.grade,
            a.grade_point,
            a.attendance_percentage,
            a.backlog
        FROM academic_records a
        JOIN curriculum c
            ON a.subject_id = c.subject_id
        WHERE a.student_id = %s
          AND a.academic_year = %s
          AND a.semester = %s
        ORDER BY a.subject_code
    """

    academic_df = pd.read_sql(
        query,
        con,
        params=(
            str(selected_student),
            student_year,
            student_semester
        )
    )

    current_semester_query = """
        SELECT academic_year, semester
        FROM academic_records
        WHERE student_id = %s
        ORDER BY academic_year DESC, semester DESC
        LIMIT 1
    """

    current_semester_df = pd.read_sql(
        current_semester_query,
        con,
        params=(selected_student,)
    )

    if not current_semester_df.empty:
        default_semester = int(current_semester_df.iloc[0]["semester"])
        default_academic_year = int(current_semester_df.iloc[0]["academic_year"])
    else:
        approved_term = pd.read_sql(
            """
            SELECT academic_year, semester
            FROM student_academic_submissions
            WHERE student_id = %s AND status = 'Approved'
            ORDER BY academic_year DESC, semester DESC
            LIMIT 1
            """,
            con,
            params=(selected_student,)
        )

        if approved_term.empty:
            st.warning("No official records or approved student submissions were found.")
            st.stop()

        default_semester = int(approved_term.iloc[0]["semester"])
        default_academic_year = int(approved_term.iloc[0]["academic_year"])

    academic_year_query = """
        SELECT academic_year
        FROM academic_records
        WHERE student_id = %s
        AND semester = %s
        ORDER BY academic_year DESC
        LIMIT 1
    """

    academic_year_df = pd.read_sql(
        academic_year_query,
        con,
        params=(selected_student, student_semester)
    )

    if academic_year_df.empty:
        approved_year = pd.read_sql(
            """
            SELECT academic_year
            FROM student_academic_submissions
            WHERE student_id = %s AND semester = %s AND status = 'Approved'
            ORDER BY academic_year DESC
            LIMIT 1
            """,
            con,
            params=(selected_student, student_semester)
        )
        if approved_year.empty:
            st.error("Could not determine the academic year from official or approved records.")
            st.stop()
        curriculum_year = int(approved_year.iloc[0]["academic_year"])
    else:
        curriculum_year = int(academic_year_df.iloc[0]["academic_year"])

    if show_student_details:
        st.write(
            f"Branch: {student_branch} | "
            f"Year: {student_year} | "
            f"Semester: {student_semester}"
        )

    subjects_query = """
        SELECT
            subject_code,
            MAX(subject_name) AS subject_name,
            MAX(credits) AS credits
        FROM curriculum
        WHERE UPPER(TRIM(branch)) = UPPER(TRIM(%s))
        AND semester = %s
        GROUP BY subject_code
        ORDER BY subject_code
    """

    subjects_df = pd.read_sql(
        subjects_query,
        con,
        params=(student_branch, student_semester)
    )

    if subjects_df.empty:
        st.warning("No curriculum subjects found for this student and semester.")
        st.stop()

    all_subjects_query = """
    SELECT
        a.subject_code,
        a.quiz1,
        a.quiz2,
        a.midsem,
        a.assignment,
        a.classes_attended,
        a.classes_held,
        a.attendance_percentage,
        sr.sga AS prev_sga,
        sr.semester_percentage AS prev_sem_percentage,
        sr.total_backlogs AS prev_total_backlogs
    FROM academic_records a
    LEFT JOIN semester_results sr
        ON sr.student_id = a.student_id
        AND (
            sr.academic_year < a.academic_year
            OR (
                sr.academic_year = a.academic_year
                AND sr.semester < a.semester
            )
        )
    WHERE a.student_id = %s
    AND a.academic_year = %s
    AND a.semester = %s
    AND NOT EXISTS (
        SELECT 1
        FROM semester_results sr2
        WHERE sr2.student_id = sr.student_id
        AND (
            sr2.academic_year > sr.academic_year
            OR (
                sr2.academic_year = sr.academic_year
                AND sr2.semester > sr.semester
            )
        )
        AND (
            sr2.academic_year < %s
            OR (
                sr2.academic_year = %s
                AND sr2.semester < %s
            )
        )
    )
    ORDER BY a.subject_code
    """

    all_subjects_data = pd.read_sql(
        all_subjects_query,
        con,
        params=(
            selected_student,
            student_year,
            student_semester,
            student_year,
            student_year,
            student_semester
        )
    )

    # Merge only admin-approved student submissions into prediction inputs.
    # Official records take precedence; submissions never alter the official table.
    approved_subjects = pd.read_sql(
        """
        SELECT subject_code, quiz1, quiz2, midsem, assignment,
               classes_attended, classes_held, attendance_percentage,
               prev_sga, prev_sem_percentage, prev_total_backlogs
        FROM student_academic_submissions
        WHERE student_id = %s AND academic_year = %s
          AND semester = %s AND status = 'Approved'
        ORDER BY reviewed_at DESC, submitted_at DESC
        """,
        con,
        params=(selected_student, curriculum_year, student_semester)
    )

    if not approved_subjects.empty:
        approved_subjects = approved_subjects.drop_duplicates(
            subset=["subject_code"], keep="first"
        )
        if all_subjects_data.empty:
            all_subjects_data = approved_subjects.copy()
        else:
            for _, approved_row in approved_subjects.iterrows():
                mask = all_subjects_data["subject_code"] == approved_row["subject_code"]
                if mask.any():
                    for col in approved_subjects.columns:
                        if col != "subject_code":
                            all_subjects_data.loc[mask, col] = approved_row[col]
                else:
                    all_subjects_data = pd.concat(
                        [all_subjects_data, pd.DataFrame([approved_row])],
                        ignore_index=True
                    )

    if all_subjects_data.empty:
        st.warning("No official or admin-approved subject assessment records were found for this student and semester.")
        st.stop()

    if show_input_values:
        st.subheader("Prediction Input Values")

        input_columns = [
            "subject_code",
            "quiz1",
            "quiz2",
            "midsem",
            "assignment",
            "classes_attended",
            "classes_held",
            "attendance_percentage",
            "prev_sga",
            "prev_sem_percentage",
            "prev_total_backlogs"
        ]

        st.dataframe(
            all_subjects_data[input_columns],
            width="stretch",
            hide_index=True
        )

    predicted_subjects = subjects_df.copy()

    predicted_subjects["predicted_marks"] = None
    predicted_subjects["quiz1"] = None
    predicted_subjects["quiz2"] = None
    predicted_subjects["midsem"] = None
    predicted_subjects["assignment"] = None
    predicted_subjects["classes_attended"] = None
    predicted_subjects["classes_held"] = None
    predicted_subjects["attendance_percentage"] = None
    predicted_subjects["estimated_grade"] = None

    subject_explanations = {}

    for i, row in all_subjects_data.iterrows():

        subject_code = row["subject_code"]

        
        predicted_marks, explanation_df = predict_subject(
            row["subject_code"],
            row["quiz1"],
            row["quiz2"],
            row["midsem"],
            row["assignment"],
            row["attendance_percentage"],
            row["prev_sga"],
            row["prev_sem_percentage"],
            row["prev_total_backlogs"],
            student_branch,
            student_semester,
            return_explanation=True
        )

        subject_explanations[row["subject_code"]] = explanation_df


        predicted_subjects.loc[
            predicted_subjects["subject_code"] == subject_code,
            "predicted_marks"
        ] = predicted_marks

        for column in [
            "quiz1",
            "quiz2",
            "midsem",
            "assignment",
            "classes_attended",
            "classes_held",
            "attendance_percentage"
        ]:
            predicted_subjects.loc[
                predicted_subjects["subject_code"] == subject_code,
                column
            ] = row[column]

        if (
            pd.notna(row["attendance_percentage"])
            and row["attendance_percentage"] < 75
        ):
            predicted_subjects.loc[
                predicted_subjects["subject_code"] == subject_code,
                "estimated_grade"
            ] = "FA"
            continue

        if pd.isna(predicted_marks):
            predicted_subjects.loc[
                predicted_subjects["subject_code"] == subject_code,
                "estimated_grade"
            ] = "N/A"
            continue

        predicted_marks = float(predicted_marks)

        if predicted_marks < 40:
            predicted_subjects.loc[
                predicted_subjects["subject_code"] == subject_code,
                "estimated_grade"
            ] = "FF"
            continue

        grade_history_query = """
        SELECT
            a.grade,
            MIN(a.marks_obtained) AS min_marks,
            MAX(a.marks_obtained) AS max_marks,
            AVG(a.marks_obtained) AS avg_marks
        FROM academic_records a
        JOIN curriculum c
            ON a.subject_id = c.subject_id
        WHERE a.subject_code = %s
        AND c.branch = %s
        AND c.semester = %s
        AND a.grade NOT IN ('FF', 'FA')
        GROUP BY a.grade
        ORDER BY avg_marks
        """

        grade_history = pd.read_sql(
            grade_history_query,
            con,
            params=(
                subject_code,
                student_branch,
                student_semester
            )
        )

        if grade_history.empty:
            estimated_grade = "N/A"

        else:
            matching_grade = grade_history[
                (grade_history["min_marks"] <= predicted_marks)
                & (grade_history["max_marks"] >= predicted_marks)
            ]

            if not matching_grade.empty:
                estimated_grade = matching_grade.iloc[0]["grade"]

            else:
                grade_history["distance"] = (
                    grade_history["avg_marks"] - predicted_marks
                ).abs()

                estimated_grade = grade_history.loc[
                    grade_history["distance"].idxmin(),
                    "grade"
                ]

        predicted_subjects.loc[
            predicted_subjects["subject_code"] == subject_code,
            "estimated_grade"
        ] = estimated_grade

    historical_average_query = """
    SELECT
        a.subject_code,
        AVG(a.marks_obtained) AS historical_average
    FROM academic_records a
    JOIN curriculum c
        ON a.subject_id = c.subject_id
    WHERE c.branch = %s
    AND c.semester = %s
    AND a.grade NOT IN ('FF', 'FA')
    GROUP BY a.subject_code
    """

    historical_average_df = pd.read_sql(
        historical_average_query,
        con,
        params=(
            student_branch,
            student_semester
        )
    )

    predicted_subjects = predicted_subjects.merge(
        historical_average_df,
        on="subject_code",
        how="left"
    )

    predicted_subjects["performance_vs_history"] = (
        pd.to_numeric(predicted_subjects["predicted_marks"], errors="coerce")
        - pd.to_numeric(predicted_subjects["historical_average"], errors="coerce")
    )

    predicted_subjects["historical_comparison"] = (
        predicted_subjects["performance_vs_history"].apply(
            lambda x:
                "Unavailable"
                if pd.isna(x)
                else "Above Historical Average"
                if x > 0
                else "Below Historical Average"
                if x < 0
                else "Same as Historical Average"
        )
    )

    predicted_subjects["historical_performance_gap"] = (
        predicted_subjects["performance_vs_history"].apply(
            lambda x:
                "Unavailable"
                if pd.isna(x)
                else "Significantly Above"
                if x >= 10
                else "Above"
                if x >= 3
                else "Near Historical Average"
                if x > -3
                else "Below"
                if x > -10
                else "Significantly Below"
        )
    )

    def generate_subject_feedback(row):

        feedback = []
        predicted_marks = row["predicted_marks"]

        if pd.isna(predicted_marks):
            return "Prediction unavailable for this subject."

        if predicted_marks >= 85:
            feedback.append(
                "Strong predicted performance in this subject."
            )
        elif predicted_marks >= 70:
            feedback.append(
                "Decent predicted performance, with room for improvement."
            )
        elif predicted_marks >= 55:
            feedback.append(
                "Average predicted performance. Improvement is recommended."
            )
        else:
            feedback.append(
                "Low predicted performance. This subject needs attention."
            )

        if (
            pd.notna(row["attendance_percentage"])
            and row["attendance_percentage"] < 75
        ):
            feedback.append(
                "Attendance is below the required 75%."
            )

        if pd.notna(row["quiz1"]) and row["quiz1"] < 10:
            feedback.append(
                "Quiz 1 performance could be improved."
            )

        if pd.notna(row["quiz2"]) and row["quiz2"] < 10:
            feedback.append(
                "Quiz 2 performance could be improved."
            )

        if pd.notna(row["midsem"]) and row["midsem"] < 20:
            feedback.append(
                "Mid-semester performance could be improved."
            )

        if pd.notna(row["assignment"]) and row["assignment"] < 5:
            feedback.append(
                "Assignment performance could be improved."
            )

        return " ".join(feedback)

    predicted_subjects["feedback"] = predicted_subjects.apply(
        generate_subject_feedback,
        axis=1
    )

    predicted_subjects["predicted_grade_point"] = (
        predicted_subjects["estimated_grade"].map(grade_points)
    )

    predicted_subjects["weighted_grade_point"] = (
        predicted_subjects["predicted_grade_point"]
        * predicted_subjects["credits"]
    )

    total_credits = predicted_subjects["credits"].sum()

    if total_credits > 0:
        predicted_sga = (
            predicted_subjects["weighted_grade_point"].sum()
            / total_credits
        )
    else:
        predicted_sga = 0.0

    previous_cg_query = """
    SELECT
        sr.academic_year,
        sr.semester,
        sr.sga,
        SUM(c.credits) AS semester_credits
    FROM semester_results sr
    JOIN students s
        ON s.student_id = sr.student_id
    JOIN curriculum c
        ON c.branch = s.branch
        AND c.academic_year = sr.academic_year
        AND c.semester = sr.semester
    WHERE sr.student_id = %s
    AND (
        sr.academic_year < %s
        OR (
            sr.academic_year = %s
            AND sr.semester < %s
        )
    )
    GROUP BY
        sr.academic_year,
        sr.semester,
        sr.sga
    ORDER BY
        sr.academic_year,
        sr.semester
    """

    previous_cg_data = pd.read_sql(
        previous_cg_query,
        con,
        params=(
            selected_student,
            student_year,
            student_year,
            student_semester
        )
    )

    if not previous_cg_data.empty:

        previous_weighted_points = (
            previous_cg_data["sga"]
            * previous_cg_data["semester_credits"]
        ).sum()

        previous_total_credits = (
            previous_cg_data["semester_credits"].sum()
        )

        denominator = previous_total_credits + total_credits

        if denominator > 0:
            estimated_cg = (
                previous_weighted_points
                + predicted_sga * total_credits
            ) / denominator
        else:
            estimated_cg = predicted_sga

    else:
        estimated_cg = predicted_sga

    predicted_subjects["backlog_risk"] = predicted_subjects[
        "estimated_grade"
    ].apply(
        lambda grade:
            "High"
            if grade in ["FF", "FA"]
            else "Low"
    )

    predicted_backlog_count = (
        predicted_subjects["backlog_risk"] == "High"
    ).sum()

    predicted_marks_numeric = pd.to_numeric(
        predicted_subjects["predicted_marks"],
        errors="coerce"
    )

    if predicted_backlog_count > 0:
        backlog_risk = "High"
    elif predicted_marks_numeric.lt(40).any():
        backlog_risk = "Moderate"
    else:
        backlog_risk = "Low"

    if predicted_sga >= 9:
        performance_category = "Amazing"
    elif predicted_sga >= 8:
        performance_category = "Decent"
    elif predicted_sga >= 7:
        performance_category = "Average"
    elif predicted_sga >= 6:
        performance_category = "Below Par"
    else:
        performance_category = "Very Bad"


    if not academic_df.empty:
        actual_marks = pd.to_numeric(
            academic_df["marks_obtained"],
            errors="coerce"
        )

        actual_maximum = pd.to_numeric(
            academic_df["maximum_marks"],
            errors="coerce"
        )

        valid_rows = (
            actual_marks.notna()
            & actual_maximum.notna()
            & (actual_maximum > 0)
        )

        actual_percentage = (
            float(
                actual_marks[valid_rows].sum()
                / actual_maximum[valid_rows].sum()
                * 100
            )
            if valid_rows.any()
            else None
        )

        actual_total_backlogs = int(
            pd.to_numeric(
                academic_df["backlog"],
                errors="coerce"
            ).fillna(0).sum()
        )
    else:
        actual_percentage = None
        actual_total_backlogs = None


    if st.session_state.get("show_overall_dashboard", False):

        if show_prediction_summary:
            st.subheader("Semester Prediction")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Estimated SGA", f"{predicted_sga:.2f}")

            with col2:
                st.metric("Estimated CG", f"{estimated_cg:.2f}")

            with col3:
                st.metric("Backlog Risk", backlog_risk)

            with col4:
                st.metric("Performance Category", performance_category)



    if st.session_state["show_overall_dashboard"] and show_prediction_explanation:
        st.subheader("Factors Influencing Predicted Marks")

        selected_explanation_subject = st.selectbox(
            "Select subject to inspect its prediction factors",
            predicted_subjects["subject_code"].tolist(),
            key="oap_explanation_subject"
        )

        explanation_df = subject_explanations.get(
            selected_explanation_subject
        )

        if explanation_df is not None and not explanation_df.empty:
            feature_labels = {
                "num__quiz1": "Quiz 1 Marks",
                "num__quiz2": "Quiz 2 Marks",
                "num__midsem": "Mid-semester Marks",
                "num__assignment": "Assignment Marks",
                "num__attendance_percentage": "Attendance Percentage",
                "num__prev_sga": "Previous SGA",
                "num__prev_sem_percentage": "Previous Semester Percentage",
                "num__prev_total_backlogs": "Previous Backlogs",
                "num__semester": "Semester"
            }

            branch_names = {
                "CS": "Computer Science",
                "IT": "Information Technology",
                "AI": "Artificial Intelligence",
                "Mech": "Mechanical Engineering",
                "Civil": "Civil Engineering",
                "Chem": "Chemical Engineering",
                "Metallurgy": "Metallurgical Engineering",
                "ECE": "Electronics & Communication",
                "EEE": "Electrical & Electronics",
                "CDS": "Computational Data Science"
            }

            selected_branch_name = branch_names.get(
                student_branch, student_branch
            )

            grouped_explanations = []

            for _, factor in explanation_df.iterrows():
                feature = factor["feature"]

                if feature.startswith("cat__subject_code_"):
                    label = f"Subject: {selected_explanation_subject}"
                elif feature.startswith("cat__branch_"):
                    label = f"Branch: {selected_branch_name}"
                elif feature in feature_labels:
                    label = feature_labels[feature]
                elif feature.startswith("cat__"):
                    label = feature.replace("cat__", "").replace("_", " ")
                else:
                    label = feature

                grouped_explanations.append({
                    "feature": label,
                    "input_value": factor["input_value"],
                    "coefficient": factor["coefficient"],
                    "contribution": factor["contribution"]
                })

            explanation_df = pd.DataFrame(grouped_explanations)

            explanation_df = (
                explanation_df.groupby("feature", as_index=False)
                .agg({
                    "input_value": "sum",
                    "coefficient": "sum",
                    "contribution": "sum"
                })
            )

            explanation_df["absolute_contribution"] = (
                explanation_df["contribution"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "absolute_contribution", ascending=False
            )

            display_df = explanation_df[
                ["feature", "input_value", "coefficient", "contribution"]
            ].rename(columns={
                "feature": "Input Factor",
                "input_value": "Processed Input Value",
                "coefficient": "Model Coefficient",
                "contribution": "Contribution to Prediction"
            })

            st.dataframe(
                display_df.round(3),
                width="stretch",
                hide_index=True
            )

            top_explanations = explanation_df.head(10).sort_values(
                "contribution"
            )

            fig = px.bar(
                top_explanations,
                x="contribution",
                y="feature",
                orientation="h",
                title="Factors Influencing Predicted Marks",
                labels={
                    "contribution": "Contribution to Prediction",
                    "feature": "Input Factor"
                },
                color="contribution",
                color_continuous_scale="RdYlGn"
            )

            fig.add_vline(x=0, line_width=1, line_dash="dash")
            
            st.bar_chart(
                top_explanations.set_index("feature")["contribution"],
                horizontal=True
            )


            st.caption(
                "Positive contributions push the model output upward; negative "
                "contributions push it downward. Contributions are based on the "
                "processed input values and model coefficients."
            )
        else:
            st.info("No factor explanation is available for this subject.")


    predicted_subjects["attendance_action"] = None
    predicted_subjects["attendance_status"] = None

    for i, row in predicted_subjects.iterrows():

        attendance = row["attendance_percentage"]
        attended = row["classes_attended"]
        held = row["classes_held"]

        if pd.isna(attendance) or pd.isna(attended) or pd.isna(held):
            predicted_subjects.loc[i, "attendance_status"] = "Unavailable"
            predicted_subjects.loc[i, "attendance_action"] = (
                "Attendance data unavailable."
            )

        elif attendance >= 75:
            predicted_subjects.loc[i, "attendance_status"] = "Safe"

            can_miss = max(
                0,
                int((attended / 0.75) - held)
            )

            if can_miss == 0:
                predicted_subjects.loc[i, "attendance_action"] = (
                    "Cannot miss another class while maintaining 75%."
                )
            else:
                predicted_subjects.loc[i, "attendance_action"] = (
                    f"Can miss approximately {can_miss} more classes."
                )

        else:
            predicted_subjects.loc[i, "attendance_status"] = "Shortage"

            required = max(
                0,
                int(np.ceil((0.75 * held - attended) / 0.25))
            )

            predicted_subjects.loc[i, "attendance_action"] = (
                f"Must attend the next {required} classes."
            )

    if st.session_state["show_overall_dashboard"] and show_predicted_subjects:
        st.subheader("Predicted Subject Results")

        st.dataframe(
            predicted_subjects[
                [
                    "subject_code",
                    "subject_name",
                    "credits",
                    "predicted_marks",
                    "estimated_grade",
                    "backlog_risk",
                    "feedback"
                ]
            ],
            width="stretch"
        )
    if st.session_state["show_overall_dashboard"] and show_attendance_tracker:
        st.subheader("Attendance Tracker")

        st.dataframe(
            predicted_subjects[
                [
                    "subject_code",
                    "credits",
                    "attendance_percentage",
                    "attendance_status",
                    "attendance_action"
                ]
            ],
            width="stretch"
        )

    historical_comparison_df = predicted_subjects[
        [
            "subject_code",
            "subject_name",
            "predicted_marks",
            "historical_average",
            "performance_vs_history",
            "historical_comparison",
            "historical_performance_gap"
        ]
    ].copy()

    historical_comparison_df["performance_vs_history"] = pd.to_numeric(
        historical_comparison_df["performance_vs_history"],
        errors="coerce"
    )

    historical_comparison_df["predicted_marks"] = pd.to_numeric(
        historical_comparison_df["predicted_marks"],
        errors="coerce"
    )

    historical_comparison_df["historical_average"] = pd.to_numeric(
        historical_comparison_df["historical_average"],
        errors="coerce"
    )

    performance_status_df = historical_comparison_df[
        [
            "subject_code",
            "subject_name",
            "predicted_marks",
            "historical_average",
            "performance_vs_history"
        ]
    ].copy()

    performance_status_df = performance_status_df.merge(
        predicted_subjects[
            [
                "subject_code",
                "attendance_percentage",
                "estimated_grade"
            ]
        ],
        on="subject_code",
        how="left"
    )

    performance_status_df["performance_status"] = (
        performance_status_df["performance_vs_history"].apply(
            lambda x:
                "Unavailable"
                if pd.isna(x)
                else "Excellent"
                if x >= 10
                else "Above Average"
                if x >= 3
                else "Average"
                if x > -3
                else "Below Average"
                if x > -10
                else "Needs Improvement"
        )
    )

    performance_status_df = performance_status_df.sort_values(
        by="performance_vs_history",
        na_position="last"
    )

    if st.session_state["show_overall_dashboard"] and show_historical_comparison:
        st.subheader("Historical Performance Comparison")

        comparison_chart_df = historical_comparison_df[
            [
                "subject_name",
                "predicted_marks",
                "historical_average"
            ]
        ].copy()

        comparison_chart_df = comparison_chart_df.melt(
            id_vars="subject_name",
            value_vars=[
                "predicted_marks",
                "historical_average"
            ],
            var_name="Performance Type",
            value_name="Marks"
        )

        comparison_chart_df["Performance Type"] = (
            comparison_chart_df["Performance Type"].replace({
                "predicted_marks": "Predicted Marks",
                "historical_average": "Historical Average"
            })
        )

        fig = px.bar(
            comparison_chart_df,
            x="subject_name",
            y="Marks",
            color="Performance Type",
            barmode="group",
            title="Predicted Marks vs Historical Average"
        )

        fig.update_layout(
            xaxis_title="Subject",
            yaxis_title="Marks",
            yaxis_range=[0, 100]
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.dataframe(
            performance_status_df[
                [
                    "subject_code",
                    "subject_name",
                    "predicted_marks",
                    "historical_average",
                    "performance_vs_history",
                    "performance_status"
                ]
            ],
            width="stretch"
        )

    def generate_subject_advice(row):

        attendance = row["attendance_percentage"]
        grade = row["estimated_grade"]
        gap = row["performance_vs_history"]

        if pd.notna(attendance) and attendance < 75:
            return (
                "Attendance shortage: improve attendance immediately "
                "and attend the required upcoming classes."
            )

        if grade in ["FF", "FA"]:
            return (
                "High risk: the predicted grade indicates a possible backlog. "
                "Prioritize this subject and seek academic support."
            )

        if pd.isna(gap):
            return (
                "Historical comparison unavailable. Maintain regular study "
                "and review performance when more data is available."
            )

        if gap <= -10:
            return (
                "High priority: significantly below historical performance. "
                "Increase study time and focus on weak concepts."
            )
        elif gap <= -3:
            return (
                "Needs improvement: performance is below the historical average. "
                "Review difficult topics and practice more questions."
            )
        elif gap < 3:
            return (
                "Maintain consistency: performance is close to the historical "
                "average. Regular revision can help improve further."
            )
        elif gap < 10:
            return (
                "Good performance: currently above the historical average. "
                "Continue the same preparation strategy."
            )
        else:
            return (
                "Excellent performance: significantly above the historical "
                "average. Maintain your current preparation level."
            )

    performance_status_df["advice"] = performance_status_df.apply(
        generate_subject_advice,
        axis=1
    )

    if st.session_state["show_overall_dashboard"] and show_subject_advice:
        st.subheader("Subject-Specific Advice")

        st.dataframe(
            performance_status_df[
                [
                    "subject_code",
                    "performance_status",
                    "advice"
                ]
            ],
            column_config={
                "advice": st.column_config.TextColumn(
                    "Advice",
                    width="large"
                ),
                "subject_code": st.column_config.TextColumn(
                    "Subject",
                    width="small"
                ),
                "performance_status": st.column_config.TextColumn(
                    "Status",
                    width="small"
                )
            },
            width="stretch"
        )

    high_risk_subjects = (
        predicted_subjects["backlog_risk"] == "High"
    ).sum()

    moderate_risk_subjects = (
        predicted_marks_numeric < 40
    ).sum()

    attendance_shortage_subjects = (
        pd.to_numeric(
            predicted_subjects["attendance_percentage"],
            errors="coerce"
        ) < 75
    ).sum()

    if high_risk_subjects > 0:
        overall_academic_risk = "High"
    elif attendance_shortage_subjects > 0 or moderate_risk_subjects > 0:
        overall_academic_risk = "Moderate"
    else:
        overall_academic_risk = "Low"

    reasons = []

    if attendance_shortage_subjects > 0:
        reasons.append(
            f"{attendance_shortage_subjects} subject(s) have attendance below 75%"
        )

    if moderate_risk_subjects > 0:
        reasons.append(
            f"{moderate_risk_subjects} subject(s) have predicted marks below 40"
        )

    if overall_academic_risk == "High":
        risk_explanation = (
            f"High academic risk: {high_risk_subjects} subject(s) "
            "currently show a predicted backlog risk. Immediate attention is recommended."
        )
    elif overall_academic_risk == "Moderate":
        risk_explanation = (
            "Moderate academic risk: " + " and ".join(reasons) + "."
        )
    else:
        risk_explanation = (
            "Low academic risk: no predicted backlog risk, "
            "no predicted marks below 40, and no attendance shortage."
        )

    if st.session_state["show_overall_dashboard"] and show_academic_risk:
        st.subheader("Overall Academic Risk")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "High-Risk Subjects",
                int(high_risk_subjects)
            )

        with col2:
            st.metric(
                "Attendance Shortages",
                int(attendance_shortage_subjects)
            )

        st.info(risk_explanation)

        if overall_academic_risk == "High":

            high_risk_df = predicted_subjects[
                predicted_subjects["backlog_risk"] == "High"
            ]

            st.error("Recommended Action")

            for _, row in high_risk_df.iterrows():
                st.write(
                    f"🔴 **{row['subject_name']}** — High backlog risk. "
                    "Prioritize this subject immediately and seek additional academic support."
                )

        elif overall_academic_risk == "Moderate":

            moderate_subjects = predicted_subjects[
                (predicted_marks_numeric < 40)
                | (
                    pd.to_numeric(
                        predicted_subjects["attendance_percentage"],
                        errors="coerce"
                    ) < 75
                )
            ]

            st.warning("Recommended Action")

            for _, row in moderate_subjects.iterrows():

                action_reasons = []

                if pd.notna(row["predicted_marks"]) and row["predicted_marks"] < 40:
                    action_reasons.append(
                        f"predicted marks are {row['predicted_marks']:.1f}"
                    )

                if (
                    pd.notna(row["attendance_percentage"])
                    and row["attendance_percentage"] < 75
                ):
                    action_reasons.append(
                        f"attendance is {row['attendance_percentage']:.1f}%"
                    )

                st.write(
                    f"🟠 **{row['subject_name']}** — "
                    + " and ".join(action_reasons)
                    + ". Focus on this subject and take corrective action."
                )

        else:

            below_average_subjects = predicted_subjects[
                predicted_subjects["historical_performance_gap"].isin(
                    ["Below", "Significantly Below"]
                )
            ]

            if not below_average_subjects.empty:

                st.success("Recommended Action")

                for _, row in below_average_subjects.iterrows():

                    marks = row["predicted_marks"]

                    if pd.notna(marks):
                        st.write(
                            f"🟢 **{row['subject_name']}** — "
                            f"currently predicted at {marks:.1f} marks, "
                            "which is below the historical average. "
                            "Consider additional revision and practice."
                        )

            else:
                st.success(
                    "Overall performance is healthy. Continue your current "
                    "preparation strategy and maintain attendance above 75%."
                )

    subject_action_df = predicted_subjects[
        [
            "subject_code",
            "subject_name",
            "predicted_marks",
            "estimated_grade",
            "attendance_percentage",
            "backlog_risk",
            "historical_performance_gap"
        ]
    ].copy()

    

    def get_action_priority(row):

        if row["backlog_risk"] == "High":
            return "Critical"

        attendance = row["attendance_percentage"]

        if pd.notna(attendance) and attendance < 75:
            return "High"

        marks = row["predicted_marks"]

        if pd.notna(marks) and marks < 40:
            return "High"

        if row["historical_performance_gap"] in [
            "Below",
            "Significantly Below"
        ]:
            return "Medium"

        return "Low"

    subject_action_df["action_priority"] = (
        subject_action_df.apply(
            get_action_priority,
            axis=1
        )
    )

    priority_order = {
        "Critical": 0,
        "High": 1,
        "Medium": 2,
        "Low": 3
    }

    subject_action_df["priority_order"] = (
        subject_action_df["action_priority"].map(priority_order)
    )

    subject_action_df = subject_action_df.sort_values(
        by="priority_order"
    )

    if st.session_state["show_overall_dashboard"] and show_action_priorities:
        st.subheader("Subject Action Priorities")

        st.dataframe(
            subject_action_df[
                [
                    "subject_code",
                    "subject_name",
                    "predicted_marks",
                    "estimated_grade",
                    "attendance_percentage",
                    "action_priority",
                    "backlog_risk"
                ]
            ],
            width="stretch")



    current_prediction_context = (
        str(selected_student),
        int(student_year),
        int(student_semester)
    )

    if st.session_state.get("prediction_context") != current_prediction_context:
        st.session_state["prediction_context"] = current_prediction_context
        st.session_state["overall_prediction_generated"] = False

    if not st.session_state["overall_prediction_generated"]:

        generate_clicked = st.button(
            "Generate Overall Academic Prediction",
            type="primary",
            key=f"generate_overall_{selected_student}_{student_year}_{student_semester}"
        )

        if generate_clicked:
            try:
                # Ensure all required metrics have been calculated.
                if total_credits <= 0:
                    st.error("Cannot generate a prediction because total credits are zero.")
                    st.stop()

                cursor.execute(
                    """
                    INSERT INTO overall_prediction_history (
                        student_id,
                        academic_year,
                        semester,
                        predicted_sga,
                        estimated_cg,
                        predicted_backlog_count,
                        backlog_risk,
                        performance_category,
                        overall_academic_risk
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING prediction_id
                    """,
                    (
                        str(selected_student),
                        int(student_year),
                        int(student_semester),
                        round(float(predicted_sga), 2),
                        round(float(estimated_cg), 2),
                        int(predicted_backlog_count),
                        str(backlog_risk),
                        str(performance_category),
                        str(overall_academic_risk)
                    )
                )

                prediction_id = cursor.fetchone()[0]

                # Save the subject-level results for this prediction.
                for _, subject_row in predicted_subjects.iterrows():
                    def safe_float(value):
                        return (
                            float(value)
                            if pd.notna(value)
                            else None
                        )

                    cursor.execute(
                        """
                        INSERT INTO overall_prediction_subjects (
                            prediction_id,
                            subject_code,
                            subject_name,
                            credits,
                            predicted_marks,
                            estimated_grade,
                            attendance_percentage,
                            backlog_risk,
                            historical_average,
                            performance_vs_history
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            int(prediction_id),
                            str(subject_row["subject_code"]),
                            str(subject_row["subject_name"]),
                            safe_float(subject_row["credits"]),
                            safe_float(subject_row["predicted_marks"]),
                            (
                                str(subject_row["estimated_grade"])
                                if pd.notna(subject_row["estimated_grade"])
                                else None
                            ),
                            safe_float(subject_row["attendance_percentage"]),
                            str(subject_row["backlog_risk"]),
                            safe_float(subject_row["historical_average"]),
                            safe_float(subject_row["performance_vs_history"])
                        )
                    )


                # Update the matching academic record with predicted results
                cursor.execute(
                    """
                    UPDATE academic_records
                    SET
                        marks_obtained = %s,
                        grade = %s,
                        attendance_percentage = %s,
                        backlog = %s
                    WHERE student_id = %s
                        AND subject_code = %s
                        AND academic_year = %s
                        AND semester = %s
                    """,
                    (
                        safe_float(subject_row["predicted_marks"]),
                        (
                            str(subject_row["estimated_grade"])
                            if pd.notna(subject_row["estimated_grade"])
                            else None
                        ),
                        safe_float(subject_row["attendance_percentage"]),
                        (
                            1
                            if str(subject_row["backlog_risk"]).strip().lower()
                            in ("high", "yes", "true")
                            else 0
                        ),
                        str(selected_student),
                        str(subject_row["subject_code"]),
                        int(student_year),
                        int(student_semester)
                    )
                )

                predicted_marks_numeric = pd.to_numeric(
                    predicted_subjects["predicted_marks"],
                    errors="coerce"
                )

                valid_marks = predicted_marks_numeric.dropna()

                semester_percentage = (
                    float(valid_marks.mean())
                    if not valid_marks.empty
                    else None
                )

                semester_status = (
                    "At Risk"
                    if str(overall_academic_risk).strip().lower()
                    in ("high", "yes")
                    else "On Track"
                )

                cursor.execute(
                    """
                    INSERT INTO semester_results (
                        student_id,
                        academic_year,
                        semester,
                        sga,
                        semester_percentage,
                        total_backlogs,
                        semester_status,
                        result_type
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(selected_student),
                        int(student_year),
                        int(student_semester),
                        round(float(predicted_sga), 2),
                        (
                            round(float(semester_percentage), 2)
                            if semester_percentage is not None
                            else None
                        ),
                        int(predicted_backlog_count),
                        semester_status,
                        "Predicted"
                    )
                )


                con.commit()

                st.cache_data.clear()

                st.session_state["saved_overall_prediction"] = {
                    "prediction_id": int(prediction_id),
                    "student_id": str(selected_student),
                    "academic_year": int(student_year),
                    "semester": int(student_semester)
                }

                st.session_state["show_overall_dashboard"] = True

                st.session_state["saved_overall_prediction"] = {
                    "prediction_id": int(prediction_id),
                    "student_id": str(selected_student),
                    "academic_year": int(student_year),
                    "semester": int(student_semester)
                }

                con.commit()

                st.session_state["overall_prediction_generated"] = True

                st.success("Overall academic prediction saved successfully.")
                st.rerun()

            except Exception as e:
                con.rollback()
                st.error(f"Could not save the prediction: {e}")




if page == "Student Academic Details":

    st.title("Student Academic Details")
    st.write(
        "Submit your marks and attendance for admin review. "
        "Approved submissions cannot be submitted again for the same subject, "
        "academic year, and semester."
    )

    # ---------------------------------------------------------
    # 1. Validate the logged-in student's profile
    # ---------------------------------------------------------
    student_row = students_df[
        students_df["student_id"].astype(str) == str(student_id)
    ]

    if student_row.empty:
        st.error("Your student profile could not be found.")
        st.stop()

    profile = student_row.iloc[0]
    student_branch = str(profile["branch"]).strip()
    student_name = str(profile["name"])

    # ---------------------------------------------------------
    # 2. Load the student's submission history
    # ---------------------------------------------------------
    submitted_rows = pd.read_sql(
        """
        SELECT
            submission_id,
            subject_code,
            subject_name,
            academic_year,
            semester,
            quiz1,
            quiz2,
            midsem,
            assignment,
            classes_attended,
            classes_held,
            attendance_percentage,
            status,
            admin_notes,
            submitted_at,
            reviewed_at
        FROM student_academic_submissions
        WHERE student_id = %s
        ORDER BY submitted_at DESC, submission_id DESC
        """,
        con,
        params=(str(student_id),)
    )

    # ---------------------------------------------------------
    # 3. Academic year and semester selection
    # ---------------------------------------------------------
    profile_year = profile.get("academic_year")
    profile_semester = profile.get("current_semester")

    default_year = (
        int(profile_year)
        if pd.notna(profile_year) and str(profile_year).strip()
        else 2026
    )

    default_semester = (
        int(profile_semester)
        if pd.notna(profile_semester) and str(profile_semester).strip()
        else 1
    )

    with st.form(key=f"student_academic_form_{student_id}"):

        st.subheader("Submit Academic Details")

        academic_year = st.number_input(
            "Academic Year",
            min_value=2000,
            max_value=2100,
            value=max(2000, min(2100, default_year)),
            step=1,
            key=f"submission_year_{student_id}"
        )

        semester = st.selectbox(
            "Semester",
            options=list(range(1, 9)),
            index=max(0, min(7, default_semester - 1)),
            key=f"submission_semester_{student_id}"
        )
        st.write("Debug student_branch:", repr(student_branch))
        st.write("Debug semester:", repr(semester))
        # Query the curriculum for the selected branch and semester.
        subjects = pd.read_sql(
            """
            SELECT DISTINCT subject_code, subject_name
            FROM curriculum
            WHERE UPPER(TRIM(branch)) = UPPER(TRIM(%s))
              AND semester = %s
            ORDER BY subject_code
            """,
            con,
            params=(student_branch, int(semester))
        )

        selected_code = None
        quiz1 = quiz2 = midsem = assignment = None
        classes_attended = classes_held = None
        prev_sga = prev_sem_percentage = None
        prev_total_backlogs = 0

        if subjects.empty:
            st.error(
                f"No curriculum subjects found. "
                f"Branch={student_branch!r}, "
                f"Semester={semester!r}"
            )

            st.write("Database:", con.database)
            st.write("Matching curriculum rows:")

            st.dataframe(
                pd.read_sql(
                    """
                    SELECT subject_code, subject_name, branch, semester
                    FROM curriculum
                    WHERE UPPER(TRIM(branch)) = UPPER(TRIM(%s))
                    AND semester = %s
                    """,
                    con,
                    params=(student_branch, int(semester))
                )
            )
        else:
            subject_options = subjects["subject_code"].astype(str).tolist()

            subject_names = {
                str(row["subject_code"]): str(row["subject_name"])
                for _, row in subjects.iterrows()
            }

            selected_code = st.selectbox(
                "Subject",
                options=subject_options,
                format_func=lambda code: (
                    f"{code} — {subject_names.get(str(code), '')}"
                ),
                key=(
                    f"submission_subject_{student_id}_"
                    f"{int(academic_year)}_{int(semester)}"
                )
            )

            # All widget keys are unique to the student, subject, year and semester.
            widget_suffix = (
                f"{student_id}_{selected_code}_"
                f"{int(academic_year)}_{int(semester)}"
            )

            st.markdown("#### Assessment Marks")

            quiz1 = st.number_input(
                "Quiz 1 Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"submission_quiz1_{widget_suffix}"
            )

            quiz2 = st.number_input(
                "Quiz 2 Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"submission_quiz2_{widget_suffix}"
            )

            midsem = st.number_input(
                "Mid-semester Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"submission_midsem_{widget_suffix}"
            )

            assignment = st.number_input(
                "Assignment Marks",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=0.5,
                key=f"submission_assignment_{widget_suffix}"
            )

            st.markdown("#### Attendance")

            classes_attended = st.number_input(
                "Classes Attended",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                key=f"submission_attended_{widget_suffix}"
            )

            classes_held = st.number_input(
                "Classes Held",
                min_value=1,
                max_value=1000,
                value=1,
                step=1,
                key=f"submission_held_{widget_suffix}"
            )

            attendance_percentage = (
                (int(classes_attended) / int(classes_held)) * 100
                if int(classes_held) > 0
                else 0.0
            )

            st.metric(
                "Calculated Attendance",
                f"{attendance_percentage:.2f}%"
            )

            if classes_attended > classes_held:
                st.error(
                    "Classes attended cannot exceed classes held."
                )

            # Previous-semester inputs are not applicable in semester 1.
            if int(semester) == 1:
                prev_sga = None
                prev_sem_percentage = None
                prev_total_backlogs = 0

                st.caption(
                    "Previous-semester details are not required for Semester 1."
                )
            else:
                st.markdown("#### Previous Academic Performance")

                prev_sga = st.number_input(
                    "Previous SGA",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key=f"submission_prev_sga_{widget_suffix}"
                )

                prev_sem_percentage = st.number_input(
                    "Previous Semester Percentage",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.5,
                    key=f"submission_prev_percentage_{widget_suffix}"
                )

                prev_total_backlogs = st.number_input(
                    "Previous Backlogs",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    key=f"submission_prev_backlogs_{widget_suffix}"
                )

        submitted = st.form_submit_button(
            "Submit for Admin Validation",
            type="primary",
            disabled=subjects.empty
        )

    # ---------------------------------------------------------
    # 4. Validate and save the submitted values
    # ---------------------------------------------------------

    # Stage 1: Capture and save the entered values temporarily
    if submitted and selected_code is not None:
        attendance = (
            float(classes_attended) / float(classes_held) * 100
            if classes_held > 0 else 0.0
        )

        submission_data = {
            "student_id": str(student_id),
            "student_name": student_name,
            "branch": student_branch,
            "subject_code": str(selected_code),
            "subject_name": str(
                subjects.loc[
                    subjects["subject_code"].astype(str) == str(selected_code),
                    "subject_name"
                ].iloc[0]
            ),
            "academic_year": int(academic_year),
            "semester": int(semester),
            "quiz1": float(quiz1),
            "quiz2": float(quiz2),
            "midsem": float(midsem),
            "assignment": float(assignment),
            "classes_attended": int(classes_attended),
            "classes_held": int(classes_held),
            "attendance_percentage": round(attendance, 2),
            "prev_sga": prev_sga,
            "prev_sem_percentage": prev_sem_percentage,
            "prev_total_backlogs": int(prev_total_backlogs),
        }

        # Validate before showing the confirmation screen
        marks = [
            submission_data["quiz1"],
            submission_data["quiz2"],
            submission_data["midsem"],
            submission_data["assignment"],
        ]

        if any(mark < 0 or mark > 100 for mark in marks):
            st.error("Marks must be between 0 and 100.")
        elif submission_data["classes_held"] <= 0:
            st.error("Classes held must be greater than zero.")
        elif submission_data["classes_attended"] > submission_data["classes_held"]:
            st.error("Classes attended cannot exceed classes held.")
        else:
            st.session_state["pending_academic_submission"] = submission_data
            st.session_state["confirm_academic_submission"] = True
            st.rerun()


    # Stage 2: Display the captured values and ask for confirmation
    if st.session_state.get("confirm_academic_submission", False):

        data = st.session_state["pending_academic_submission"]

        st.subheader("Confirm Academic Submission")
        st.write(
            f"**Subject:** {data['subject_code']} — "
            f"{data['subject_name']}"
        )

        st.dataframe(
            pd.DataFrame([
                {"Field": "Quiz 1", "Value": data["quiz1"]},
                {"Field": "Quiz 2", "Value": data["quiz2"]},
                {"Field": "Mid-semester", "Value": data["midsem"]},
                {"Field": "Assignment", "Value": data["assignment"]},
                {"Field": "Classes Attended", "Value": data["classes_attended"]},
                {"Field": "Classes Held", "Value": data["classes_held"]},
                {
                    "Field": "Attendance (%)",
                    "Value": data["attendance_percentage"],
                },
            ]),
            hide_index=True,
            width="stretch",
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            confirm_clicked = st.button(
                "Confirm and Submit for Validation",
                type="primary",
                key="confirm_academic_submission_button",
            )

        with cancel_col:
            cancel_clicked = st.button(
                "Cancel",
                key="cancel_academic_submission_button",
            )

        if cancel_clicked:
            st.session_state.pop("pending_academic_submission", None)
            st.session_state["confirm_academic_submission"] = False
            st.rerun()

        if confirm_clicked:
            data = st.session_state["pending_academic_submission"]

            try:
                # Prevent duplicate Pending or already Approved submissions.
                existing = pd.read_sql(
                    """
                    SELECT status
                    FROM student_academic_submissions
                    WHERE student_id = %s
                    AND subject_code = %s
                    AND academic_year = %s
                    AND semester = %s
                    ORDER BY submitted_at DESC, submission_id DESC
                    """,
                    con,
                    params=(
                        data["student_id"],
                        data["subject_code"],
                        data["academic_year"],
                        data["semester"],
                    ),
                )

                statuses = (
                    existing["status"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.casefold()
                )

                if statuses.eq("approved").any():
                    st.error(
                        "This subject has already been approved for this "
                        "academic year and semester."
                    )

                elif statuses.eq("pending").any():
                    st.error(
                        "A submission for this subject is already pending review."
                    )

                else:
                    cursor.execute(
                        """
                        INSERT INTO student_academic_submissions
                        (
                            student_id, student_name, branch,
                            subject_code, subject_name,
                            academic_year, semester,
                            quiz1, quiz2, midsem, assignment,
                            classes_attended, classes_held,
                            attendance_percentage,
                            prev_sga, prev_sem_percentage,
                            prev_total_backlogs, status
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, 'Pending'
                        )
                        """,
                        (
                            data["student_id"],
                            data["student_name"],
                            data["branch"],
                            data["subject_code"],
                            data["subject_name"],
                            data["academic_year"],
                            data["semester"],
                            data["quiz1"],
                            data["quiz2"],
                            data["midsem"],
                            data["assignment"],
                            data["classes_attended"],
                            data["classes_held"],
                            data["attendance_percentage"],
                            data["prev_sga"],
                            data["prev_sem_percentage"],
                            data["prev_total_backlogs"],
                        ),
                    )

                    con.commit()
                    st.session_state.pop("pending_academic_submission", None)
                    st.session_state["confirm_academic_submission"] = False

                    st.success("Your academic details were submitted for admin review.")
                    st.rerun()

            except Exception as exc:
                con.rollback()
                st.error(f"Submission failed: {exc}")


    # ---------------------------------------------------------
    # 5. Show the student's submission history
    # ---------------------------------------------------------
    st.subheader("My Submission Status")

    refreshed_submissions = pd.read_sql(
        """
        SELECT
            submission_id,
            subject_code,
            subject_name,
            academic_year,
            semester,
            quiz1,
            quiz2,
            midsem,
            assignment,
            classes_attended,
            classes_held,
            attendance_percentage,
            status,
            admin_notes,
            submitted_at,
            reviewed_at
        FROM student_academic_submissions
        WHERE student_id = %s
        ORDER BY submitted_at DESC, submission_id DESC
        """,
        con,
        params=(str(student_id),)
    )

    if refreshed_submissions.empty:
        st.info("You have not submitted academic details yet.")
    else:
        st.dataframe(
            refreshed_submissions,
            width="stretch",
            hide_index=True
        )

if page == "Student Data Review" and role == "admin":
    st.title("Student Data Review")
    st.write("Review student-submitted academic details before they are used by prediction pages.")

    status_filter = st.selectbox(
        "Filter by status",
        ["Pending", "Needs Correction", "Approved", "Rejected", "All"]
    )

    review_query = """
        SELECT submission_id, student_id, student_name, branch,
               subject_code, subject_name, academic_year, semester,
               quiz1, quiz2, midsem, assignment, classes_attended,
               classes_held, attendance_percentage, prev_sga,
               prev_sem_percentage, prev_total_backlogs, status,
               admin_notes, submitted_at
        FROM student_academic_submissions
    """
    if status_filter != "All":
        review_query += " WHERE status = %s"
        review_params = (status_filter,)
    else:
        review_params = ()

    review_query += " ORDER BY submitted_at DESC"
    review_df = pd.read_sql(review_query, con, params=review_params)

    if review_df.empty:
        st.info("No submissions match this filter.")
    else:
        st.dataframe(review_df, width="stretch", hide_index=True)
        submission_ids = review_df["submission_id"].astype(int).tolist()
        selected_submission_id = st.selectbox("Select submission", submission_ids)
        selected_submission = review_df[
            review_df["submission_id"] == selected_submission_id
        ].iloc[0]

        st.subheader(
            f"{selected_submission['student_name']} — "
            f"{selected_submission['subject_code']} "
            f"(Semester {selected_submission['semester']})"
        )
        st.write(
            f"Marks: Quiz 1 {selected_submission['quiz1']}, "
            f"Quiz 2 {selected_submission['quiz2']}, "
            f"Midsem {selected_submission['midsem']}, "
            f"Assignment {selected_submission['assignment']}"
        )
        st.write(
            f"Attendance: {selected_submission['classes_attended']} / "
            f"{selected_submission['classes_held']} "
            f"({selected_submission['attendance_percentage']}%)"
        )

        with st.form("admin_submission_review_form"):
            decision = st.radio(
                "Review decision",
                ["Approved", "Rejected", "Needs Correction"],
                horizontal=True
            )
            notes = st.text_area(
                "Admin notes / correction instructions",
                value="" if pd.isna(selected_submission["admin_notes"]) else str(selected_submission["admin_notes"])
            )
            review_clicked = st.form_submit_button("Save Review", type="primary")

        if review_clicked:
            cursor.execute(
                """
                UPDATE student_academic_submissions
                SET status = %s, admin_notes = %s,
                    reviewed_at = CURRENT_TIMESTAMP, reviewed_by = %s
                WHERE submission_id = %s
                """,
                (decision, notes, str(user.get("username", "admin")), int(selected_submission_id))
            )
            con.commit()
            st.success(f"Submission marked as {decision}.")
            st.rerun()


if page == "Individual Subject Prediction":
    st.title("Individual Subject Prediction")

    # Identify the student based on role.
    if role == "admin":
        student_ids = students_df["student_id"].tolist()

        if not student_ids:
            st.warning("No students are available.")
            st.stop()

        selected_student = st.selectbox(
            "Select Student",
            student_ids,
            key="individual_prediction_student"
        )
    else:
        selected_student = student_id

    student_rows = students_df[
        students_df["student_id"] == selected_student
    ]

    if student_rows.empty:
        st.error("Student information could not be found.")
        st.stop()

    student_info = student_rows.iloc[0]
    student_branch = student_info["branch"]

    st.write("Student ID:", selected_student)
    st.write("Branch:", student_branch)
    
    # Determine the latest available academic term.
    current_semester_df = pd.read_sql(
        """
        SELECT academic_year, semester
        FROM academic_records
        WHERE student_id = %s
        ORDER BY academic_year DESC, semester DESC
        LIMIT 1
        """,
        con,
        params=(selected_student,)
    )

    if not current_semester_df.empty:
        latest_semester = int(
            current_semester_df.iloc[0]["semester"]
        )
        curriculum_year = int(
            current_semester_df.iloc[0]["academic_year"]
        )
    else:
        approved_term = pd.read_sql(
            """
            SELECT academic_year, semester
            FROM student_academic_submissions
            WHERE student_id = %s
              AND status = 'Approved'
            ORDER BY academic_year DESC, semester DESC
            LIMIT 1
            """,
            con,
            params=(selected_student,)
        )

        profile_semester = student_info.get("current_semester")

        if not approved_term.empty:
            latest_semester = int(
                approved_term.iloc[0]["semester"]
            )
            curriculum_year = int(
                approved_term.iloc[0]["academic_year"]
            )
        else:
            latest_semester = (
                int(profile_semester)
                if pd.notna(profile_semester)
                else 1
            )

            profile_year = student_info.get("academic_year")
            curriculum_year = (
                int(profile_year)
                if pd.notna(profile_year)
                else None
            )

    if current_semester_df.empty and curriculum_year is None:
        st.warning(
            "Academic year is unavailable. "
            "Submit details with an academic year first."
        )
        st.stop()

    # Allow manual semester selection.
    student_semester = st.selectbox(
        "Select Semester",
        options=list(range(1, 9)),
        index=max(0, min(latest_semester - 1, 7)),
        key=f"prediction_semester_{selected_student}"
    )

    # Study year is derived from the selected semester.
    student_year = ((student_semester - 1) // 2) + 1

    is_first_semester = student_semester == 1


    st.write(
        f"Branch: {student_branch} | "
        f"Study Year: {student_year} | "
        f"Curriculum Year: {curriculum_year} | "
        f"Semester: {student_semester}"
    )

    # Load the subjects for this branch and semester.
    subjects_df = pd.read_sql(
        """
        SELECT DISTINCT subject_id, subject_code, subject_name, credits
        FROM curriculum
        WHERE UPPER(TRIM(branch)) = UPPER(TRIM(%s))
        AND semester = %s
        ORDER BY subject_code
        """,
        con,
        params=(student_branch, student_semester)
    )

    subjects_df = subjects_df.drop_duplicates(subset=["subject_code"])

    if subjects_df.empty:
        st.warning(
            "No subjects were found for this student's selected semester."
        )
        st.stop()

    subject_names = dict(
        zip(subjects_df["subject_code"], subjects_df["subject_name"])
    )

    selected_subject = st.selectbox(
        "Select Subject",
        subjects_df["subject_code"].tolist(),
        format_func=lambda code: (
            f"{code} — {subject_names.get(code, 'Unknown Subject')}"
        ),
        key="individual_prediction_subject"
    )

    selected_subject_row = subjects_df[
        subjects_df["subject_code"] == selected_subject
    ]

    if selected_subject_row.empty:
        st.error("Could not find the selected subject.")
        st.stop()

    selected_subject_id = int(
        selected_subject_row.iloc[0]["subject_id"]
    )

    input_columns = [
        "quiz1",
        "quiz2",
        "midsem",
        "assignment",
        "attendance_percentage",
        "prev_sga",
        "prev_sem_percentage",
        "prev_total_backlogs"
    ]

    # --------------------------------------------------
    # ADMIN: Load marks from the database.
    # --------------------------------------------------
    if role == "admin":

        st.subheader("Stored Student Marks")

        if st.button(
            "Load Marks & Predict",
            type="primary",
            key="load_marks_predict"
        ):
            subject_data = pd.read_sql(
                subject_data_query,
                con,
                params=(
                    selected_student,
                    selected_subject,
                    curriculum_year,
                    student_semester
                )
            )

            if subject_data.empty:
                st.session_state.pop(
                    "individual_prediction_input", None
                )
                st.session_state.pop(
                    "individual_prediction_result", None
                )

                st.error(
                    "No stored marks were found for this student and subject."
                )

            else:
                row = subject_data.iloc[0]

                st.session_state["individual_prediction_input"] = {
                    "student_id": selected_student,
                    "subject_code": selected_subject,
                    "branch": student_branch,
                    "semester": student_semester,
                    "academic_year": curriculum_year,
                    **{
                        column: row[column]
                        for column in input_columns
                    }
                }

                st.session_state.pop(
                    "individual_prediction_result", None
                )

        saved_input = st.session_state.get(
            "individual_prediction_input"
        )

        if (
            not saved_input
            or saved_input["student_id"] != selected_student
            or saved_input["subject_code"] != selected_subject
            or saved_input["academic_year"] != curriculum_year
            or saved_input["semester"] != student_semester
        ):
            st.info(
                "Click **Load Marks & Predict** to fetch marks from the database."
            )
            st.stop()

    # --------------------------------------------------
    # STUDENT: Predict only from admin-approved submissions.
    # New or corrected marks should be submitted in Student Academic Details.
    # --------------------------------------------------
    else:
        approved_input_df = pd.read_sql(
            """
            SELECT quiz1, quiz2, midsem, assignment, attendance_percentage,
                   prev_sga, prev_sem_percentage, prev_total_backlogs
            FROM student_academic_submissions
            WHERE student_id = %s AND subject_code = %s
              AND academic_year = %s AND semester = %s
              AND status = 'Approved'
            ORDER BY reviewed_at DESC, submitted_at DESC
            LIMIT 1
            """,
            con,
            params=(selected_student, selected_subject, curriculum_year, student_semester)
        )

        if approved_input_df.empty:
            submission_status = pd.read_sql(
                """
                SELECT status, admin_notes
                FROM student_academic_submissions
                WHERE student_id = %s AND subject_code = %s
                  AND academic_year = %s AND semester = %s
                ORDER BY submitted_at DESC
                LIMIT 1
                """,
                con,
                params=(selected_student, selected_subject, curriculum_year, student_semester)
            )
            if submission_status.empty:
                st.info(
                    "No submission exists for this subject yet. Go to **Student Academic Details** "
                    "to submit your marks and attendance for admin validation."
                )
            else:
                status = submission_status.iloc[0]["status"]
                notes = submission_status.iloc[0]["admin_notes"]
                st.warning(
                    f"Your latest submission is **{status}**. Predictions use only approved data."
                )
                if pd.notna(notes) and str(notes).strip():
                    st.write(f"Admin notes: {notes}")
            st.stop()

        approved_row = approved_input_df.iloc[0]
        saved_input = {
            "student_id": selected_student,
            "subject_code": selected_subject,
            "branch": student_branch,
            "semester": student_semester,
            "academic_year": curriculum_year,
            **{column: approved_row[column] for column in input_columns}
        }
        st.success("Using admin-approved academic details for this prediction.")

    # --------------------------------------------------
    # Prepare the final input consistently for both roles.
    # --------------------------------------------------
    prediction_data = saved_input.copy()

    if is_first_semester:
        prediction_data["prev_sga"] = None
        prediction_data["prev_sem_percentage"] = None
        prediction_data["prev_total_backlogs"] = 0

    prediction_input = pd.DataFrame([{
        "subject_code": prediction_data["subject_code"],
        "quiz1": prediction_data["quiz1"],
        "quiz2": prediction_data["quiz2"],
        "midsem": prediction_data["midsem"],
        "assignment": prediction_data["assignment"],
        "attendance_percentage": prediction_data[
            "attendance_percentage"
        ],
        "prev_sga": prediction_data["prev_sga"],
        "prev_sem_percentage": prediction_data[
            "prev_sem_percentage"
        ],
        "prev_total_backlogs": prediction_data[
            "prev_total_backlogs"
        ],
        "branch": prediction_data["branch"],
        "semester": prediction_data["semester"]
    }])

    if role == "admin":
        st.dataframe(
            prediction_input,
            hide_index=True,
            width="stretch"
        )

    # --------------------------------------------------
    # Run the prediction.
    # --------------------------------------------------
    predicted_marks = predict_subject(
        prediction_data["subject_code"],
        prediction_data["quiz1"],
        prediction_data["quiz2"],
        prediction_data["midsem"],
        prediction_data["assignment"],
        prediction_data["attendance_percentage"],
        prediction_data["prev_sga"],
        prediction_data["prev_sem_percentage"],
        prediction_data["prev_total_backlogs"],
        prediction_data["branch"],
        prediction_data["semester"]
    )

    predicted_marks = float(
        np.asarray(predicted_marks).reshape(-1)[0]
    )

    predicted_marks = max(
        0.0,
        min(100.0, predicted_marks)
    )

    prediction_lower = max(
        0.0,
        predicted_marks - prediction_error
    )

    prediction_upper = min(
        100.0,
        predicted_marks + prediction_error
    )

    # PREDICTION INPUT

    # ---------------------------------------------------------

    if show_prediction_input:

        st.subheader("Prediction Input")

        st.dataframe(

            prediction_input,

            width="stretch",

            hide_index=True

        )

    # ---------------------------------------------------------

    # PREDICTION CONFIDENCE

    # ---------------------------------------------------------

    if show_prediction_confidence:

        st.subheader("Prediction Confidence")

        st.metric(

            label="Predicted Marks",

            value=f"{predicted_marks:.2f}/100"

        )

        st.write(

            f"Estimated prediction range: "

            f"{prediction_lower:.2f} – {prediction_upper:.2f}"

        )

        st.caption(

            "This range is estimated from training-set prediction errors. "

            "It is not a guarantee, and actual marks may fall outside it."

        )

    # ---------------------------------------------------------

    # HISTORICAL GRADE DATA

    # ---------------------------------------------------------

    grade_history_query = """

    SELECT

        a.grade,

        COUNT(*) AS student_count,

        MIN(a.marks_obtained) AS min_marks,

        MAX(a.marks_obtained) AS max_marks,

        AVG(a.marks_obtained) AS avg_marks

    FROM academic_records a

    JOIN curriculum c

        ON a.subject_id = c.subject_id

    WHERE a.subject_code = %s

      AND c.branch = %s

      AND c.semester = %s

      AND a.grade NOT IN ('FF', 'FA')

    GROUP BY a.grade

    ORDER BY avg_marks DESC

    """

    grade_history_df = pd.read_sql(

        grade_history_query,

        con,

        params=(

            selected_subject,

            student_branch,

            student_semester

        )

    )

    attendance_percentage = prediction_data["attendance_percentage"]

    if pd.notna(attendance_percentage) and attendance_percentage < 75:

        estimated_grade = "FA"

        possible_grades = ["FA (Attendance below 75%)"]

    elif pd.notna(predicted_marks) and predicted_marks < 40:

        estimated_grade = "FF"

        possible_grades = ["FF (Predicted marks below 40)"]

    elif not grade_history_df.empty:

        grade_history_df["distance"] = (

            grade_history_df["avg_marks"] - predicted_marks

        ).abs()

        estimated_grade = grade_history_df.loc[

            grade_history_df["distance"].idxmin(),

            "grade"

        ]

        possible_grades = []

    else:

        estimated_grade = "N/A"

        possible_grades = []

    if estimated_grade not in ["FA", "FF"]:
        possible_grades = []

    if not grade_history_df.empty and estimated_grade not in ["FA", "FF"]:

        matching_grades_df = grade_history_df[

            (grade_history_df["max_marks"] >= prediction_lower)

            & (grade_history_df["min_marks"] <= prediction_upper)

        ].copy()

        if not matching_grades_df.empty:

            matching_grades_df["distance_to_prediction"] = (

                matching_grades_df["avg_marks"] - predicted_marks

            ).abs()

            matching_grades_df = matching_grades_df.sort_values(

                "distance_to_prediction"

            ).head(3)

            total_count = matching_grades_df["student_count"].sum()

            if total_count > 0:

                possible_grades = [

                    f'{row["grade"]} '

                    f'({row["student_count"] / total_count:.1%})'

                    for _, row in matching_grades_df.iterrows()

                ]

    if not possible_grades:

        possible_grades = ["N/A"]

    historical_subject_query = """

    SELECT

        a.marks_obtained,

        a.grade,

        a.grade_point

    FROM academic_records a

    JOIN curriculum c

        ON a.subject_id = c.subject_id

    WHERE a.subject_code = %s

      AND c.branch = %s

      AND c.semester = %s

    """

    historical_subject_data = pd.read_sql(

        historical_subject_query,

        con,

        params=(

            selected_subject,

            student_branch,

            student_semester

        )

    )

    historical_subject_data = historical_subject_data[

        historical_subject_data["grade"].notna()

        & (historical_subject_data["grade"].str.upper() != "FA")

    ].copy()

    historical_count = (

        int(grade_history_df["student_count"].sum())

        if not grade_history_df.empty

        else 0

    )

    # ---------------------------------------------------------

    # ESTIMATED GRADE AND RELIABILITY

    # ---------------------------------------------------------

    mae = float(model_metrics.get("test mae", 0))

    mean_marks = float(model_metrics.get("mean_training_marks", 0))

    if mean_marks > 0:

        model_reliability = max(

            0.0,

            min(100.0, (1 - mae / mean_marks) * 100)

        )

    else:

        model_reliability = 0.0

    if historical_count >= 100:

        history_reliability = 100

    elif historical_count >= 50:

        history_reliability = 90

    elif historical_count >= 20:

        history_reliability = 75

    else:

        history_reliability = 60

    history_missing = prediction_input[

        ["prev_sga", "prev_sem_percentage", "prev_total_backlogs"]

    ].isna().any(axis=1).iloc[0]

    if history_missing:

        history_reliability -= 15

    prediction_reliability = (

        0.7 * model_reliability

        + 0.3 * history_reliability

    )

    prediction_reliability = max(

        0.0,

        min(100.0, prediction_reliability)

    )

    if show_estimated_grade:

        st.subheader("Estimated Grade")

        st.metric(

            "Grade Based on Historical Performance",

            estimated_grade

        )

        st.write(

            f"**Possible Historical Grades:** {' - '.join(possible_grades)}"

        )

        st.metric(

            "Prediction Reliability",

            f"{prediction_reliability:.1f}%"

        )

        st.caption(

            "Reliability indicator based on model validation performance "

            "and historical data availability. It is not the probability "

            "that the prediction is correct."

        )
    # ---------------------------------------------------------
    # SAVE PREDICTION
    # ---------------------------------------------------------


    if st.button("Save Prediction", key="save_subject_prediction"):


        subject_id_result = pd.read_sql(
            """
            SELECT subject_id
            FROM curriculum
            WHERE TRIM(subject_code) = TRIM(%s)
            AND UPPER(TRIM(branch)) = UPPER(TRIM(%s))
            AND semester = %s
            ORDER BY academic_year DESC
            LIMIT 1
            """,
            con,
            params=(
                selected_subject,
                student_branch,
                student_semester
            )
        )

        if subject_id_result.empty:
            st.error(
                "No curriculum record found for this subject, branch, and semester. "
                "Check the subject code, branch, and semester in the curriculum table."
            )
            st.stop()

        subject_id = int(subject_id_result.iloc[0]["subject_id"])

        existing_prediction = pd.read_sql(
            """
            SELECT prediction_id
            FROM predictions
            WHERE student_id = %s
            AND subject_id = %s
            AND prediction_type = %s
            LIMIT 1
            """,
            con,
            params=(
                str(selected_student),
                int(subject_id),
                "Individual Subject"
            )
        )

        if not existing_prediction.empty:
            st.warning(
                "A prediction has already been saved for this student and subject. "
                "You cannot save another one."
            )


        else:
            subject_id = int(subject_id_result.iloc[0]["subject_id"])

            cursor = con.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO predictions (
                        student_id,
                        subject_id,
                        prediction_type,
                        predicted_marks,
                        predicted_grade,
                        lower_bound,
                        upper_bound,
                        prediction_confidence
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        selected_student,
                        subject_id,
                        "Individual Subject",
                        float(predicted_marks),
                        estimated_grade,
                        float(prediction_lower),
                        float(prediction_upper),
                        float(prediction_reliability)
                    )
                )

                con.commit()


                st.success("Prediction saved successfully!")
                st.rerun()

            except Exception as e:
                con.rollback()
                st.error(f"Could not save prediction: {e}")

            finally:
                cursor.close()

        # ---------------------------------------------------------
    # FACTOR CONTRIBUTIONS
    # ---------------------------------------------------------

    if show_factor_explanation:
        st.subheader("Factors Influencing Prediction")

        try:
            _, explanation_df = predict_subject(
                prediction_data["subject_code"],
                prediction_data["quiz1"],
                prediction_data["quiz2"],
                prediction_data["midsem"],
                prediction_data["assignment"],
                prediction_data["attendance_percentage"],
                prediction_data["prev_sga"],
                prediction_data["prev_sem_percentage"],
                prediction_data["prev_total_backlogs"],
                prediction_data["branch"],
                prediction_data["semester"],
                return_explanation=True
            )

            explanation_df = explanation_df[
                ["feature", "contribution"]
            ].copy()

        except (ValueError, KeyError, AttributeError) as e:
            st.info(f"Factor explanations are unavailable: {e}")
            explanation_df = pd.DataFrame(
                columns=["feature", "contribution"]
            )

        if not explanation_df.empty:
            feature_labels = {
                "num__quiz1": "Quiz 1",
                "num__quiz2": "Quiz 2",
                "num__midsem": "Midsem",
                "num__assignment": "Assignment",
                "num__attendance_percentage": "Attendance",
                "num__prev_sga": "Previous SGA",
                "num__prev_sem_percentage": "Previous Semester Percentage",
                "num__prev_total_backlogs": "Previous Backlogs",
                "num__semester": "Semester"
            }

            branch_names = {
                "CS": "Computer Science",
                "IT": "Information Technology",
                "AI": "Artificial Intelligence",
                "Mech": "Mechanical Engineering",
                "Civil": "Civil Engineering",
                "Chem": "Chemical Engineering",
                "Metallurgy": "Metallurgical Engineering",
                "ECE": "Electronics & Communication",
                "EEE": "Electrical & Electronics",
                "CDS": "Computational Data Science"
            }

            selected_branch_name = branch_names.get(
                student_branch,
                student_branch
            )

            grouped_explanations = []

            for _, row in explanation_df.iterrows():
                feature = row["feature"]
                contribution = row["contribution"]

                if feature.startswith("cat__subject_code_"):
                    label = f"Subject: {selected_subject}"

                elif feature.startswith("cat__branch_"):
                    label = f"Branch: {selected_branch_name}"

                elif feature in feature_labels:
                    label = feature_labels[feature]

                else:
                    label = (
                        feature
                        .replace("num__", "")
                        .replace("cat__", "")
                    )

                grouped_explanations.append({
                    "feature": label,
                    "contribution": contribution
                })

            explanation_df = pd.DataFrame(grouped_explanations)

            explanation_df = (
                explanation_df
                .groupby("feature", as_index=False)["contribution"]
                .sum()
            )

            explanation_df["absolute_contribution"] = (
                explanation_df["contribution"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "absolute_contribution",
                ascending=False
            )

            top_explanations = (
                explanation_df.head(10)
                .sort_values("contribution")
            )

            fig = px.bar(
                top_explanations,
                x="contribution",
                y="feature",
                orientation="h",
                title="Factors Influencing Predicted Marks",
                labels={
                    "contribution": "Linear Contribution",
                    "feature": "Factor"
                }
            )

            fig.add_vline(x=0, line_width=1)

            st.plotly_chart(
                fig,
                width="stretch",
                key="individual_factor_contributions"
            )

            st.caption(
                "Positive contributions push the linear model output upward; "
                "negative contributions push it downward. Contributions use "
                "transformed feature values and model coefficients. The intercept "
                "is not included."
            )

            st.subheader("Individual Factor Explanations")

            factor_descriptions = {

                "Quiz 1": (

                    "The first quiz score. Its contribution shows the "

                    "direction and size of this input's linear term."

                ),

                "Quiz 2": (

                    "The second quiz score. Its contribution reflects "

                    "the learned relationship between this score and marks."

                ),

                "Midsem": (

                    "The mid-semester examination score. Its contribution "

                    "shows how its transformed value affects the model output."

                ),

                "Assignment": (

                    "The assignment score. Its contribution reflects the "

                    "relationship learned from assignment marks during training."

                ),

                "Attendance": (

                    "The attendance percentage supplied to the model. "

                    "Its contribution reflects the learned relationship "

                    "between attendance and marks."

                ),

                "Previous SGA": (

                    "The student's previous SGA, representing earlier "

                    "academic performance used as a prediction input."

                ),

                "Previous Semester Percentage": (

                    "The previous semester's percentage, representing "

                    "earlier academic performance."

                ),

                "Previous Backlogs": (

                    "The number of previous backlogs. Its contribution "

                    "reflects the relationship learned between backlog "

                    "history and marks."

                ),

                "Semester": (

                    "The current semester number. Its contribution reflects "

                    "the relationship learned from semester-related patterns."

                ),

                f"Subject: {selected_subject}": (

                    "The selected subject's encoded category. Its contribution "

                    "represents the learned subject-specific term, if present "

                    "in the model."

                ),

                f"Branch: {selected_branch_name}": (

                    "The student's encoded branch category. Its contribution "

                    "represents the learned branch-specific term, if present "

                    "in the model."

                )

            }

            for _, row in explanation_df.iterrows():

                factor = row["feature"]

                contribution = row["contribution"]

                description = factor_descriptions.get(

                    factor,

                    "This feature was supplied to the model during prediction. "

                    "Its contribution reflects the relationship learned during "

                    "training."

                )

                if contribution > 0:

                    direction = "pushes the prediction upward"

                elif contribution < 0:

                    direction = "pushes the prediction downward"

                else:

                    direction = "has no direct contribution for this input"

                with st.expander(f"{factor}: {contribution:+.2f}"):

                    st.write(description)

                    st.write(

                        f"For this student, this factor {direction} "

                        f"by approximately {abs(contribution):.2f} "

                        "contribution units."

                    )

            positive_factors = explanation_df[

                explanation_df["contribution"] > 0

            ].sort_values(

                "contribution",

                ascending=False

            )

            negative_factors = explanation_df[

                explanation_df["contribution"] < 0

            ].sort_values(

                "contribution"

            )

            col1, col2 = st.columns(2)

            with col1:

                st.write("### Strongest Positive Factors")

                if positive_factors.empty:

                    st.info(

                        "No positive factor contributions for this input."

                    )

                else:

                    for _, row in positive_factors.head(3).iterrows():

                        st.write(

                            f"🟢 **{row['feature']}**: "

                            f"+{row['contribution']:.2f}"

                        )

            with col2:

                st.write("### Strongest Negative Factors")

                if negative_factors.empty:

                    st.info(

                        "No negative factor contributions for this input."

                    )

                else:

                    for _, row in negative_factors.head(3).iterrows():

                        st.write(

                            f"🔴 **{row['feature']}**: "

                            f"{row['contribution']:.2f}"

                        )

# ---------------------------------------------------------

# HISTORICAL PERFORMANCE

# ---------------------------------------------------------

    if show_historical_metrics:

        st.subheader("Historical Performance")

        historical_df = historical_subject_data.copy()

        if historical_df.empty:

            st.info(

                "No historical performance data is available for this subject."

            )

        else:

            historical_df["marks_obtained"] = pd.to_numeric(

                historical_df["marks_obtained"],

                errors="coerce"

            )

            historical_df = historical_df.dropna(

                subset=["marks_obtained"]

            )

            if historical_df.empty:

                st.info(

                    "No valid historical marks are available for comparison."

                )

            else:

                estimated_grade_row = grade_history_df[

                    grade_history_df["grade"] == estimated_grade

                ]

                if not estimated_grade_row.empty:

                    grade_min_marks = estimated_grade_row.iloc[0]["min_marks"]

                    grade_max_marks = estimated_grade_row.iloc[0]["max_marks"]

                    grade_student_count = estimated_grade_row.iloc[0]["student_count"]

                else:

                    grade_min_marks = None

                    grade_max_marks = None

                    grade_student_count = 0

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(

                        f"{estimated_grade} Historical Minimum",

                        f"{grade_min_marks:.0f}"

                        if pd.notna(grade_min_marks)

                        else "N/A"

                    )

                with col2:

                    st.metric(

                        f"{estimated_grade} Historical Maximum",

                        f"{grade_max_marks:.0f}"

                        if pd.notna(grade_max_marks)

                        else "N/A"

                    )

                with col3:

                    st.metric(

                        f"Students with {estimated_grade}",

                        f"{int(grade_student_count)}"

                    )

                historical_percentile = (

                    historical_df["marks_obtained"] < predicted_marks

                ).mean() * 100

                st.metric(

                    "Historical Percentile",

                    f"{historical_percentile:.1f}%"

                )

                if historical_percentile >= 90:

                    percentile_category = "Top 10%"

                elif historical_percentile >= 75:

                    percentile_category = "Above Average"

                elif historical_percentile >= 50:

                    percentile_category = "Average"

                elif historical_percentile >= 25:

                    percentile_category = "Below Average"

                else:

                    percentile_category = "Bottom 25%"

                st.write(

                    f"**Historical Performance Category:** {percentile_category}"

                )

                fig = px.histogram(

                    historical_df,

                    x="marks_obtained",

                    nbins=15,

                    title="Historical Marks Distribution",

                    labels={"marks_obtained": "Marks"}

                )

                fig.add_vline(

                    x=predicted_marks,

                    line_width=3,

                    line_color="#00E5FF",

                    line_dash="solid",

                    annotation_text=f"Predicted: {predicted_marks:.1f}",

                    annotation_position="top"

                )

                fig.add_vline(

                    x=prediction_lower,

                    line_width=3,

                    line_color="#FFD166",

                    line_dash="dot",

                    annotation_text=f"Lower: {prediction_lower:.1f}",

                    annotation_position="bottom"

                )

                fig.add_vline(

                    x=prediction_upper,

                    line_width=3,

                    line_color="#FF6B6B",

                    line_dash="dot",

                    annotation_text=f"Upper: {prediction_upper:.1f}",

                    annotation_position="bottom"

                )

                x_min = min(

                    historical_df["marks_obtained"].min(),

                    prediction_lower

                )

                x_max = max(

                    historical_df["marks_obtained"].max(),

                    prediction_upper

                )

                fig.update_xaxes(

                    range=[max(0, x_min - 2), min(100, x_max + 2)]

                )

                st.plotly_chart(

                    fig,

                    width="stretch",

                    key="historical_marks_distribution"

                )

                grade_distribution = (

                    historical_df["grade"]

                    .dropna()

                    .value_counts()

                    .reindex(

                        ["AA", "AB", "BB", "BC", "CC", "CD", "DD", "FF", "FA"],

                        fill_value=0

                    )

                    .rename_axis("grade")

                    .reset_index(name="student_count")

                )

                if not grade_distribution.empty:

                    grade_distribution["highlight"] = (

                        grade_distribution["grade"] == estimated_grade

                    )

                    fig = px.bar(

                        grade_distribution,

                        x="grade",

                        y="student_count",

                        title="Historical Grade Distribution",

                        labels={

                            "grade": "Grade",

                            "student_count": "Number of Students"

                        },

                        color="highlight",

                        color_discrete_map={

                            True: "white",

                            False: "#444444"

                        },

                        category_orders={

                            "grade": [

                                "AA", "AB", "BB", "BC", "CC",

                                "CD", "DD", "FF", "FA"

                            ]

                        }

                    )

                    fig.update_layout(showlegend=False)

                    st.plotly_chart(

                        fig,

                        width="stretch",

                        key="historical_grade_distribution"

                    )

    # ---------------------------------------------------------

    # SUBJECT-WISE GRADE CUTOFFS

    # ---------------------------------------------------------

    matching_grades = []

    if show_grade_cutoffs or show_grade_ranges:

        subject_cutoffs_df = pd.read_sql(

            """

            SELECT

                grade,

                MIN(marks_obtained) AS minimum_marks,

                MAX(marks_obtained) AS maximum_marks

            FROM academic_records

            WHERE subject_code = %s

              AND grade <> 'FA'

            GROUP BY grade, grade_point

            ORDER BY grade_point DESC

            """,

            con,

            params=(selected_subject,)

        )

        if not subject_cutoffs_df.empty:

            subject_cutoffs_df["minimum_marks"] = pd.to_numeric(

                subject_cutoffs_df["minimum_marks"],

                errors="coerce"

            )

            subject_cutoffs_df["maximum_marks"] = pd.to_numeric(

                subject_cutoffs_df["maximum_marks"],

                errors="coerce"

            )

            subject_cutoffs_df = subject_cutoffs_df.dropna(

                subset=["minimum_marks", "maximum_marks"]

            )

        if show_grade_cutoffs:

            st.subheader("Subject-wise Grade Cutoffs")

            if subject_cutoffs_df.empty:

                st.info(

                    "No historical grade cutoffs are available for this subject."

                )

            else:

                st.dataframe(

                    subject_cutoffs_df.rename(columns={

                        "grade": "Grade",

                        "minimum_marks": "Minimum Historical Marks",

                        "maximum_marks": "Maximum Historical Marks"

                    }),

                    width="stretch",

                    hide_index=True

                )

    # ---------------------------------------------------------

    # PREDICTED MARKS VS HISTORICAL GRADE RANGES

    # ---------------------------------------------------------

    if show_grade_ranges:

        st.subheader("Predicted Marks vs Historical Grade Ranges")

        if subject_cutoffs_df.empty:

            st.info(

                "No historical grade ranges are available for comparison."

            )

        else:

            fig_comparison = go.Figure()

            grade_colors = {

                "AA": "#2ca02c",

                "AB": "#4da6ff",

                "BB": "#f1c40f",

                "BC": "#ff9933",

                "CC": "#ff6677",

                "CD": "#9966cc",

                "DD": "#999999",

                "FF": "#d62728",

                "FA": "#8c564b"

            }

            for _, row in subject_cutoffs_df.iterrows():

                grade = row["grade"]

                minimum = float(row["minimum_marks"])

                maximum = float(row["maximum_marks"])

                contains_prediction = (

                    minimum <= predicted_marks <= maximum

                )

                if contains_prediction:

                    matching_grades.append(grade)

                fig_comparison.add_trace(

                    go.Bar(

                        x=[grade],

                        y=[maximum - minimum],

                        base=[minimum],

                        name=grade,

                        marker_color=grade_colors.get(grade, "#aaaaaa"),

                        opacity=1.0 if contains_prediction else 0.45,

                        showlegend=False,

                        customdata=[[minimum, maximum]],

                        hovertemplate=(

                            "Grade: %{x}\<br>"

                            "Minimum: %{customdata[0]:.2f}\<br>"

                            "Maximum: %{customdata[1]:.2f}"

                            "\<extra>\</extra>"

                        )

                    )

                )

            fig_comparison.add_hline(

                y=predicted_marks,

                line_color="red",

                line_width=2,

                line_dash="dash",

                annotation_text=f"Predicted marks: {predicted_marks:.2f}",

                annotation_position="top left"

            )

            fig_comparison.update_layout(

                title=(

                    "Predicted Marks vs Historical Grade Ranges — "

                    f"{selected_subject}"

                ),

                xaxis_title="Grade",

                yaxis_title="Marks",

                yaxis=dict(range=[0, 100]),

                barmode="overlay"

            )

            st.plotly_chart(

                fig_comparison,

                width="stretch",

                key="predicted_marks_historical_grade_ranges"

            )

            if matching_grades:

                st.success(

                    f"Predicted marks of {predicted_marks:.2f} fall within "

                    f"the historical range for grade(s): "

                    f"{', '.join(matching_grades)}."

                )

            else:

                st.warning(

                    f"Predicted marks of {predicted_marks:.2f} fall outside "

                    "all observed historical grade ranges."

                )

    # ---------------------------------------------------------

    # PREDICTION SUMMARY

    # ---------------------------------------------------------

    if show_prediction_summary:

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "Predicted Marks",

                f"{predicted_marks:.2f}/100"

            )

        with col2:

            st.metric(

                "Estimated Grade",

                estimated_grade

            )

        with col3:

            historical_match = (

                ", ".join(matching_grades)

                if matching_grades

                else "Outside observed ranges"

            )

            st.metric(

                "Historical Range Match",

                historical_match

            )

    # ---------------------------------------------------------

    # PREDICTION UNCERTAINTY

    # ---------------------------------------------------------

    if show_uncertainty:

        st.subheader("Prediction Uncertainty")

        residual_std_value = float(residual_std)

        prediction_lower_uncertainty = max(

            0.0,

            predicted_marks - 1.96 * residual_std_value

        )

        prediction_upper_uncertainty = min(

            100.0,

            predicted_marks + 1.96 * residual_std_value

        )

        st.metric(

            "Approximate Prediction Range",

            f"{prediction_lower_uncertainty:.2f} – "

            f"{prediction_upper_uncertainty:.2f}"

        )

        st.caption(

            "This is an approximate range based on the model's residual "

            "standard deviation, assuming roughly normal and stable prediction "

            "errors. It is not a guaranteed interval."

        )

    # ---------------------------------------------------------

    # MODEL VALIDATION SUMMARY

    # ---------------------------------------------------------

    if show_model_validation:

            st.subheader("Model Validation Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric("MAE", f"{model_metrics['test mae']:.2f}")

            with col2:

                st.metric("RMSE", f"{model_metrics['test rmse']:.2f}")

            with col3:

                st.metric("R² Score", f"{model_metrics['test r2']:.4f}")

            with col4:

                st.metric("Mean Marks", f"{model_metrics['mean_training_marks']:.2f}")

            st.caption(

                "These metrics summarize model evaluation performance. "

                "They do not establish the confidence of an individual prediction."

            )
    if st.button("New Prediction", key="new_individual_prediction"):
        st.session_state.pop("individual_prediction_input", None)
        st.session_state.pop("individual_prediction_result", None)
        st.rerun()

else:
    st.session_state.individual_prediction_page_active = False

if page == "Student Risk Analysis":
    st.title("Student Academic Risk Analysis")

    risk_query = """
    SELECT
        s.student_id,
        sr.academic_year,
        sr.semester,
        sr.sga,
        sr.semester_percentage,
        sr.total_backlogs,
        sr.semester_status
    FROM students s
    JOIN semester_results sr
        ON s.student_id = sr.student_id
    ORDER BY s.student_id, sr.academic_year, sr.semester
    """

    risk_df = pd.read_sql(risk_query, con)

    if risk_df.empty:
        st.warning("No semester records available for risk analysis.")

    else:
        risk_df["sga_change"] = (
            risk_df.groupby("student_id")["sga"].diff()
        )

        risk_df["risk_factors"] = (
            (risk_df["sga"] < 5) |
            (risk_df["sga_change"] <= -1) |
            (risk_df["total_backlogs"] >= 3) |
            (risk_df["semester_percentage"] < 50)
        )

        risk_df["Risk Indicators"] = risk_df["risk_factors"].apply(
            lambda value: (
                "Needs Review" if value
                else "No Flagged Indicators"
            )
        )

        risk_summary = risk_df.groupby("student_id").agg(
            Semesters_Recorded=("semester", "count"),
            Flagged_Semesters=("risk_factors", "sum"),
            Latest_SGA=("sga", "last"),
            Latest_Backlogs=("total_backlogs", "last"),
            Latest_Percentage=("semester_percentage", "last")
        ).reset_index()

        risk_summary["Risk Level"] = risk_summary["Flagged_Semesters"].apply(
            lambda count: (
                "High Review Priority" if count >= 3
                else "Needs Review" if count >= 1
                else "No Flagged Indicators"
            )
        )

        risk_summary = risk_summary.sort_values(
            by=["Flagged_Semesters", "Latest_Backlogs"],
            ascending=[False, False]
        )

        backlog_history_df = pd.read_csv(
            "backlog_history_demo_download.csv"
        )

        backlog_counts = (
            backlog_history_df.assign(
                is_pending=backlog_history_df["clearance_status"].eq("Pending"),
                is_cleared=backlog_history_df["clearance_status"].eq("Cleared")
            )
            .groupby("student_id")
            .agg(
                Total_Recorded_Backlogs=("student_id", "size"),
                Cleared_Backlogs=("is_cleared", "sum"),
                Pending_Backlogs=("is_pending", "sum")
            )
            .reset_index()
        )

        risk_summary["student_id"] = risk_summary["student_id"].astype(str)
        backlog_counts["student_id"] = backlog_counts["student_id"].astype(str)

        risk_summary = risk_summary.merge(
            backlog_counts,
            on="student_id",
            how="left"
        )

        for col in [
            "Total_Recorded_Backlogs",
            "Cleared_Backlogs",
            "Pending_Backlogs"
        ]:
            risk_summary[col] = risk_summary[col].fillna(0).astype(int)

        risk_summary["Risk Level"] = risk_summary.apply(
            lambda row: (
                "High Review Priority"
                if row["Pending_Backlogs"] >= 3
                or row["Flagged_Semesters"] >= 3
                else "Needs Review"
                if row["Pending_Backlogs"] >= 1
                or row["Flagged_Semesters"] >= 1
                else "No Flagged Indicators"
            ),
            axis=1
        )

        # 1. Student Risk Summary
        if show_risk_summary:
            st.subheader("Student Risk Summary")

            st.dataframe(
                risk_df[
                    [
                        "student_id",
                        "academic_year",
                        "semester",
                        "sga",
                        "sga_change",
                        "semester_percentage",
                        "total_backlogs",
                        "Risk Indicators"
                    ]
                ],
                width="stretch"
            )

        # 2. Semester Risk Indicators
        if show_semester_indicators:
            st.subheader("Semester Risk Indicators")

            risk_levels = [
                "High Review Priority",
                "Needs Review",
                "No Flagged Indicators"
            ]

            selected_risk_levels = st.multiselect(
                "Filter by Risk Level",
                options=risk_levels,
                default=risk_levels
            )

            filtered_risk_summary = risk_summary[
                risk_summary["Risk Level"].isin(selected_risk_levels)
            ]

            student_search = st.text_input("Search by Student ID")

            if student_search:
                filtered_risk_summary = filtered_risk_summary[
                    filtered_risk_summary["student_id"]
                    .astype(str)
                    .str.contains(student_search, case=False, na=False)
                ]

            st.dataframe(
                filtered_risk_summary,
                width="stretch"
            )

if page == "Historical Comparison":
    st.title("Historical Academic Performance")
    st.caption(
        "Track semester-wise SGA, academic percentage, and backlog trends."
    )
    st.divider()

    student_ids = students_df["student_id"].tolist()

    selected_student = st.selectbox(
        "Select Student for Historical Comparison",
        student_ids,
        key="historical_student"
    )

    semester_average_query = """
        SELECT
            academic_year,
            semester,
            AVG(sga) AS average_semester_sga
        FROM semester_results
        GROUP BY academic_year, semester
        ORDER BY academic_year, semester
        """

    semester_average_df = pd.read_sql(
        semester_average_query,
        con
    )

    history_query = """
    SELECT
        sr.academic_year,
        sr.semester,
        sr.sga,
        sr.total_backlogs,
        sr.semester_status,
        (
            SELECT AVG(sr2.sga)
            FROM semester_results sr2
            WHERE sr2.academic_year = sr.academic_year
            AND sr2.semester = sr.semester
        ) AS average_semester_sga
    FROM semester_results sr
    WHERE sr.student_id = %s
    ORDER BY sr.academic_year, sr.semester
    """

    history_df = pd.read_sql(
        history_query,
        con,
        params=(selected_student,)
    )

    if history_df.empty:
        st.warning("No historical semester records found for this student.")

    else:
        history_df["Semester Label"] = (
            history_df["academic_year"].astype(str)
            + " - Sem "
            + history_df["semester"].astype(str)
        )

        

        st.markdown("### Performance Overview")

        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Semesters",
                len(history_df)
            )

            col2.metric(
                "Average SGA",
                f"{history_df['sga'].mean():.2f}/10"
            )

            col3.metric(
                "Highest SGA",
                f"{history_df['sga'].max():.2f}/10"
            )

            col4.metric(
                "Total Backlogs",
                int(history_df["total_backlogs"].sum())
            )

        history_df["Historical Average SGA"] = history_df.apply(
            lambda row: semester_average_df.loc[
                (semester_average_df["academic_year"] == row["academic_year"])
                & (semester_average_df["semester"] == row["semester"]),
                "average_semester_sga"
            ].iloc[0],
            axis=1
        )

        history_df["Change from Previous Semester"] = (
            history_df["sga"].diff()
        )
        history_df["SGA Change"] = history_df["sga"].diff()


        cga_query = """
        SELECT
            student_id,
            academic_year,
            semester,
            sga
        FROM semester_results
        ORDER BY student_id, academic_year, semester
        """

        cga_df = pd.read_sql(cga_query, con)

        semester_average_query = """
        SELECT
            academic_year,
            semester,
            AVG(sga) AS average_semester_sga
        FROM semester_results
        GROUP BY academic_year, semester
        ORDER BY academic_year, semester
        """

        semester_average_df = pd.read_sql(
            semester_average_query,
            con
        )
        

        if show_sga_trend:
            st.subheader("Your SGA vs Semester Average")

            sga_chart_df = history_df.set_index("Semester Label")[
                ["sga", "average_semester_sga"]
            ].rename(columns={
                "sga": "Your SGA",
                "average_semester_sga": "Average SGA of All Students"
            })

            fig = px.line(
                sga_chart_df.reset_index(),
                x="Semester Label",
                y=["Your SGA", "Average SGA of All Students"],
                markers=True,
                labels={
                    "value": "SGA",
                    "Semester Label": "Semester",
                    "variable": "Series"
                },
                title="Your SGA vs Semester Average"
            )

            fig.update_yaxes(range=[0, 10], dtick=1)
            fig.update_layout(height=400)

            st.plotly_chart(fig, width="stretch")

        if show_cga_tracker:
            st.subheader("Your CGA vs Historical CGA Average")

            cga_df["CGA"] = (
                cga_df.groupby("student_id")["sga"]
                .transform(lambda values: values.expanding().mean())
            )

            cga_df["Semester Label"] = (
                cga_df["academic_year"].astype(str)
                + " - Sem "
                + cga_df["semester"].astype(str)
            )

            selected_cga = cga_df[
                cga_df["student_id"] == selected_student
            ].copy()

            average_cga = (
                cga_df.groupby(["academic_year", "semester"])["CGA"]
                .mean()
                .reset_index(name="Average CGA of All Students")
            )

            selected_cga = selected_cga.merge(
                average_cga,
                on=["academic_year", "semester"],
                how="left"
            )

            selected_cga_chart = selected_cga.set_index("Semester Label")[
                ["CGA", "Average CGA of All Students"]
            ].rename(columns={
                "CGA": "Your CGA",
                "Average CGA of All Students": "Average CGA"
            })

            fig = px.line(
                selected_cga_chart.reset_index(),
                x="Semester Label",
                y=["Your CGA", "Average CGA"],
                markers=True,
                labels={
                    "value": "CGA",
                    "Semester Label": "Semester",
                    "variable": "Series"
                },
                title="Your CGA vs Historical CGA Average"
            )

            fig.update_yaxes(range=[0, 10], dtick=1)
            fig.update_layout(height=400)

            st.plotly_chart(fig, width="stretch")

        if show_backlog_history:
            st.markdown("### Backlog History")

            with st.container(border=True):
                st.caption("Review recorded backlogs for each semester.")

                st.bar_chart(
                    history_df.set_index("Semester Label")[["total_backlogs"]],
                    y="total_backlogs",
                    height=300
                )

        if show_semester_table:
            semester_details_df = history_df[
                [
                    "Semester Label",
                    "sga",
                    "Historical Average SGA",
                    "Change from Previous Semester",
                    "total_backlogs",
                    "semester_status"
                ]
            ].rename(columns={
                "sga": "Your SGA",
                "total_backlogs": "Total Backlogs",
                "semester_status": "Semester Status"
            })

            semester_details_df["Historical Average SGA"] = (
                semester_details_df["Historical Average SGA"].round(2)
            )

            semester_details_df["Change from Previous Semester"] = (
                semester_details_df["Change from Previous Semester"].apply(
                    lambda value: "—" if pd.isna(value) else f"{value:+.2f}"
                )
            )

            st.dataframe(
                semester_details_df,
                hide_index=True,
                width="stretch"
            )



if page == "Backlog Tracker":
    st.subheader("Student Backlog Tracker")

    backlog_query = """
        SELECT
            student_id,
            subject_code,
            semester,
            academic_year,
            marks_obtained,
            maximum_marks,
            grade,
            backlog
        FROM academic_records
        WHERE backlog = 1
        ORDER BY student_id, semester, subject_code
    """

    with con.cursor() as cursor:
        cursor.execute(backlog_query)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()

    backlog_df = pd.DataFrame(rows, columns=columns)

    col1, col2 = st.columns(2)

    col1.metric("Total Backlog Records", len(backlog_df))
    col2.metric(
        "Students with Backlogs",
        backlog_df["student_id"].nunique()
    )

    student_filter = st.text_input("Filter by Student ID")

    if student_filter:
        backlog_df = backlog_df[
            backlog_df["student_id"].astype(str).str.contains(
                student_filter, case=False, na=False
            )
        ]

    if backlog_df.empty:
        st.info("No backlog records found for the selected filter.")
    else:
        st.dataframe(backlog_df, hide_index=True, width="stretch")

    st.download_button(
        "Download Backlog Report",
        data=backlog_df.to_csv(index=False).encode("utf-8"),
        file_name="backlog_report.csv",
        mime="text/csv"
    )

if page == "Prediction Validation":
    st.subheader("Prediction Validation & Feedback")


    # Load predictions
    validation_query = """
    SELECT
        p.prediction_id,
        p.student_id,
        p.subject_id,
        c.subject_code,
        c.subject_name,
        p.prediction_type,
        p.predicted_marks,
        p.predicted_grade,
        p.created_at
    FROM predictions p
    LEFT JOIN curriculum c
        ON p.subject_id = c.subject_id
    WHERE p.predicted_marks IS NOT NULL
    """

    if role == "student":
        validation_query += " AND p.student_id = %s"
        predictions_df = pd.read_sql(
            validation_query + " ORDER BY p.created_at DESC",
            con,
            params=(str(student_id),)
        )
    else:
        predictions_df = pd.read_sql(
            validation_query + " ORDER BY p.created_at DESC",
            con
        )

    # Load existing validation records
    validated_query = """
    SELECT prediction_id
    FROM prediction_feedback
    """

    validated_df = pd.read_sql(validated_query, con)

    validated_ids = set(
        validated_df["prediction_id"].astype(int).tolist()
    )

    if predictions_df.empty:
        st.info(
            "No saved predictions are available for validation yet. "
            "Generate and save an individual subject prediction first."
        )

    # Assign validation status when saved predictions exist.
    predictions_df["Validation Status"] = predictions_df[
        "prediction_id"
    ].apply(
        lambda prediction_id: (
            "Validated"
            if int(prediction_id) in validated_ids
            else "Pending Validation"
        )
    )

    # Calculate counts before filtering
    all_status_counts = predictions_df["Validation Status"].value_counts()

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Pending Validation",
        int(all_status_counts.get("Pending Validation", 0))
    )

    col2.metric(
        "Total Validated",
        int(all_status_counts.get("Validated", 0))
    )

    # Filter predictions
    if show_validation_filters:
        status_filter = st.multiselect(
            "Filter by Validation Status",
            options=["Pending Validation", "Validated"],
            default=["Pending Validation", "Validated"],
            key="validation_status_multiselect"
        )

        filtered_predictions_df = predictions_df[
            predictions_df["Validation Status"].isin(status_filter)
        ].copy()
    else:
        filtered_predictions_df = predictions_df.copy()

    if filtered_predictions_df.empty:
        st.info(
            "No predictions match the selected validation status."
        )

    else:
        st.subheader("Predictions Available for Validation")
        st.dataframe(
            filtered_predictions_df,
            width="stretch"
        )

        # Prediction validation form

        if show_validation_form:
            st.divider()
            st.subheader("Validate an Individual Prediction")

            pending_predictions = filtered_predictions_df[
                filtered_predictions_df["Validation Status"]
                == "Pending Validation"
            ]

            if pending_predictions.empty:
                st.info("No pending predictions available for validation.")

            else:
                selected_prediction = st.selectbox(
                    "Select Prediction ID",
                    pending_predictions["prediction_id"].tolist(),
                    key="validation_prediction_select"
                )

                selected_row = pending_predictions[
                    pending_predictions["prediction_id"]
                    == selected_prediction
                ].iloc[0]

                prediction_id = int(selected_row["prediction_id"])

                st.write(f"**Student ID:** {selected_row['student_id']}")
                st.write(
                    f"**Subject:** {selected_row['subject_code']} — "
                    f"{selected_row['subject_name']}"
                )
                st.write(
                    f"**Prediction Type:** {selected_row['prediction_type']}"
                )
                st.write(
                    f"**Predicted Marks:** {selected_row['predicted_marks']}"
                )
                st.write(
                    f"**Predicted Grade:** {selected_row['predicted_grade']}"
                )

                with st.form(key=f"validation_form_{prediction_id}"):
                    actual_marks = st.number_input(
                        "Actual Final Marks",
                        min_value=0.0,
                        max_value=100.0,
                        value=50.0,
                        step=0.5,
                        key=f"actual_marks_{prediction_id}"
                    )

                    grade_options = [
                        "AA", "AB", "BB", "BC", "CC",
                        "CD", "DD", "FF", "FA"
                    ]

                    actual_grade = st.selectbox(
                        "Actual Final Grade",
                        options=["Select a grade"] + grade_options,
                        index=0,
                        key=f"actual_grade_{prediction_id}"
                    )

                    feedback_notes = st.text_area(
                        "Feedback Notes",
                        placeholder="Optional observations about the prediction",
                        key=f"feedback_notes_{prediction_id}"
                    )

                    confirm_validation = st.checkbox(
                        "I confirm these are the actual marks and grade.",
                        key=f"confirm_validation_{prediction_id}"
                    )

                    save_validation = st.form_submit_button(
                        "Save Validation"
                    )

                if save_validation:
                    if not confirm_validation:
                        st.warning("Please confirm the actual marks and grade.")

                    elif actual_grade == "Select a grade":
                        st.warning("Please select the actual final grade.")

                    else:
                        absolute_error = abs(
                            float(selected_row["predicted_marks"])
                            - float(actual_marks)
                        )

                        try:
                            cursor.execute(
                                """
                                SELECT COUNT(*)
                                FROM prediction_feedback
                                WHERE prediction_id = %s
                                """,
                                (prediction_id,)
                            )

                            existing_count = cursor.fetchone()[0]

                            if existing_count > 0:
                                st.warning(
                                    "This prediction already has a feedback "
                                    "record. Refresh the page and select a "
                                    "pending prediction."
                                )

                            else:
                                cursor.execute(
                                    """
                                    INSERT INTO prediction_feedback
                                    (
                                        prediction_id,
                                        actual_marks,
                                        actual_grade,
                                        absolute_error,
                                        feedback_notes
                                    )
                                    VALUES (%s, %s, %s, %s, %s)
                                    """,
                                    (
                                        prediction_id,
                                        float(actual_marks),
                                        actual_grade,
                                        absolute_error,
                                        feedback_notes
                                    )
                                )

                                con.commit()

                                st.success(
                                    "Validation saved successfully. "
                                    f"Absolute error: {absolute_error:.2f} marks."
                                )

                                st.rerun()

                        except Exception as e:
                            con.rollback()
                            st.error(f"Could not save validation: {e}")

    # Validation results
    if show_validation_results:
        st.divider()
        st.subheader("Prediction Validation Results")

        results_query = """
        SELECT
            pf.prediction_id,
            p.student_id,
            c.subject_code,
            pf.actual_marks,
            p.predicted_marks,
            pf.absolute_error,
            pf.feedback_date AS validated_at,
            pf.actual_grade,
            p.predicted_grade,
            pf.feedback_notes
        FROM prediction_feedback pf
        JOIN predictions p
            ON pf.prediction_id = p.prediction_id
        LEFT JOIN curriculum c
            ON p.subject_id = c.subject_id
        ORDER BY pf.feedback_date DESC
        """

        results_df = pd.read_sql(results_query, con)

        if results_df.empty:
            st.info("No predictions have been validated yet.")

        else:
            # Grade accuracy
            grade_order = [
                "AA", "AB", "BB", "BC", "CC",
                "CD", "DD", "FF", "FA"
            ]

            grade_to_rank = {
                grade: rank
                for rank, grade in enumerate(grade_order)
            }

            grade_comparison_df = results_df[
                results_df["actual_grade"].isin(grade_order)
                & results_df["predicted_grade"].isin(grade_order)
            ].copy()

            if not grade_comparison_df.empty:
                exact_matches = (
                    grade_comparison_df["actual_grade"]
                    == grade_comparison_df["predicted_grade"]
                ).sum()

                grade_accuracy = (
                    exact_matches / len(grade_comparison_df)
                ) * 100

                grade_comparison_df["actual_rank"] = (
                    grade_comparison_df["actual_grade"]
                    .map(grade_to_rank)
                )

                grade_comparison_df["predicted_rank"] = (
                    grade_comparison_df["predicted_grade"]
                    .map(grade_to_rank)
                )

                within_one_band = (
                    (
                        grade_comparison_df["actual_rank"]
                        - grade_comparison_df["predicted_rank"]
                    ).abs() <= 1
                ).mean() * 100

                col1, col2 = st.columns(2)

                col1.metric(
                    "Exact Grade Accuracy",
                    f"{grade_accuracy:.1f}%"
                )

                col2.metric(
                    "Accuracy Within One Grade Band",
                    f"{within_one_band:.1f}%"
                )

            else:
                st.info(
                    "No validated grade comparisons are available yet."
                )

            # Marks error metrics
            total_validations = len(results_df)
            mae = results_df["absolute_error"].mean()
            median_error = results_df["absolute_error"].median()

            within_5_marks = (
                results_df["absolute_error"] <= 5
            ).mean() * 100

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Validated Predictions",
                total_validations
            )

            col2.metric(
                "Mean Absolute Error",
                f"{mae:.2f} marks"
            )

            col3.metric(
                "Median Absolute Error",
                f"{median_error:.2f} marks"
            )

            col4.metric(
                "Within 5 Marks",
                f"{within_5_marks:.1f}%"
            )

            # Actual vs predicted chart
            st.subheader("Actual vs Predicted Marks")

            chart_df = results_df[
                ["prediction_id", "actual_marks", "predicted_marks"]
            ].copy()

            st.scatter_chart(
                chart_df,
                x="actual_marks",
                y="predicted_marks"
            )

            # Validation records
            st.subheader("Validation Records")

            st.dataframe(
                results_df,
                width="stretch"
            )

            # Download report
            st.download_button(
                "Download Validation Report",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_validation_report.csv",
                mime="text/csv"
            )
