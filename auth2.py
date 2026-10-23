
import os
import bcrypt
import psycopg2
import streamlit as st
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv(".env2", override=False)


def get_connection():
    try:
        db = st.secrets["database"]
    except (KeyError, FileNotFoundError):
        db = None

    if db is not None:
        return psycopg2.connect(
            host=db["host"],
            port=int(db.get("port", 5432)),
            user=db["user"],
            password=db["password"],
            dbname=db["name"],
            connect_timeout=10
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME"),
        connect_timeout=10
    )


def verify_password(password, password_hash):
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )


def login(username, password):
    con = get_connection()

    try:
        with con.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT user_id, student_id, username, email,
                       password_hash, role, is_active
                FROM user_accounts
                WHERE username = %s
                """,
                (username,)
            )
            user = cursor.fetchone()

        if (
            user
            and user["is_active"]
            and verify_password(password, user["password_hash"])
        ):
            st.session_state["authenticated"] = True
            st.session_state["user"] = {
                "user_id": user["user_id"],
                "student_id": user["student_id"],
                "username": user["username"],
                "role": user["role"]
            }
            return True

        return False

    finally:
        con.close()


def signup_student(
    name, username, email, password, confirm_password,
    phone="", section="", branch="",
    current_year=None, current_semester=None,
    academic_year=None
):
    name = name.strip()
    username = username.strip()
    email = email.strip().lower()
    phone = phone.strip()
    section = section.strip()
    branch = branch.strip()

    if not all([name, username, email, password, confirm_password]):
        return False, "Please fill in every required field."

    if password != confirm_password:
        return False, "Passwords do not match."

    if len(password) < 8:
        return False, "Password must be at least 8 characters long."

    con = get_connection()

    try:
        with con.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT user_id
                FROM user_accounts
                WHERE username = %s OR email = %s
                """,
                (username, email)
            )

            if cursor.fetchone():
                return False, "Username or email is already registered."

            cursor.execute("SELECT student_id FROM students")
            existing_ids = {
                row["student_id"] for row in cursor.fetchall()
            }

            number = 1
            while f"STU{number:06d}" in existing_ids:
                number += 1

            student_id = f"STU{number:06d}"

            password_hash = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt()
            ).decode("utf-8")

            cursor.execute(
                """
                INSERT INTO students (
                    student_id, name, email, phone, section, branch,
                    current_year, current_semester, academic_year
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    student_id, name, email, phone or None,
                    section or None, branch or None,
                    current_year, current_semester, academic_year
                )
            )

            cursor.execute(
                """
                INSERT INTO user_accounts (
                    student_id, username, email, password_hash,
                    role, is_active
                )
                VALUES (%s, %s, %s, %s, 'student', TRUE)
                """,
                (student_id, username, email, password_hash)
            )

        con.commit()

        return (
            True,
            f"Account created successfully! Your Student ID is {student_id}."
        )

    except psycopg2.Error:
        con.rollback()
        return False, (
            "Registration failed. Please check your details "
            "or contact the administrator."
        )

    finally:
        con.close()
