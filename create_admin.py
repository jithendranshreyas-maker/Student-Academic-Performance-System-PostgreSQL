
import getpass
import bcrypt
import psycopg2
from auth2 import get_connection


def main():
    username = input("Admin username: ").strip()
    email = input("Admin email: ").strip().lower()

    if not username or not email:
        print("Username and email are required.")
        return

    password = getpass.getpass("Admin password (minimum 8 characters): ")
    confirm = getpass.getpass("Confirm admin password: ")

    if len(password) < 8:
        print("Password must be at least 8 characters.")
        return

    if password != confirm:
        print("Passwords do not match.")
        return

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    con = get_connection()

    try:
        with con.cursor() as cursor:
            cursor.execute(
                """
                SELECT user_id
                FROM user_accounts
                WHERE username = %s OR email = %s
                """,
                (username, email)
            )

            if cursor.fetchone():
                print("That username or email already exists.")
                con.rollback()
                return

            cursor.execute(
                """
                INSERT INTO user_accounts
                    (student_id, username, email, password_hash,
                     role, is_active)
                VALUES (NULL, %s, %s, %s, 'admin', TRUE)
                """,
                (username, email, password_hash)
            )

        con.commit()
        print("Admin account created successfully.")

    except psycopg2.Error as error:
        con.rollback()
        print("Admin account creation failed:", error)

    finally:
        con.close()


if __name__ == "__main__":
    main()

