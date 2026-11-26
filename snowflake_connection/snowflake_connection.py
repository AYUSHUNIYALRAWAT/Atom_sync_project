
import os
from pathlib import Path

import snowflake.connector
from dotenv import load_dotenv


# Load environment variables from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def get_connection():
    """
    Establish and return a Snowflake connection.
    """

    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        authenticator="OAUTH_AUTHORIZATION_CODE",
        client_store_temporary_credential=False,
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )

    return conn


# Test connection when this file is executed directly
if __name__ == "__main__":

    conn = None

    try:
        conn = get_connection()

        print("Snowflake connection successful!")

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                CURRENT_USER(),
                CURRENT_DATABASE(),
                CURRENT_SCHEMA()
        """)

        print("Connection details:", cursor.fetchone())

        cursor.close()

    except Exception as error:
        print("Connection failed:", error)

    finally:
        if conn:
            conn.close()

