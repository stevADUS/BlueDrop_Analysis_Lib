import datetime
import sqlite3

def convert_to_iso_datetime(dt):
    """
    Convert a datetime.datetime object to ISO 8601 formatted datetime string.
    
    Parameters:
        dt (datetime.datetime): The datetime object to convert.
    
    Returns:
        str: The ISO 8601 formatted datetime string.
    """
    return dt.isoformat()

def table_exists(table_name, db_name):
    try:
        # Connect to the database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Query SQLite system tables to check if the table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        result = cursor.fetchone()

        # Close the connection
        conn.close()

        # If the result is not None, the table exists
        return result is not None
    except sqlite3.Error as e:
        print("Error checking table existence:", e)
        return False
    
# Example usage:
if __name__ == "__main__":
    pass