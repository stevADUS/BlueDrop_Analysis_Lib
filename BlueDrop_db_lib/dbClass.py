import sqlite3
from BlueDrop_db_lib.db_general_functions import convert_to_iso_datetime, table_exists

import sqlite3
from datetime import datetime
import numpy as np

class PFFPDatabase:
    def __init__(self, db_name):
        self.db_name = db_name

        # Init dict to hold the table names so I don't have to remember them
        self.table_names = {"drops":"Drops", "accel": "AccelerometerData"}
    
    def load_database(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            print(f"Database '{self.db_name}' loaded successfully.")
        except sqlite3.Error as e:
            print("Error loading database:", e)

    def create_database(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()

            # Create the Drops table with a unique constraint on drop_name and drop_datetime
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS Drops (
                                        drop_id TEXT PRIMARY KEY,
                                        drop_name TEXT,
                                        drop_datetime TEXT,
                                        survey_name TEXT,
                                        survey_id TEXT,
                                        analysis_date,
                                        UNIQUE (drop_name, drop_datetime, survey_id)
                                    )''')
            print(f"Database '{self.db_name}' created successfully.")
        except sqlite3.Error as e:
            print("Error creating database:", e)

    def add_drop_data(self, drop, survey_name, survey_id):

        # Unpack the information from the drop
        drop_name = drop.name
        drop_datetime = convert_to_iso_datetime(drop.datetime)
        current_time = datetime.now()
        analysis_date = convert_to_iso_datetime(current_time)

        try:
            # Generate a unique drop_id using current timestamp and random value
            drop_id = datetime.now().strftime('%Y%m%d%H%M%S%f')  # Example format: '20220430120155012345'

            # Insert the drop data into the Drops table
            self.cursor.execute('''INSERT INTO Drops (drop_id, drop_name, drop_datetime, survey_name, survey_id, analysis_date)
                                   VALUES (?, ?, ?, ?, ?, ?)''',
                                 (drop_id, drop_name, drop_datetime, survey_name, survey_id, analysis_date))
            
            # Add the data to the other tables
            self.add_data_to_other_tables(drop_id, drop, survey_name= survey_name,survey_id= survey_id)

            self.conn.commit()
            print("Drop data added successfully.")
        except sqlite3.Error as e:
            print("Error adding drop data:", e)
    
    def add_data_to_other_tables(self, drop_id, drop, survey_name, survey_id):
        try:
            # Check if the accel table exists
            accel_name = self.table_names["accel"]
            accel_table_exist = table_exists(table_name = accel_name, db_name= self.db_name)
            
            # Create the table if it doesn't exist
            if not accel_table_exist:
                self.create_accelerometer_table()
            
            # Add the data to the accel table
            self.add_accelerometer_data(drop_id=drop_id, drop = drop)

            print("Accelerometer data added successfully.")
        except sqlite3.Error as e:
            print("Error adding accelerometer data:", e)

    def create_accelerometer_table(self):
        try:
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS AccelerometerData (
                                    drop_id TEXT,
                                    drop_timestamp TEXT,
                                    impulse_start_index INTEGER,
                                    impulse_time BLOB,
                                    impulse_accel BLOB,    
                                    release_start_index INTEGER,
                                    release_time BLOB,
                                    release_accel BLOB,
                                    peak_deceleration REAL,
                                    PRIMARY KEY (drop_id, drop_timestamp)
                                )''')
            
            print("Accelerometer table created successfully.")
        except sqlite3.Error as e:
            print("Error creating accelerometer table:", e)

    def view_entries(self, table_name):
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_name)
            cursor = conn.cursor()

            # Execute a SELECT statement to retrieve data from the specified table
            cursor.execute(f"SELECT * FROM {table_name}")

            # Fetch all rows returned by the SELECT statement
            rows = cursor.fetchall()

            # Print the rows
            for row in rows:
                print(row)

            # Close the connection
            conn.close()

        except sqlite3.Error as e:
            print("Error viewing entries:", e)

    def add_accelerometer_data(self, drop_id, drop):
        try:
            timestamp = convert_to_iso_datetime(drop.datetime)

            # Extract impulse and release time and acceleration data from DataFrames
            impulse_time_bytes = drop.impulse_df["Time"].to_numpy().tobytes()
            impulse_accel_bytes = drop.impulse_df["accel"].to_numpy().tobytes()
            release_time_bytes = drop.release_df["Time"].to_numpy().tobytes()
            release_accel_bytes = drop.release_df["accel"].to_numpy().tobytes()
            
            # Store the indices
            impulse_start_index = drop.drop_indices["start_impulse_index"]
            relase_start_index = drop.drop_indices["release_index"]

            # Insert or replace data into AccelerometerData table
            self.cursor.execute('''INSERT OR REPLACE INTO AccelerometerData (
                                        drop_id, drop_timestamp, impulse_start_index, impulse_time, 
                                        impulse_accel, release_start_index, release_time,
                                        release_accel, peak_deceleration
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (
                                    drop_id, timestamp, impulse_start_index, impulse_time_bytes, 
                                    impulse_accel_bytes, relase_start_index, release_time_bytes, 
                                    release_accel_bytes, drop.peak_deceleration
                                ))
            self.conn.commit()
            print("Accelerometer data added successfully.")
        except sqlite3.Error as e:
            print("Error adding accelerometer data:", e)

    def close_database(self):
        try:
            self.conn.close()
            print(f"Database '{self.db_name}' closed successfully.")
        except sqlite3.Error as e:
            print("Error closing database:", e)


# Example usage:
if __name__ == "__main__":
    # Instantiate the PFFPDatabase class
    pffp_db = PFFPDatabase("example.db")

    # Load an existing database or create a new one
    pffp_db.create_database()

    # Create the accelerometer table
    pffp_db.create_accelerometer_table()

    # Add sample accelerometer data
    # pffp_db.add_accelerometer_data("drop123", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0.1, 0.2, 0.3)

    # Close the database connection
    pffp_db.close_database()

    pffp_db.view_entries("AccelerometerData")