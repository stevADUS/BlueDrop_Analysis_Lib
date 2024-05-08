import sqlite3
from BlueDrop_db_lib.db_general_functions import convert_to_iso_datetime, table_exists, get_bearing_names

import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd

class PFFPDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        # Try to load the database and create it if it doens't exist
        self.load_database()

        # Init dict to hold the table names so I don't have to remember them
        self.table_names = {"drops":"Drops", "accel": "AccelerometerData", "bearing":"BearingCapacityData", 
                            "pffp_calib":"PffpCalibration", "survey":"Survey", "pffp_config":"PffpConfiguration"}

        # Create the Drop table if it doesn't already exist
        self.create_database()

    def load_database(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            print(f"Database '{self.db_name}' loaded successfully.")
        except sqlite3.Error as e:
            print("Error loading database:", e)

    def create_database(self):
        try:
            # Create the Drops table with a unique constraint on drop_name and drop_datetime
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS Drops (
                                        drop_id TEXT PRIMARY KEY,
                                        drop_name TEXT,
                                        drop_datetime TEXT,
                                        survey_name TEXT,
                                        survey_id TEXT,
                                        analysis_date TEXT,
                                        water_drop INTEGER,
                                        UNIQUE (drop_name, survey_id)
                                    )''')
            print(f"Database '{self.db_name}' created successfully.")
        except sqlite3.Error as e:
            print("Error creating database:", e)

    def add_drop_data(self, drop, pffp_calibration_dict, location_dict, pffp_config_dict):

        # Unpack the information from the drop
        drop_name = drop.name
        drop_datetime = convert_to_iso_datetime(drop.datetime)
        current_time = datetime.now()
        analysis_date = convert_to_iso_datetime(current_time)
        water_drop = drop.water_drop

        # Unpack the survey values
        survey_name = location_dict["survey_name"]
        survey_id = location_dict["survey_id"]

        try:
            # Generate a unique drop_id using current timestamp and random value
            drop_id = datetime.now().strftime('%Y%m%d%H%M%S%f')  # Example format: '20220430120155012345'

            # Insert the drop data into the Drops table
            self.cursor.execute('''INSERT INTO Drops (drop_id, drop_name, drop_datetime, survey_name, survey_id, analysis_date, water_drop)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                 (drop_id, drop_name, drop_datetime, survey_name, survey_id, analysis_date, water_drop))
            
            # Add the data to the other tables
            self.add_data_to_other_tables(drop_id, drop, pffp_calibration_dict=pffp_calibration_dict, location_dict = location_dict, 
                                          pffp_config_dict = pffp_config_dict)

            self.conn.commit()
            print("Drop data added successfully.")
        except sqlite3.Error as e:
            print("Error adding drop data:", e)
    
    def add_data_to_other_tables(self, drop_id, drop, pffp_calibration_dict, location_dict, pffp_config_dict):
        try:
            #------------ Acceleration Table ------------ 
            # Check if the accel table exists
            accel_name = self.table_names["accel"]
            accel_table_exist = table_exists(table_name = accel_name, db_name= self.db_name)
            
            # Create the table if it doesn't exist
            if not accel_table_exist:
                self.create_accelerometer_table()
            
            # Add the data to the accel table
            self.add_accelerometer_data(drop_id=drop_id, drop = drop)

            #------------ Bearing Table ------------ 
            # Make sure that the table exists
            bearing_table_name = self.table_names["bearing"]
            bearing_table_exist = table_exists(table_name=bearing_table_name, db_name = self.db_name)
            
            if not bearing_table_exist:
                self.create_bearing_capacity_table() 

            # Add bearing data to the database if it exists
            for name in ["projected", "mantle"]:
                df=  drop.bearing_dfs[name]
                if isinstance(df, pd.DataFrame):
                    self.add_bearing_data(drop_id=drop_id, area_type=name, drop = drop, df = df)
            
            #------------ Calibration Table ------------ 
            # Setup the calibration table
            calibration_name = self.table_names["pffp_calib"]
            calibration_table_exist = table_exists(table_name=calibration_name, db_name=self.db_name)

            if not calibration_table_exist:
                self.create_pffp_calbration_table()

            # Add the calibration data to the table
            self.add_pffp_calibration_data(drop_id=drop_id, pffp_calibration_dict=pffp_calibration_dict)

            #------------ survey Table ------------
            survey_name = self.table_names["survey"]
            survey_table_exist = table_exists(table_name=survey_name, db_name=self.db_name)

            if not survey_table_exist:
                self.create_survey_table()
            
            # Add the survey data to the table
            self.add_survey_data(drop_id, location_dict)

            #------------ survey Table ------------
            pffp_config_name = self.table_names["pffp_config"]
            pffp_config_table_exist = table_exists(table_name=pffp_config_name, db_name=self.db_name)

            if not pffp_config_table_exist:
                self.create_pffp_configuration_table()
            
            # Add the pffp_config data to the table
            self.add_pffp_configuration_data(drop_id, config_dict=pffp_config_dict)

        except sqlite3.Error as e:
            print("Error adding data:", e)

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
   
    def create_bearing_capacity_table(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS BearingCapacityData (
                drop_id TEXT,
                drop_name TEXT,
                area_type TEXT,
                contact_area_name TEXT,
                contact_area_data BLOB,
                dynamic_name TEXT,
                dynamic_data BLOB,
                qsbc_name_1 TEXT,
                qsbc_data_1 BLOB,
                qsbc_name_2 TEXT,
                qsbc_data_2 BLOB,
                qsbc_name_3 TEXT,
                qsbc_data_3 BLOB,
                qsbc_name_4 TEXT,
                qsbc_data_4 BLOB,
                qsbc_name_5 TEXT,
                qsbc_data_5 BLOB,
                PRIMARY KEY (drop_id, area_type)
            )
            ''')
            self.conn.commit()
            print("BearingCapacityData table created successfully.")
        except sqlite3.Error as e:
            print("Error creating BearingCapacityData table:", e)
   
    def create_pffp_calbration_table(self):
        try:
            # SQL statement to create the table
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS PffpCalibration (
                    drop_id TEXT,
                    pffp_id TEXT,
                    calib_name TEXT,
                    "2g_accel_offset" REAL,
                    "2g_accel_scale" REAL,
                    "18g_accel_offset" REAL,
                    "18g_accel_scale" REAL,
                    "50g_accel_offset" REAL,
                    "50g_accel_scale" REAL,
                    "200g_accel_offset" REAL,
                    "200g_accel_scale" REAL,
                    "250g_accel_offset" REAL,
                    "250g_accel_scale" REAL,
                    "55g_x_tilt_offset" REAL,
                    "55g_x_tilt_scale" REAL,
                    "55g_y_tilt_offset" REAL,
                    "55g_y_tilt_scale" REAL,
                    "pore_pressure_offset" REAL,	
                    "pore_pressure_scale"  REAL,	
                    PRIMARY KEY (drop_id)
                );
            '''

            # Execute the SQL statement
            self.cursor.execute(create_table_query)
            self.conn.commit()
            print("Accelerometer calibration table created successfully.")

        except sqlite3.Error as e:
            print("Error creating accelerometer calibration table:", e)
    
    def create_pffp_configuration_table(self):
        try:
            # SQL query to create the table
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS PffpConfiguration (
                    drop_id TEXT,
                    pffp_id INTEGER,
                    tip_type TEXT,
                    pffp_mass_unit TEXT,
                    tip_height_unit TEXT,
                    base_radius_unit TEXT,
                    tip_radius_unit TEXT,
                    pffp_volume_unit TEXT,
                    pffp_mass REAL,
                    tip_height REAL,
                    base_radius REAL,
                    tip_radius REAL,
                    radius_coeff REAL,
                    pffp_volume REAL,
                    tip_col_name TEXT,
                    PRIMARY KEY (drop_id)
                );
            '''
            # Execute the SQL query
            self.cursor.execute(create_table_query)
            self.conn.commit()
            print("PffpConfiguration table created successfully.")
        except sqlite3.Error as e:
            print("Error creating PffpConfiguration table:", e)

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
   
    def get_table_as_df(self, name):
        """
        Purpose: View a table as a df
        """
        table_name = self.table_names[name]

        # SQL query to select all rows from the table
        sql_query = "SELECT * FROM {}".format(table_name)

        # Fetch the data into a pandas DataFrame
        df = pd.read_sql_query(sql_query, self.conn)

        return df
    
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
            print("Accelerometer data updated/added successfully.")
        except sqlite3.Error as e:
            print("Error adding accelerometer data:", e)
    
    def add_bearing_data(self, drop_id, drop, df, area_type):
        try:
            drop_name = drop.name
            columns = df.columns
            contact_area_name, dynamic_name, qsbc_names = get_bearing_names(columns)

            # Convert qsbc_names list to a dictionary for easier processing
            qsbc_name_dict = {f'qsbc_name_{i}': name for i, name in enumerate(qsbc_names, start=1)}

            # Serialize data before storing in database
            contact_area_data_serialized = df[contact_area_name].to_numpy().tobytes()
            dynamic_data_serialized = df[dynamic_name].to_numpy().tobytes()
            qsbc_data_serialized = {f'qsbc_data_{i}': df[name].to_numpy().tobytes() for i, name in enumerate(qsbc_names, start=1)}

            # Construct the SQL query
            sql_query = f'''
                INSERT OR REPLACE INTO BearingCapacityData (
                    drop_id, drop_name, area_type, contact_area_name, contact_area_data,
                    dynamic_name, dynamic_data, 
                    {", ".join(qsbc_name_dict.keys())},
                    {", ".join(qsbc_data_serialized.keys())}
                ) VALUES (?, ?, ?, ?, ?, ?, ?, {", ".join(["?" for _ in qsbc_names])}, {", ".join(["?" for _ in qsbc_data_serialized])})
            '''

            # Prepare data for insertion
            data = (drop_id, drop_name, area_type, contact_area_name, contact_area_data_serialized,
                    dynamic_name, dynamic_data_serialized,
                    *qsbc_name_dict.values(), 
                    *qsbc_data_serialized.values())

            # Execute the SQL query
            self.cursor.execute(sql_query, data)
            self.conn.commit()
            print("Bearing capacity data updated/added successfully.")
        except sqlite3.Error as e:
            print("Error updating bearing capacity data:", e)

    def add_pffp_calibration_data(self, drop_id, pffp_calibration_dict):
        try:
            
            pffp_id = pffp_calibration_dict["pffp_id"]
            pffp_calibration_df =pffp_calibration_dict["pffp_calibration_df"]

            # Get the column names
            column_names = pffp_calibration_df.columns
            offset_col = [s for s in column_names if "_offset" in s][0]
            scale_col = [s for s in column_names if "_scale" in s][0]
            calib_name = offset_col.replace("_offset", "")

            # Store the drop id, bluedrop id and calibration name to the drop
            data = [drop_id, pffp_id, calib_name]
            
            # Iterate over rows of the DataFrame and insert data into the table
            for _, row in pffp_calibration_df.iterrows():
                # sensor = row['Sensor']
                offset_value = row[offset_col]
                scale_value = row[scale_col]

                # Add the offset value and the scale value to the list
                data.extend([offset_value, scale_value])

            # Construct the SQL query
            sql_query = '''
                INSERT INTO PffpCalibration (
                    drop_id, pffp_id, calib_name,
                    "2g_accel_offset", "2g_accel_scale",
                    "18g_accel_offset", "18g_accel_scale",
                    "50g_accel_offset", "50g_accel_scale",
                    "200g_accel_offset", "200g_accel_scale",
                    "250g_accel_offset", "250g_accel_scale",
                    "55g_x_tilt_offset", "55g_x_tilt_scale",
                    "55g_y_tilt_offset", "55g_y_tilt_scale",
                    "pore_pressure_offset", "pore_pressure_scale"
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
                
            # Execute the SQL query
            self.cursor.execute(sql_query, data)

            # Commit the transaction and close the connection
            self.conn.commit()
            print("Accelerometer calibration data added successfully.")

        except sqlite3.Error as e:
            print("Error adding accelerometer calibration data:", e)
    
    def add_pffp_configuration_data(self, drop_id, config_dict):
        try:
            # Extract values from config_dict
            tip_type = config_dict['tip_type']
            tip_props_df = config_dict['tip_props']
            df_column_names = tip_props_df.columns

            tip_col_name = config_dict['tip_col_name']

  
            data = [drop_id, tip_type]

            value_col = [s for s in df_column_names if tip_type in s][0]

            units= list(tip_props_df["units"])
            values = list(tip_props_df[value_col])
            
            # Add the units a
            data = data + units + values
            
            # Add the tip_col_name
            data.append(tip_col_name)

            insert_query = '''
            INSERT OR REPLACE INTO PffpConfiguration (
                drop_id, pffp_id, tip_type, pffp_mass_unit, tip_height_unit,
                base_radius_unit, tip_radius_unit, pffp_volume_unit,
                pffp_mass, tip_height, base_radius, tip_radius,
                radius_coeff, pffp_volume, tip_col_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)            
            '''

            self.cursor.execute(insert_query, data)
            self.conn.commit()
            print("PffpConfiguration updated successfully.")

        except sqlite3.Error as e:
            print("Error updating PffpConfiguration:", e)

    def create_survey_table(self):
        try:
            # SQL query to create the Survey table
            create_table_query = '''
                CREATE TABLE IF NOT EXISTS Survey (
                    drop_id TEXT,
                    survey_id TEXT,
                    survey_name TEXT,
                    location_name TEXT,
                    transect TEXT,
                    latitude REAL,
                    longitude REAL,
                    PRIMARY KEY (drop_id)
                );
            '''
            # Execute the SQL query
            self.cursor.execute(create_table_query)
            self.conn.commit()
            print("Survey table created successfully.")
        except sqlite3.Error as e:
            print("Error creating Survey table:", e)

    def add_survey_data(self, drop_id, location_dict):
        try:
            # Extract values from location_dict
            survey_id = location_dict["survey_id"]
            survey_name = location_dict["survey_name"]
            location_name = location_dict["location_name"]
            transect = location_dict["transect"]
            latitude = location_dict["latitude"]
            longitude = location_dict["longitude"]

            # Insert data into the Survey table
            insert_query = '''
                INSERT INTO Survey (drop_id, survey_id, survey_name, location_name, transect, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            '''
            data = (drop_id, survey_id, survey_name, location_name, transect, latitude, longitude)
            self.cursor.execute(insert_query, data)
            self.conn.commit()
            print("Survey data added successfully.")
        except sqlite3.Error as e:
            print("Error adding survey data:", e)

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