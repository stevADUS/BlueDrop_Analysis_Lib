import sqlite3
from BlueDrop_db_lib.db_general_functions import convert_to_iso_datetime, table_exists, get_bearing_names

import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd

class PFFPDatabase:
    def __init__(self, db_name):
        """
        Initialize the PFFPDatabase class.

        Parameters:
        ----------
        db_name : str
            The name of the SQLite database.

        Notes:
        ------
        Tries to load the database and creates it if it doesn't exist. Initializes a dictionary
        to hold table names for easy reference. Creates the 'Drops' table if it doesn't already exist.
        """
        self.db_name = db_name
        # Try to load the database and create it if it doesn't exist
        self.load_database()

        # Init dict to hold the table names so I don't have to remember them
        self.table_names = {"drops": "Drops", "accel": "AccelerometerData", "bearing": "BearingCapacityData", 
                            "pffp_calib": "PffpCalibration", "survey": "Survey", "pffp_config": "PffpConfiguration"}

        # Create the Drops table if it doesn't already exist
        self.create_database()

    def load_database(self):
        """
        Load the SQLite database.

        Raises:
        ------
        sqlite3.Error: If there is an error connecting to the database.

        Notes:
        ------
        Attempts to connect to the SQLite database specified by 'db_name'. If successful, initializes
        a cursor object to interact with the database. Prints a success message upon successful connection,
        or prints an error message and the exception if connection fails.
        """
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            print(f"Database '{self.db_name}' loaded successfully.")
        except sqlite3.Error as e:
            print("Error loading database:", e)

    def create_database(self):
        """
        Create the database and the Drops table if they do not already exist.

        Raises:
        ------
        sqlite3.Error: If there is an error creating the database or table.

        Notes:
        ------
        Tries to create the Drops table with a unique constraint on 'drop_name' and 'survey_id' columns
        if it does not exist already in the specified SQLite database ('db_name'). Prints a success
        message upon successful creation, or prints an error message and the exception if creation fails.
        """
        try:
            # Create the Drops table with a unique constraint on drop_name and survey_id
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
        """
        Add drop data to the Drops table and related tables in the SQLite database.

        Parameters:
        ----------
        drop : Drop
            The drop object containing drop information.
        pffp_calibration_dict : dict
            Dictionary containing PFFP calibration data.
        location_dict : dict
            Dictionary containing location/survey information.
        pffp_config_dict : dict
            Dictionary containing PFFP configuration data.

        Notes:
        ------
        Unpacks information from the drop object and location_dict to add data to the Drops table.
        Generates a unique drop_id using the current timestamp and random value. Inserts drop data into
        the Drops table and calls add_data_to_other_tables to add data to related tables (AccelerometerData,
        BearingCapacityData, PffpCalibration, Survey, PffpConfiguration). Commits changes to the database
        upon successful insertion or prints an error message and the exception if insertion fails.
        """
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
            self.add_data_to_other_tables(drop_id, drop, pffp_calibration_dict=pffp_calibration_dict, location_dict=location_dict, 
                                          pffp_config_dict=pffp_config_dict)

            self.conn.commit()
            print("Drop data added successfully.")
        except sqlite3.Error as e:
            print("Error adding drop data:", e)

    def add_data_to_other_tables(self, drop_id, drop, pffp_calibration_dict, location_dict, pffp_config_dict):
        """
        Add data to related tables in the SQLite database.

        Parameters:
        ----------
        drop_id : str
            Unique identifier for the drop.
        drop : Drop
            The drop object containing drop information.
        pffp_calibration_dict : dict
            Dictionary containing PFFP calibration data.
        location_dict : dict
            Dictionary containing location/survey information.
        pffp_config_dict : dict
            Dictionary containing PFFP configuration data.

        Notes:
        ------
        Checks if tables (AccelerometerData, BearingCapacityData, PffpCalibration, Survey, PffpConfiguration) exist
        in the database, and creates them if they do not. Adds data to each table if it exists in the corresponding drop
        object or dictionary. Commits changes to the database upon successful insertion or prints an error message and
        the exception if insertion fails.
        """
        try:
            #------------ Acceleration Table ------------ 
            # Check if the accel table exists
            accel_name = self.table_names["accel"]
            accel_table_exist = table_exists(table_name=accel_name, db_name=self.db_name)
            
            # Create the table if it doesn't exist
            if not accel_table_exist:
                self.create_accelerometer_table()
            
            # Add the data to the accel table
            self.add_accelerometer_data(drop_id=drop_id, drop=drop)

            #------------ Bearing Table ------------ 
            # Make sure that the table exists
            bearing_table_name = self.table_names["bearing"]
            bearing_table_exist = table_exists(table_name=bearing_table_name, db_name=self.db_name)
            
            if not bearing_table_exist:
                self.create_bearing_capacity_table() 

            # Add bearing data to the database if it exists
            for name in ["projected", "mantle"]:
                df = drop.bearing_dfs.get(name)
                if isinstance(df, pd.DataFrame):
                    self.add_bearing_data(drop_id=drop_id, area_type=name, drop=drop, df=df)
            
            #------------ Calibration Table ------------ 
            # Setup the calibration table
            calibration_name = self.table_names["pffp_calib"]
            calibration_table_exist = table_exists(table_name=calibration_name, db_name=self.db_name)

            if not calibration_table_exist:
                self.create_pffp_calibration_table()

            # Add the calibration data to the table
            self.add_pffp_calibration_data(drop_id=drop_id, pffp_calibration_dict=pffp_calibration_dict)

            #------------ Survey Table ------------
            survey_name = self.table_names["survey"]
            survey_table_exist = table_exists(table_name=survey_name, db_name=self.db_name)

            if not survey_table_exist:
                self.create_survey_table()
            
            # Add the survey data to the table
            self.add_survey_data(drop_id, location_dict)

            #------------ PFFP Configuration Table ------------
            pffp_config_name = self.table_names["pffp_config"]
            pffp_config_table_exist = table_exists(table_name=pffp_config_name, db_name=self.db_name)

            if not pffp_config_table_exist:
                self.create_pffp_configuration_table()
            
            # Add the pffp_config data to the table
            self.add_pffp_configuration_data(drop_id, config_dict=pffp_config_dict)

        except sqlite3.Error as e:
            print("Error adding data:", e)

    def create_accelerometer_table(self):
        """
        Create the AccelerometerData table in the SQLite database.

        Notes:
        ------
        Creates the table with columns for drop ID, drop timestamp, impulse and release acceleration data, and peak
        deceleration. Sets the drop ID and drop timestamp as the primary key. Prints a success message upon
        successfully creating the table, or prints an error message and the exception if table creation fails.
        """
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
        """
        Create the BearingCapacityData table in the SQLite database.

        Notes:
        ------
        Creates the table with columns for drop ID, drop name, area type, contact area name and data, dynamic name and
        data, and multiple columns for QSBC (Quasi-Static Bearing Capacity) name and data. Sets the drop ID and area type
        as the primary key. Prints a success message upon successfully creating the table, or prints an error message and
        the exception if table creation fails.
        """
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

    def create_pffp_calibration_table(self):
        """
        Create the PffpCalibration table in the SQLite database.

        Notes:
        ------
        Creates the table with columns for drop ID, PFFP ID, calibration name, and various accelerometer offsets and scales.
        Sets the drop ID as the primary key. Prints a success message upon successfully creating the table, or prints an
        error message and the exception if table creation fails.
        """
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
            print("PffpCalibration table created successfully.")

        except sqlite3.Error as e:
            print("Error creating PffpCalibration table:", e)

    def create_pffp_configuration_table(self):
        """
        Create the PffpConfiguration table in the SQLite database.

        Notes:
        ------
        Creates the table with columns for drop ID, PFFP ID, configuration details such as mass, height, and radius,
        and units for various measurements. Sets the drop ID as the primary key. Prints a success message upon
        successfully creating the table, or prints an error message and the exception if table creation fails.
        """
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
        """
        View all entries in a specified table in the SQLite database.

        Parameters:
        -----------
        table_name : str
            The name of the table from which to retrieve entries.

        Notes:
        ------
        Connects to the SQLite database, executes a SELECT statement to retrieve all rows from the specified table,
        prints each row, and then closes the database connection. If an error occurs during this process, prints
        an error message along with the exception.

        """
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
        Retrieve a table from the SQLite database and return it as a pandas DataFrame.

        Parameters:
        -----------
        name : str
            The logical name of the table as defined in `self.table_names`.

        Returns:
        --------
        df : pandas.DataFrame
            A DataFrame containing all rows from the specified table.

        Notes:
        ------
        Executes an SQL query to select all rows from the specified table, fetches the data into a pandas DataFrame,
        and returns the DataFrame. If an error occurs during this process, it prints an error message along with the exception.

        """
        try:
            # Get the actual table name from the dictionary
            table_name = self.table_names[name]

            # SQL query to select all rows from the table
            sql_query = f"SELECT * FROM {table_name}"

            # Fetch the data into a pandas DataFrame
            df = pd.read_sql_query(sql_query, self.conn)

            return df

        except sqlite3.Error as e:
            print("Error retrieving table as DataFrame:", e)

    def add_accelerometer_data(self, drop_id, drop):
        """
        Add accelerometer data to the AccelerometerData table.

        Parameters:
        -----------
        drop_id : str
            Unique identifier for the drop.
        drop : Drop
            Drop object containing accelerometer data.

        Notes:
        ------
        Extracts impulse and release time and acceleration data from Drop object DataFrames,
        stores indices, and inserts or replaces data into AccelerometerData table. Commits
        changes to the database and prints success message upon completion. If an error occurs,
        prints an error message along with the exception.

        """
        try:
            # Convert drop datetime to ISO format
            timestamp = convert_to_iso_datetime(drop.datetime)

            # Extract impulse and release time and acceleration data from DataFrames
            impulse_time_bytes = drop.impulse_df["Time"].to_numpy().tobytes()
            impulse_accel_bytes = drop.impulse_df["accel"].to_numpy().tobytes()
            release_time_bytes = drop.release_df["Time"].to_numpy().tobytes()
            release_accel_bytes = drop.release_df["accel"].to_numpy().tobytes()
            
            # Store the indices
            impulse_start_index = drop.drop_indices["start_impulse_index"]
            release_start_index = drop.drop_indices["release_index"]

            # Insert or replace data into AccelerometerData table
            self.cursor.execute('''INSERT OR REPLACE INTO AccelerometerData (
                                        drop_id, drop_timestamp, impulse_start_index, impulse_time, 
                                        impulse_accel, release_start_index, release_time,
                                        release_accel, peak_deceleration
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (
                                    drop_id, timestamp, impulse_start_index, impulse_time_bytes, 
                                    impulse_accel_bytes, release_start_index, release_time_bytes, 
                                    release_accel_bytes, drop.peak_deceleration
                                ))
            self.conn.commit()
            print("Accelerometer data updated/added successfully.")
        except sqlite3.Error as e:
            print("Error adding accelerometer data:", e)

    def add_bearing_data(self, drop_id, drop, df, area_type):
        """
        Add bearing data to the BearingCapacityData table.

        Parameters:
        -----------
        drop_id : str
            Unique identifier for the drop.
        drop : Drop
            Drop object containing bearing data.
        df : pd.DataFrame
            DataFrame containing bearing data.
        area_type : str
            Type of area (e.g., 'projected', 'mantle').

        Notes:
        ------
        Extracts and serializes bearing data from DataFrames, constructs SQL query,
        prepares data for insertion, executes SQL query, commits changes to the database,
        and prints success message upon completion. If an error occurs, prints an error
        message along with the exception.

        """
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
        """
        Add PFFP calibration data to the PffpCalibration table.

        Parameters:
        -----------
        drop_id : str
            Unique identifier for the drop.
        pffp_calibration_dict : dict
            Dictionary containing PFFP calibration data.

        Notes:
        ------
        Retrieves PFFP ID and calibration DataFrame from dictionary, extracts relevant
        column names, iterates over rows of the DataFrame to prepare data for insertion,
        constructs SQL query, executes SQL query, commits changes to the database, and
        prints success message upon completion. If an error occurs, prints an error
        message along with the exception.

        """
        try:
            pffp_id = pffp_calibration_dict["pffp_id"]
            pffp_calibration_df = pffp_calibration_dict["pffp_calibration_df"]

            # Get the column names
            column_names = pffp_calibration_df.columns
            offset_col = [s for s in column_names if "_offset" in s][0]
            scale_col = [s for s in column_names if "_scale" in s][0]
            calib_name = offset_col.replace("_offset", "")

            # Store the drop id, bluedrop id and calibration name to the drop
            data = [drop_id, pffp_id, calib_name]

            # Iterate over rows of the DataFrame and insert data into the table
            for _, row in pffp_calibration_df.iterrows():
                offset_value = row[offset_col]
                scale_value = row[scale_col]
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
        """
        Add PFFP configuration data to the PffpConfiguration table.

        Parameters:
        -----------
        drop_id : str
            Unique identifier for the drop.
        config_dict : dict
            Dictionary containing PFFP configuration data.

        Notes:
        ------
        Retrieves tip type, properties DataFrame, and column names from dictionary,
        iterates over rows of the DataFrame to prepare data for insertion, constructs
        SQL query, executes SQL query, commits changes to the database, and prints
        success message upon completion. If an error occurs, prints an error message
        along with the exception.

        """
        try:
            # Extract values from config_dict
            tip_type = config_dict['tip_type']
            tip_props_df = config_dict['tip_props']
            df_column_names = tip_props_df.columns
            tip_col_name = config_dict['tip_col_name']

            # Initialize data list with drop_id and tip_type
            data = [drop_id, tip_type]

            # Extract value column based on tip_type
            value_col = [s for s in df_column_names if tip_type in s][0]

            # Extract units and values from DataFrame
            units = list(tip_props_df["units"])
            values = list(tip_props_df[value_col])

            # Add units and values to data list
            data = data + units + values

            # Add tip_col_name to data list
            data.append(tip_col_name)

            # SQL query to insert or replace data into PffpConfiguration table
            insert_query = '''
                INSERT OR REPLACE INTO PffpConfiguration (
                    drop_id, pffp_id, tip_type, pffp_mass_unit, tip_height_unit,
                    base_radius_unit, tip_radius_unit, pffp_volume_unit,
                    pffp_mass, tip_height, base_radius, tip_radius,
                    radius_coeff, pffp_volume, tip_col_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''

            # Execute the SQL query
            self.cursor.execute(insert_query, data)
            self.conn.commit()
            print("PffpConfiguration updated successfully.")

        except sqlite3.Error as e:
            print("Error updating PffpConfiguration:", e)

    def create_survey_table(self):
        """
        Create the Survey table in the SQLite database.

        Notes:
        ------
        Attempts to create a new table 'Survey' in the database,
        with columns for drop_id, survey_id, survey_name, location_name,
        transect, latitude, and longitude, as well as the primary key
        for drop_id. If an error occurs during creation, an error message
        is printed.

        """
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
        """
        Add survey data to the Survey table in the SQLite database.

        Parameters:
        -----------
        drop_id : str
            The drop ID associated with the survey data.
        location_dict : dict
            A dictionary containing survey information including survey_id,
            survey_name, location_name, transect, latitude, and longitude.

        Notes:
        ------
        Attempts to insert survey data into the 'Survey' table of the SQLite database.
        If an error occurs during insertion, an error message is printed.

        """
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
        """
        Close the SQLite database connection.

        Notes:
        ------
        Attempts to close the database connection. Prints a success message upon
        successful closure of the connection. If an error occurs during closure,
        an error message is printed.

        """
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