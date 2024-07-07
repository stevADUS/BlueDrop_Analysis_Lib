# To-Do list for the BlueDrop Analysis Libary - Database side
This is a list of things that need to be done to make the Database side of the Library testable and usable

## Testing that needs to be done
1) Testing the creation of the database table
2) db_general_functions
3) dbClass
   
## Documentation that needs to be written
1) db_general_functions
2) dbClass

## Things that need to be collected
1) A complete dataset of work so that the database can be filled.
   * Ideally this dataset should include enough data so that all of the tables can be tested

## Examples that need to be done
1) Example of loading/putting data into the database
2) Example of extracting data from the database

## Code that needs to be reorganized
1) Might work on making some of the table. Possibly move some of the table creating to another file

## Code that needs to be written
1) Table to store the unit soil parameters
2) Table to store the pressure data
3) Table to store the raw data
4) Table to store the grain size data
5) Make it so multiple databases can be joined together.
    * I think the easist way to build this database is going to be on the project level
    * Then you need to be able to concatenate the data from the project level into one large database
6) Functions to interact with the database and conversate with it

## Design Safe
1) Look into the structue of designsafe and load a database onto design safe using design safe
2) Set up a jupyter notebook so that project data can be read on designsafe using the computers that they provide. This would allow people to only download the data that they need

## Documentation integration
1) I think moving the database libary into it's own repo is probably for the best.
2) Need to write sphinx for the db library
3) Need to create a readthedocs for the db library

