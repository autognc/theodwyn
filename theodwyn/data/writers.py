import csv
from io                         import TextIOWrapper
from typing                     import Optional, List, Any, Dict, Union

class CSVWriter:

    filename    : str
    fieldnames  : List[str]
    file        : Optional[TextIOWrapper]  = None
    csv_writer  : Optional[csv.DictWriter] = None 


    def __init__(
        self,
        filename   : str,
        fieldnames : List[str],
    ):
        self.filename   = filename
        self.fieldnames = fieldnames


    def __enter__( self ):
        self.open_file()
        return self


    def __exit__( self, exception_type, exception_value, traceback ):
        self.close_file()


    def open_file( self ) : 
        """
        Open file specified by name in initializer
        """
        if isinstance(self.file, TextIOWrapper):
            self.close_file()
        try:
            self.file       = open(self.filename,"w",newline='')
            self.csv_writer = csv.DictWriter( self.file, fieldnames=self.fieldnames )
            self.csv_writer.writeheader()
        except Exception as e:
            self.file = None


    def close_file( self ):
        """
        Close file specified by name in initializer if open
        """
        self.csv_writer = None
        if isinstance(self.file,TextIOWrapper):
            self.file.close()
        self.file = None


    def write_data( self, data : Union[List[Any],Dict[str,Any]] ):
        """
        Write data to csv file (must be organized wrt provided fieldnames)
        """
        if isinstance(data,Dict):
            self.csv_writer.writerow( data )
        else:    
            self.csv_writer.writerow(  dict( zip(self.fieldnames,data) ) ) 