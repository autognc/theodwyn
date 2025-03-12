import csv
from theodwyn.data.file         import FileHandler
from io                         import TextIOWrapper
from typing                     import Optional, List, Any, Dict, Union

class CSVWriter(FileHandler):

    fieldnames  : List[str]
    csv_writer  : Optional[csv.DictWriter] = None 


    def __init__(
        self,
        filename   : str,
        fieldnames : List[str],
    ):
        super().__init__(
            filename=filename
        )
        self.fieldnames = fieldnames


    def open_file( self ) : 
        """
        Open file specified by name in initializer
        """
        super().open_file(reading=False)
        if isinstance( self.file, TextIOWrapper ):
            self.csv_writer = csv.DictWriter( self.file, fieldnames=self.fieldnames )
            self.csv_writer.writeheader()


    def close_file( self ):
        """
        Close file specified by name in initializer if open
        """
        self.csv_writer = None
        super().close_file()


    def write_data( self, data : Union[List[Any],Dict[str,Any]] ):
        """
        Write data to csv file (must be organized wrt provided fieldnames)
        """
        if self.csv_writer:
            if isinstance(data,Dict):
                self.csv_writer.writerow( data )
            else:    
                self.csv_writer.writerow(  dict( zip(self.fieldnames,data) ) ) 