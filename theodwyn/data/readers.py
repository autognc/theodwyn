import csv
from theodwyn.data.file         import FileHandler
from io                         import TextIOWrapper
from typing                     import Optional

class CSVReader(FileHandler):

    csv_reader  : Optional[csv.DictReader] = None 

    def __init__(
        self,
        filename   : str,
    ):
        super().__init__(
            filename=filename
        )


    def open_file( self ) : 
        """
        Open file specified by name in initializer
        """
        super().open_file()
        if isinstance( self.file, TextIOWrapper ):
            self.csv_reader = csv.DictReader( self.file )


    def close_file( self ):
        """
        Close file specified by name in initializer if open
        """
        self.csv_reader = None
        super().close_file()

    def reset_iterator( self ):
        if isinstance( self.file, TextIOWrapper ): 
            self.file.seek(0)
            if self.csv_reader is None:
                self.csv_reader = csv.DictReader( self.file )
            _ = next(self.csv_reader)

    def read_nextrow( self ):
        """
        read next row of csv file (must be organized wrt provided fieldnames)
        """
        data = None
        if self.csv_reader:
            try:
                data = next( self.csv_reader )
            except StopIteration:
                data = None
        return data
