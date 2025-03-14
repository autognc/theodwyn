from io                         import TextIOWrapper
from typing                     import Optional

class FileHandler:

    filename    : str
    file        : Optional[TextIOWrapper]  = None


    def __init__(
        self,
        filename   : str,
    ):
        self.filename   = filename


    def __enter__( self ):
        self.open_file()
        return self


    def __exit__( self, exception_type, exception_value, traceback ):
        self.close_file()


    def open_file( self , reading=True ) : 
        """
        Open file specified by name in initializer
        """
        if isinstance(self.file, TextIOWrapper):
            self.close_file()
        try:
            if reading:
                self.file       = open(self.filename)
            else:
                self.file       = open(self.filename,"w",newline='')
        except Exception as e:
            self.file = None


    def close_file( self ):
        """
        Close file specified by name in initializer if open
        """
        if isinstance(self.file,TextIOWrapper):
            self.file.close()
        self.file = None
