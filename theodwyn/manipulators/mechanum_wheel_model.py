import numpy as np
from rohan.common.base                  import _RohanBase
from numpy.typing                       import NDArray
from typing                             import TypeVar
SelfMechanum4Wheels = TypeVar("SelfMechanum4Wheels", bound="Mechanum4Wheels" )
class Mechanum4Wheels(_RohanBase):
    """
    The manipulator robotic model for a 4-mechanum-wheeled UGV system.
    :param lx: distance between the front two mechanum wheels
    :param ly: distance between the front and rear mechanum wheels
    """

    process_name: str = "4MechanumWheel"
    
    ly          : float
    lx          : float
    wheel_radius: float
    
    def __init__(
        self,
        lx           : float,
        ly           : float,
        wheel_radius : float,
        **config_kwargs
    ):
        self.load( 
            lx=lx, 
            ly=ly, 
            wheel_radius=wheel_radius,
            **config_kwargs
        )

    def get_invjacobian(
        self,
    ) -> NDArray:
        """
        Returns the Inverse-Jacobian for a point centered betwixt the 4MW
        """
        lsum    = (self.lx + self.ly)
        inv_wr  = 1/self.wheel_radius
        return inv_wr * np.array(
            [
                [ 1 , -1  ,  lsum ],
                [ 1 ,  1  ,  lsum ],
                [ 1 ,  1  , -lsum ],
                [ 1 , -1  , -lsum ]
            ]
        )