import numpy                            as np
from math                               import cos, sin
from rohan.common.base                  import _RohanBase
from numpy.typing                       import NDArray
from typing                             import Union, TypeVar, List
from rohan.common.type_aliases          import Joints

SelfPanTiltModel = TypeVar("SelfPanTiltModel", bound="PanTiltModel" )
class PanTiltModel(_RohanBase):

    """
    The manipulator robotic model for a pan-tilt (PT) servo-ed system.
    :param ofs10_0: Offset from the 0 frame to the 1 frame respective to the 0
    :param ofs43_3: Offset from the 3 frame to the 4 frame respective to the 3
    """
    process_name : str = "pan-tilt manipulator"
    n_joints : int = 2
    ofs10_0  : float
    ofs43_3  : float
    
    def __init__(
        self,
        ofs10_0 : float = 0.,
        ofs43_3 : float = 0.,
        **config_kwargs
    ):
        self.load( 
            ofs10_0=ofs10_0, 
            ofs43_3=ofs43_3, 
            **config_kwargs
        )

    def _R12(
        self,
        pan_angle : float,
    ) -> NDArray:
        """
        Pan Direction Rotation matrix
        :param pan_angle: Angle of pan in radians
        """
        return np.array([
            [ cos(pan_angle) , -sin(pan_angle), 0 ],
            [ sin(pan_angle) ,  cos(pan_angle), 0 ],
            [ 0 , 0 , 1]
        ], dtype=np.float64 )

    def _R23(
        self,
        tilt_angle : float,
    ) -> NDArray:
        """
        Pan Direction Rotation matrix
        :param tilt_angle: Angle of tilt in radians
        """
        return np.array([
            [ cos(tilt_angle) , 0, -sin(tilt_angle) ],
            [ 0,  1 , 0 ],
            [ sin(tilt_angle) , 0,  cos(tilt_angle) ]
        ], dtype=np.float64 )
        
    def forward_kinematics( 
        self, 
        joint_angles : Union[ List[float], NDArray ],
    ) -> NDArray:
        """
        Returns End-Effector Location in Pan-Tilt System O-Frame
        :param joint_angles: Pan and tilt angles of PT system
        """
        pan_angle, tilt_angle = self.maybe_format_array(joint_angles)
        ofs40_0 = ( self._R12(pan_angle) @ ( self._R23(tilt_angle) @ self.ofs43_3 ) ) + self.ofs10_0 # Recall: R01 = Identity
        return ofs40_0

    def manipulator_jacobian(
        self,
        joint_angles : Union[ List[float], NDArray ],
    ) -> NDArray:
        """
        Returns Pan-Tilt System Manipulator Jacobian 
        :param joint_angles: Pan and tilt angles of PT system
        """
        pan_angle, tilt_angle = self.maybe_format_array(joint_angles)
        R12, R23              = self._R12(pan_angle), self._R23(tilt_angle)
        rel_fulcrum = ( R12 @ ( R23 @ self.ofs43_3 ) ) # Happens to be identical for both joints
        J = np.zeros( (6,self.n_joints), dtype=np.float64 )
        J[0:3,0] =  np.cross( 
                        np.array([[0],[0],[1]],dtype=np.float64),
                        rel_fulcrum
                    )   # [0:3,0] of Manipulator Jacobian
        J[0:3,1] =  np.cross( 
                        R12[:,1],
                        rel_fulcrum
                    )   # [0:3,1] of Manipulator Jacobian
        J[3:6,0] =  np.array([[0],[0],[1]],dtype=np.float64) 
        J[3:6,1] =  R12[:,1]
        return J
    
    def maybe_format_array(
        self,
        joint_angles : Joints
    ) -> List[ float ]:
        """
        Format List or Numpy array for functional consistency
        """
        angle_list = joint_angles
        if isinstance( joint_angles, NDArray ): 
            angle_list = angle_list.flatten().tolist()
        if not len(angle_list) == self.n_joints:
            raise TypeError( "Incorrect number of joints in provided array" ) 
        return angle_list