import numpy as np
from math import cos, sin

def Rot_1( angle : float ):
    return np.array([
        [ 1., 0.        ,  0.         ],
        [ 0., cos(angle),  sin(angle) ],
        [ 0.,-sin(angle),  cos(angle) ]
    ])

def Rot_2( angle : float ):
    return np.array([
        [ cos(angle), 0. , -sin(angle) ],
        [ 0.        , 1. , 0.         ],
        [ sin(angle), 0. , cos(angle) ]
    ])

def Rot_3( angle : float ):
    return np.array([
        [ cos(angle), sin(angle) , 0. ],
        [-sin(angle), cos(angle) , 0. ],
        [ 0.        , 0.         , 1. ]
    ])


def Rot_123( 
    roll    : float, 
    pitch   : float, 
    yaw     : float 
):
    return Rot_3(yaw) @ Rot_2(pitch) @ Rot_1(roll)