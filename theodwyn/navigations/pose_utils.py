""" File contains functions and classes for working with poses in Python """

# imports
import numpy as np
import os, cv2, json
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any 
import itertools
import logging  
import pdb 

class setup_JAX:
    """ Custom class for setting up JAX """
    @staticmethod
    def set_jax_cpu():
        """ Force JAX to use cpu"""
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    @staticmethod
    def set_jax_backend():
        try:
            # check if CUDA is available
            devices     = jax.devices()
            has_cuda    = any(d.platform == 'gpu' for d in devices)
            if has_cuda:
                print('CUDA is available, testing memory allocation...')

                try:
                    # test memory allocation (small array to trigger OOM if memory is full)
                    jnp.ones((1024, 1024), dtype = jnp.float32).block_until_ready()
                    print('Using CUDA b/c GPU memory is available')
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print('Using CPU b/c CUDA Out of Memory error detected')
                        os.environ['JAX_PLATFORMS'] = 'cpu'
                    else:
                        raise  # re-raise the exception

            else:
                print('Using CPU b/c CUDA inoperable')
                os.environ['JAX_PLATFORMS'] = 'cpu'

        except Exception as e:
            print(f'Defaulting to CPU due to the following error: {e}')
            os.environ['JAX_PLATFORMS'] = 'cpu'
    
    @staticmethod
    def setup_safely():
        """ Safely set up JAX """
        try:
            import jax
            import jax.numpy as jnp
            setup_JAX.set_jax_cpu() # mainly for safety
            # setup_JAX.set_jax_backend()
            return jax, jnp
        except ImportError as e:
            cur_file    = os.path.abspath(__file__)
            print(f'Received following error attempting to import jax at {cur_file}: {e}')
            print(f'Falling back to NumPy: jax = dummy_JAX(), jnp = np')            
            # create dummy class for @jax.jit decorator
            class dummy_JAX:
                @staticmethod
                def jit(func):
                    return func
            import numpy as np
            return dummy_JAX, np


jax, jnp    = setup_JAX.setup_safely()

class custom_Quaternion:
    """
    A Python class that implements a right scalar first quaternion
    """

    def __init__(self, w = 1.0, x = 0.0, y = 0.0, z = 0.0):
        """ Initialize the quaternion, default is identity quaternion; will always be normalized """
        n           = norm( np.array([w, x, y, z]), ord = 2 )
        self.q_w    = w/n
        self.q_x    = x/n
        self.q_y    = y/n
        self.q_z    = z/n
        self.npy    = np.array([self.q_w, self.q_x, self.q_y, self.q_z])
        self.q_vec  = np.array([self.q_x, self.q_y, self.q_z]) 
 
    def vec(self):
        """ Return the vector part of the quaternion """
        return self.q_vec
    
    def w(self):
        """ Return the scalar part of the quaternion """
        return self.q_w
    
    def x(self):
        """ Return the x component of the quaternion """
        return self.q_x
    
    def y(self):
        """ Return the y component of the quaternion """
        return self.q_y
    
    def z(self):
        """ Return the z component of the quaternion """
        return self.q_z
    
    def inverse(self):
        """ Return the inverse of the quaternion """
        return custom_Quaternion(self.q_w, -self.q_x, -self.q_y, -self.q_z)
    
    def quat2gibbs_vec(self):
        """ Convert quaternion to Gibbs vector """
        return self.vec() / self.w()
    
    def __str_rep__(self):
        """ Return a string representation of the quaternion """
        return f'Quaterion (w = {self.q_w}, x = {self.q_x}, y = {self.q_y}, z = {self.q_z})'

class QuatMath:
    """
    A Python class that implements quaternion math operations, namely Shuster quaternion products, assumes right scalar first quaternions that are provided as (4,) numpy arrays
    """
    # deg2rad = np.pi / 180.0
    # rad2deg = 180.0 / np.pi

    # note a static method does not have access to the class or instance
    @staticmethod
    def sscp_R3(v):
        """ Compute skew-symmetric cross product matrix for R^3 vectors """
        m_out   = np.array([
                            [0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                            ])
        return m_out
    
    @staticmethod
    def qmult_shu(q2, q1):
        """ Compute the Shuster quaternion product of two quaternions that are right scalar first """
        q1v         = q1[1:4]
        q1s         = q1[0]
        q2v         = q2[1:4]
        q2s         = q2[0]
        q2v_sscp    = QuatMath.sscp_R3(q2v)
        q_out       = np.append(
                                    [q1s*q2s - np.dot(q2v, q1v)],
                                    [q1s*q2v + q2s*q1v - q2v_sscp@q1v]
                                )
        q_out       = QuatMath.q_norm(q_out)
        return q_out
    
    @staticmethod
    def qmulut_ham(q2, q1):
        """ Compute the Hamilton quaternion product of two quaternions that are right scalar first """
        
        q1v         = q1[1:4]
        q1s         = q1[0]
        q2v         = q2[1:4]
        q2s         = q2[0]
        q2v_sscp    = QuatMath.sscp_R3(q2v)
        
        q_out       = np.append(
                                    [q1s*q2s - np.dot(q2v, q1v)],
                                    [q1s*q2v + q2s*q1v + q2v_sscp@q1v]
                                )
        q_out       = QuatMath.q_norm(q_out)

        return q_out
    
    @staticmethod
    def q_norm(q):
        """ Normalize a quaternion (4-vector form [w, x, y, z]) """
        n   = np.linalg.norm(q)
        if n < 1e-12:
            # avoid divide-by-zero; return identity
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / n
    
    @staticmethod
    def q_conj(q):
        """ Compute the conjugate of a quaternion """
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def q_error_shu(q2, q1):
        """ Compute the quaternion error between two quaternions """
        q_err   = QuatMath.qmult_shu(q2, QuatMath.q_conj(q1))
        q_err   = QuatMath.q_norm(q_err)
        return q_err
    
    @staticmethod
    def q_error_ham(q2, q1):
        """ Compute the quaternion error between two quaternions """
        q_err   = QuatMath.qmulut_ham(q2, QuatMath.q_conj(q1))
        q_err   = QuatMath.q_norm(q_err)
        return q_err

    @staticmethod
    def q2trfm(q): 
        """ Convert a right scalar first quaternion to a transformation matrix, which is transpose of a rotation matrix """ 
        q       = QuatMath.q_norm(q)
        qv      = q[1:4]
        qs      = q[0]
        qv_sscp = QuatMath.sscp_R3(qv)
        Trfm    = (np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp).T
        return Trfm
    
    @staticmethod
    def q2rotm(q):
        """ Converts a quaternion to a rotation matrix (active rotation), not a transformation matrix """
        q       = QuatMath.q_norm(q)
        qv      = q[1:4]
        qs      = q[0]
        qv_sscp = QuatMath.sscp_R3(qv)
        Rm      = np.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp
        return Rm
    
    @staticmethod
    def jax_q_conj(q):
        """ Compute the conjugate of a quaternion using JAX """
        return jnp.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def jax_sscp_R3(v):
        """ Compute skew-symmetric cross product matrix for R^3 vectors using JAX"""
        m_out   = jnp.array([
                            [0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                            ])
        return m_out
    
    @staticmethod
    def jax_q2trfm(q): 
        """ Convert a right scalar first quaternion to a transformation matrix, which is transpose of a rotation matrix using JAX """ 
        q       = QuatMath.jax_q_norm(q)
        qv      = q[1:4]
        qs      = q[0]
        qv_sscp = QuatMath.jax_sscp_R3(qv)
        # Trfm    = (jnp.eye(3) + 2*qs*qv_sscp + 2*qv_sscp@qv_sscp).T
        Trfm    = jnp.eye(3) - 2*qs*qv_sscp + 2*qv_sscp@qv_sscp
        return Trfm
    
    @staticmethod
    def jax_q_norm(q):
        """ Normalize a quaternion (4-vector form [w, x, y, z]) """
        n   = jnp.linalg.norm(q)        
        return jnp.where(n < 1e-12, jnp.array([1.0, 0.0, 0.0, 0.0]), q / n)
    
    @staticmethod
    def jax_qmulut_ham(q2, q1):
        """ Compute the Hamilton quaternion product of two quaternions that are right scalar first """
        q1v         = q1[1:4]
        q1s         = q1[0]
        q2v         = q2[1:4]
        q2s         = q2[0]
        q2v_sscp    = QuatMath.jax_sscp_R3(q2v)
        scalar      = q1s*q2s - jnp.dot(q2v, q1v)
        vector      = q1s*q2v + q2s*q1v + q2v_sscp@q1v
        q_out       = jnp.concatenate( [scalar[None], vector], axis = 0)
        q_out       = QuatMath.jax_q_norm(q_out)       
        return q_out
    
    @staticmethod
    def safe_arctan2(x, y, eps = 1e-12):
        """ Safe arctan2 function that avoids division by zero """
        y_safe  = jnp.where(jnp.abs(y) < eps, eps * jnp.sign(y), y)
        return jnp.arctan2(x, y_safe)

        
    @staticmethod
    def jax_unnormalized_sinc(x):
        """ Compute the unnormalized sinc function """
        return jnp.where( jnp.abs(x) < 1e-12, 1.0, jnp.sin(x) / x)

    @staticmethod
    def jax_qexp_map(v):
        """ 
        Creates a quaternion from a vector v in R^3 based on the exponential map
        based on  Practical Parameterization of Rotations Using the Exponential Map(Grassia, 1998)
        """
        v       = jnp.array(v)
        theta   = jnp.linalg.norm(v)
        scalar  = jnp.cos(theta/2)
        vector  = (1/2) * QuatMath.jax_unnormalized_sinc(theta/2) * v 
        quat    = jnp.append(scalar, vector)
        quat    = QuatMath.jax_q_norm(quat) # precaution, should not be required b/c of exp map
        # returns identity quaternion if v is zero
        return quat
    
    @staticmethod
    def robust_jax_qexp_map(v, eps = 1e-8, max_theta_margin = 1e-3):
        """ Computes a quaternoion from a vector using the exponential map with modifications to handle small angles and angles approaching 180 """
        v               = jnp.array(v)
        theta           = jnp.linalg.norm(v)
        theta_adj       = jnp.where(jnp.abs(theta - jnp.pi) < max_theta_margin, jnp.pi - max_theta_margin, theta)  
        half_theta_adj  = theta_adj / 2
        
        # Taylor series: sin(half_theta)/(theta) ~ 1/2 - (1/48) * (half_theta)^2
        sin_half_theta_over_theta   = jnp.where(jnp.abs(theta_adj) < eps, 0.5 - (theta_adj**2) / 48.0, jnp.sin(half_theta_adj) / theta_adj)
        
        quat            = jnp.append(jnp.cos(half_theta_adj), sin_half_theta_over_theta * v)
        return quat
        
    @staticmethod
    def jax_project_to_tangent(quat, grad_q):
        """ 
        Project the gradient grad_q (4-vector) onto the tangent space of quaternion q
        For a unit quaternion, the tangent space consists of all 4-vectors orthogonal to q 
        """
        return grad_q - jnp.dot(grad_q, quat) * quat
    
    @staticmethod
    def rmat_to_euler_angles(R, sequence = 'xyz', intrinsic = True, thr = 1e-6, return_seq = False):
        """
        Converts a 3x3 rotation matrix to Euler angles in a given Tait-Bryan sequence in degrees
        These sequences are: 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'
        
        xyz: first rotate about x, then y, then z
        xzy: first rotate about x, then z, then y
        yxz: first rotate about y, then x, then z
        yzx: first rotate about y, then z, then x
        zxy: first rotate about z, then x, then y
        zyx: first rotate about z, then y, then x
        
        Handles both intrinsic and extrinsic rotations
        Intrinsic rotations are rotations about the axes of a body-fixed frame 
        Extrinsic rotations are rotations about the axes of a global frame 
        For pose estimation, we typically use intrinsic rotations 

        Source: https://en.wikipedia.org/wiki/Euler_angles
        See Tait-Bryan angles table

        Inputs: 
            R (np.ndarray): 3x3 rotation matrix
            sequence (str): Tait-Bryan sequence of rotations
            intrinsic (bool): True for intrinsic rotations, False for extrinsic rotations
            thr (float): threshold for numerical stability
            return_seq (bool): whether to return the sequence of rotations applied

        Outputs:
            angles (np.ndarray): Euler angles in degrees
            apply_seq (str): Sequence of rotations applied
        """

        sequence    = sequence.lower()  # Ensure lowercase input
        prop_str    = 'Intrinsic' if intrinsic else 'Extrinsic'
        if not intrinsic:
            sequence    = sequence[::-1]  # reverse the order for extrinsic rotations


        half_pi     = np.pi / 2
        if sequence == 'xyz':
            # alpha = arctan(-R_{23} / R_{33})
            # beta = arcsin(R_{13})
            # gamma = arctan(-R_{12} / R_{11})
            a2  = np.arcsin(R[0, 2]) 
            # check for gimbal lock 
            if np.isclose(a2, half_pi, atol = thr):
                # gimbal lock at 90 degrees
                a1  = np.arctan2(R[1, 0], R[1, 1])
                a3  = 0
            elif np.isclose(a2, -half_pi, atol = thr):
                # gimbal lock at -90 degrees
                a1  = np.arctan2(-R[1, 0], R[1, 1])
                a3  = 0
            else:
                a1  = np.arctan2(-R[1, 2], R[2, 2])
                a3  = np.arctan2(-R[0, 1], R[0, 0])
            apply_seq   = f'Rotate about X, then Y, then Z ({prop_str})'

        elif sequence == 'xzy':
            # alpha = arctan(R_{32} / R_{22})
            # beta = -arcsin(R_{12})
            # gamma = arctan(R_{13} / R_{11})
            a2  = np.arcsin(-R[0, 1])
            if np.isclose(a2, half_pi, atol = thr):
                a1  = np.arctan2(R[2, 0], R[2, 2])
                a3  = 0
            elif np.isclose(a2, -half_pi, atol = thr):
                a1  = np.arctan2(-R[2, 0], R[2, 2])
                a3  = 0
            else:
                a1  = np.arctan2(R[2, 1], R[1, 1])
                a3  = np.arctan2(R[0, 2], R[0, 0])
            apply_seq   = f'Rotate about X, then Z, then Y ({prop_str})'
        
        elif sequence == 'yxz':
            # alpha = arctan(R_{13} / R_{33})
            # beta = arcsin(-R_{23})
            # gamma = arctan(R_{21} / R_{22})
            a2  = np.arcsin(-R[1, 2])
            if np.isclose(a2, half_pi, atol = thr):
                a1  = np.arctan2(R[0, 1], R[0, 0])
                a3  = 0
            elif np.isclose(a2, -half_pi, atol = thr):
                a1  = np.arctan2(-R[0, 1], R[0, 0])
                a3  = 0
            else:
                a1  = np.arctan2(R[0, 2], R[2, 2])
                a3  = np.arctan2(R[1, 0], R[1, 1])
            apply_seq   = f'Rotate about Y, then X, then Z ({prop_str})'

        elif sequence == 'yzx':
            # alpha = arctan(-R_{31} / R_{11})
            # beta = arcsin(R_{21})
            # gamma = arctan(-R_{23} / R_{22})
            a2  = np.arcsin(R[1, 0])
            if  np.isclose(a2, half_pi, atol = thr):
                a1  = np.arctan2(R[2, 1], R[2, 2])
                a3  = 0 
            elif np.isclose(a2, -half_pi, atol = thr):
                a1  = np.arctan2(-R[2, 1], R[2, 2])
                a3  = 0
            else:
                a1  = np.arctan2(-R[2, 0], R[0, 0])
                a3  = np.arctan2(-R[1, 2], R[1, 1])
            apply_seq   = f'Rotate about Y, then Z, then X ({prop_str})'

        elif sequence == 'zxy':
            # alpha = arctan( -R_{12} / R_{22})
            # beta = arcsin(R_{32})
            # gamma = arctan(-R_{31} / R_{33})
            a2  = np.arcsin(R[2, 1])
            if np.isclose(a2, half_pi, atol = thr):
                a1  = np.arctan2(R[0, 2], R[0, 0])
                a3  = 0
            elif np.isclose(a2, -half_pi, atol = thr):
                a1  = np.arctan2(-R[0, 2], R[0, 0])
                a3  = 0
            else:
                a1  = np.arctan2(-R[0, 1], R[1, 1])
                a3  = np.arctan2(-R[2, 0], R[2, 2])
            apply_seq   = f'Rotate about Z, then X, then Y ({prop_str})'
        
        elif sequence == 'zyx':
            # alpha = arctan(R_{21} / R_{11})
            # beta = arcsin(-R_{31})
            # gamma = arctan(R_{32} / R_{33})
            a2  = np.arcsin(-R[2, 0])
            if np.isclose(a2, half_pi, atol = thr):
                a1  = np.arctan2(R[0, 1], R[1, 1])
                a3  = 0
            elif np.isclose(a2, -half_pi, atol = thr):
                a1  = np.arctan2(-R[0, 1], R[1, 1])
                a3  = 0
            else:
                a1  = np.arctan2(R[1, 0], R[0, 0])
                a3  = np.arctan2(R[2, 1], R[2, 2])
            apply_seq   = f'Rotate about Z, then Y, then X ({prop_str})'

        else:
            raise ValueError("Invalid sequence. Choose from 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', or 'zyx'")

        if return_seq:
            return np.degrees([a1, a2, a3]), apply_seq
        else:
            return np.degrees([a1, a2, a3])
        
    @staticmethod
    def quat2gibbs(q):
        """ Converts a Right Scalar First quaternion to Gibbs vector """
        qv      = q[1:4]
        qs      = q[0]
        gv      = qv / qs
        return gv
        
    @staticmethod
    def gibbs2quat(gv):
        gnorm_sq    = np.linalg.norm(gv, ord = 2)**2
        qs          = 1 / np.sqrt(1 + gnorm_sq)
        qv          = qs * gv
        q           = np.append(qs, qv)
        return q

    @staticmethod
    def quat2rotangle(q):
        """ Compute the rotation angle (radians) from a quaternion using euler axis and angle representation
        # q = [q_s, q_v] = [cos(theta/2), sin(theta/2) * axis]
        """
        return 2 * np.arccos(QuatMath.q_norm(q)[0])
    
    @staticmethod
    def Rot_x( angle : float ):
        """ Compute cannonical rotation matrix for rotation about x-axis from an angle in radians """
        c, s    = np.cos(angle), np.sin(angle)
        R       = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return R
    
    @staticmethod 
    def Rot_y( angle : float ):
        """ Compute cannonical rotation matrix for rotation about y-axis from an angle in radians """
        c, s    = np.cos(angle), np.sin(angle)
        R       = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return R
    
    @staticmethod
    def Rot_z( angle : float ):
        """ Compute cannonical rotation matrix for rotation about z-axis from an angle in radians """
        c, s    = np.cos(angle), np.sin(angle)
        R       = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return R
    
    @staticmethod 
    def Rot_xyz(
                    roll : float,
                    pitch : float,
                    yaw : float
                ):
        """ Compute the rotation matrix from the Euler angles in radians for a XYZ sequence """
        R_x     = QuatMath.Rot_x(roll)
        R_y     = QuatMath.Rot_y(pitch)
        R_z     = QuatMath.Rot_z(yaw)
        R_xyz   = R_z @ R_y @ R_x
        return R_xyz

    @staticmethod
    def rotm_to_quat(Rotm):
        """ Convert a 3x3 rotation matrix to a quaternion """        
        trace       = np.trace(Rotm)
        if trace > 0:
            S   = np.sqrt(trace + 1.0) * 2  # S = 4*w
            w   = 0.25 * S
            x   = (Rotm[2, 1] - Rotm[1, 2]) / S
            y   = (Rotm[0, 2] - Rotm[2, 0]) / S
            z   = (Rotm[1, 0] - Rotm[0, 1]) / S
        elif (Rotm[0, 0] > Rotm[1, 1]) and (Rotm[0, 0] > Rotm[2, 2]):
            S   = np.sqrt(1.0 + Rotm[0, 0] - Rotm[1, 1] - Rotm[2, 2]) * 2  # S = 4*x
            w   = (Rotm[2, 1] - Rotm[1, 2]) / S
            x   = 0.25 * S
            y   = (Rotm[0, 1] + Rotm[1, 0]) / S
            z   = (Rotm[0, 2] + Rotm[2, 0]) / S
        elif Rotm[1, 1] > Rotm[2, 2]:
            S   = np.sqrt(1.0 + Rotm[1, 1] - Rotm[0, 0] - Rotm[2, 2]) * 2  # S = 4*y
            w   = (Rotm[0, 2] - Rotm[2, 0]) / S
            x   = (Rotm[0, 1] + Rotm[1, 0]) / S
            y   = 0.25 * S
            z   = (Rotm[1, 2] + Rotm[2, 1]) / S
        else:
            S   = np.sqrt(1.0 + Rotm[2, 2] - Rotm[0, 0] - Rotm[1, 1]) * 2  # S = 4*z
            w   = (Rotm[1, 0] - Rotm[0, 1]) / S
            x   = (Rotm[0, 2] + Rotm[2, 0]) / S
            y   = (Rotm[1, 2] + Rotm[2, 1]) / S
            z   = 0.25 * S
        quat    = np.array([w, x, y, z])
        return quat


    @staticmethod
    def jax_quat_to_exp_map(quat, eps = 1e-12):
        """
        Converts a right scalar first quaternion to its exponential map (rotation vector) representation

        Applies the euler axis and angle representation to the quaternion:
            - The rotation angle: theta = 2 * arccos(w)
            - The rotation axis: axis = [x, y, z] / sin(theta/2)
        The exponential map vector is then given by:
            rotation_vector = theta * axis

        For very small rotation angles (i.e. sin(theta/2) close to zero), a first order approximation
        is used to avoid numerical instability.

        Inputs:
            quat (np.array): (4,) array representing the quaternion in right scalar first format
            eps (float): small value to avoid division by zero
        
        Outputs:
            exp_map (np.array): (3,) array representing the exponential map vector

        """
        # ensure the quaternion is normalized
        quat        = quat / jnp.linalg.norm(quat)
        w, x, y, z  = quat
        # clamp w to the valid range for arccos to avoid numerical issues
        w           = jnp.clip(w, -1.0, 1.0)
        theta       = 2.0 * jnp.arccos(w)
        
        # compute sin(theta/2) (should be positive for 0 <= theta <= pi)
        half_sin_theta  = jnp.sqrt(1.0 - w * w)
        
        # when the rotation is very small, use the first order approximation:
        # sin(theta/2) ~ theta/2, so rotation_vector ~ 2*[x, y, z]
        def small_angle():
            return 2.0 * jnp.array([x, y, z])
        
        def regular_case():
            axis = jnp.array([x, y, z]) / half_sin_theta
            return theta * axis

        exp_map = jnp.where(half_sin_theta < eps, small_angle(), regular_case())
        return exp_map
    
    @staticmethod
    def jax_random_quat(key = None):
        """ Generate a random quaternion using JAX, fall back to NumPy if JAX is not available """
        try:
            import jax
            import jax.numpy as jnp
            if key is None:
                key = jax.random.PRNGKey(0)
                # fixes to avoid reproducibility issues
            quat    = jax.random.normal(key, (4,))
            quat    = QuatMath.jax_q_norm(quat)
        except ImportError as e:
            print(f'Error: {e} --> Falling back to NumPy for random quaternion generation')
            import numpy as np
            seed    = 0
            if key is not None:
                try:
                    seed = int( key[-1]) # expected PRNG key format
                except Exception:
                    pass  # use default seed 0 if conversion fails
            np.random.seed(seed)
            quat   = np.random.normal(loc = 0, scale = 1, size = (4,))
            quat   = QuatMath.q_norm(quat)
    
        return quat
    
    @staticmethod
    def covar_block_switch(covar):
        """ Switch the blocks of a covariance matrix """
        twoN        = covar.shape[0]
        n           = twoN // 2
        I_n         = np.eye(n)
        O_n         = np.zeros((n, n))
        Q           = np.block([
                                [O_n, I_n], 
                                [I_n, O_n]
                    ])
        upd_covar   = Q @ covar @ Q
        return upd_covar

# # Testing
# q1  = custom_Quaternion(1, 2, 3, 4)
# q2  = custom_Quaternion(5, 6, 7, 8)
# vec = q1.vec()
# z   = QuatMath.sscp_R3(vec)
# g   = QuatMath.qmult_shu(q2.npy, q1.npy)
# trf = QuatMath.q2trfm(q1.npy)
# print(f'Vector part of q1: {vec} with shape {vec.shape}')
# print(f'Skew-symmetric cross product matrix for q1: {z} with shape {z.shape}')
# print(f'custom_Quaternion product of q1 and q2: {g} with shape {g.shape}')
# print(f'Transformation matrix from q1: {trf} with shape {trf.shape}')
# v   = np.array([0, 0, 0])
# q   = QuatMath.jax_qexp_map(v)
# print(f'custom_Quaternion from vector using exponential map with JAX: {q} with shape {q.shape}')
# q3  = QuatMath.jax_qmulut_ham(jnp.array(q2.npy), jnp.array(q1.npy))
# print(f'Quaternion product of q1 and q2 using Hamilton product with JAX: {q3} with shape {q3.shape}')
# gv1 = QuatMath.quat2gibbs(q1.npy)
# print(f'Gibbs vector from quaternion q1: {gv1} with shape {gv1.shape}')
# q4  = QuatMath.gibbs2quat(gv1)
# print(f'Quaternion from Gibbs vector: {q4} with shape {q4.shape}')
# qe  = QuatMath.q_error_shu(q4, q1.npy)
# print(f'Quaternion error between q4 and q1: {qe} with shape {qe.shape}')
# q5  = QuatMath.robust_jax_qexp_map(np.array([0, 0, 0]))
# print(f'Quaternion from vector using robust exponential map with JAX: {q5} with shape {q5.shape}')
# angle   = np.radians(45)
# Rotm    = np.array([
#                     [ np.cos(angle), -np.sin(angle), 0.0 ],
#                     [ np.sin(angle),  np.cos(angle), 0.0 ],
#                     [ 0.0,             0.0,             1.0 ]
#     ])
# q6      = QuatMath.rotm_to_quat(Rotm)
# print(f'Quaternion from rotation matrix R: {q6} with shape {q6.shape}')
# # scipy sanity check
# temp    = R.from_matrix(Rotm)
# # try:
# #     q7     = temp.as_quat(scalar_first = True)
# # except:
# #     q7      = temp.as_quat()
# #     q7      = np.array([q7[3], q7[0], q7[1], q7[2]])
# q7      = temp.as_quat() # scalar last is default
# q7      = np.concatenate([q7[-1:], q7[:-1]])
# print(f'Quaternion from rotation matrix R using scipy: {q7} with shape {q7.shape}')
# print(f'Difference between q6 and q7: {np.linalg.norm(q6 - q7)}')
# random_quat_test    = QuatMath.jax_random_quat()
# print(f'Random quaternion generated using JAX: {random_quat_test} with shape {random_quat_test.shape}')
# print(f'Random quaterion norm: ', np.linalg.norm(random_quat_test))
# pdb.set_trace()    

class TorchQuatMath:
    """ Class that implements quaternion math operations using PyTorch, setup for batch processing, assumes right scalar first quaternions """
    
    @staticmethod
    def torch_q_norm(q, eps = 1e-8):
        """ Normalize a quaterion tensor that is (4,) or (N, 4) """
        import torch 
        norm    = q.norm(dim = -1, keepdim = True)
        return q / (norm + eps)

    @staticmethod
    def torch_q_conj(q):
        """ Compute the conjugate of a quaternion tensor that is (4,) or (N, 4) """
        import torch
        q       = TorchQuatMath.torch_q_norm(q) 
        sign    = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype = q.dtype, device = q.device)
        return q * sign

    @staticmethod
    def torch_sscp_R3(v):
        """ Compute skew-symmetric cross product matrix for R^3 vectors using PyTorch --> output is (3, 3) or (N, 3, 3) """
        import torch
        v_flat      = v.reshape(-1, 3)
        m_          = torch.zeros((v_flat.size(0), 3, 3), dtype = v.dtype, device = v.device)
        m_[:, 0, 1] = -v_flat[:, 2]
        m_[:, 0, 2] = v_flat[:, 1]
        m_[:, 1, 0] = v_flat[:, 2]
        m_[:, 1, 2] = -v_flat[:, 0]
        m_[:, 2, 0] = -v_flat[:, 1]
        m_[:, 2, 1] = v_flat[:, 0]
        m_final     = m_.reshape(v.shape[:-1] + (3, 3))
        return m_final
    
    @staticmethod
    def torch_q2trfm(q):
        """ Convert a single or batch of right scalar first quaternions to a transformation matrix using PyTorch """
        import torch
        q       = TorchQuatMath.torch_q_norm(q)
        qv      = q[..., 1:4] # shape (N, 3)
        qs      = q[..., 0] # shape (N,)
        qv_sscp = TorchQuatMath.torch_sscp_R3(qv) # shape (N, 3, 3)
        # expand the identity matrix to match the batch size
        I       = torch.eye(3, dtype = q.dtype, device = q.device).expand(q.shape[:-1] + (3, 3))
        Trfm    = I - 2*qs.unsqueeze(-1).unsqueeze(-1) * qv_sscp + 2 * torch.matmul(qv_sscp, qv_sscp)
        # Trfm    = I - 2*qs.unsqueeze(-1).unsqueeze(-1) * qv_sscp + 2 * torch.bmm(qv_sscp, qv_sscp)
        return Trfm

    @staticmethod
    def torch_unnormalized_sinc(x, eps = 1e-12):
        """ Compute the unnormalized sinc function using PyTorch """
        import torch
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(x) / x)
    
    @staticmethod
    def torch_qexp_map(v):
        """ Creates a quaternion from a single or batch of vectors v in R^3 based on the exponential map using PyTorch"""
        import torch
        original_shape  = v.shape[:-1] # means all dimensions except the last one
        v_flat          = v.reshape(-1, 3) # -1 means infer the size from the other dimensions
        theta           = v_flat.norm(dim = -1, keepdim = True)
        scalar          = torch.cos(theta/2)
        vector          = (1/2) * TorchQuatMath.torch_unnormalized_sinc(theta/2) * v_flat
        quat_flat       = TorchQuatMath.torch_q_norm( torch.cat([scalar, vector], dim = -1) )
        quat            = quat_flat.reshape(original_shape + (4,))
        return quat    
    
    @staticmethod
    def torch_qmult_ham(q2, q1):
        """ Compute the Hamilton quaternion product of two quaternions that are right scalar first using PyTorch """
        import torch
        q2              = TorchQuatMath.torch_q_norm(q2)
        q1              = TorchQuatMath.torch_q_norm(q1)
        original_shape  = q1.shape[:-1] 
        q1_flat         = q1.reshape(-1, 4)
        q2_flat         = q2.reshape(-1, 4)

        q1s             = q1_flat[:, 0] # shape (N,)
        q1v             = q1_flat[:, 1:4] # shape (N, 3)
        q2s             = q2_flat[:, 0] # shape (N,) 
        q2v             = q2_flat[:, 1:4] # shape (N, 3)
        q2v_sscp        = TorchQuatMath.torch_sscp_R3(q2v) # shape (N, 3, 3)
        scalar          = q1s*q2s - torch.sum(q2v * q1v, dim = -1) # shape (N,)
        vector          = q1s.unsqueeze(-1) * q2v + q2s.unsqueeze(-1) * q1v + \
                        torch.matmul(q2v_sscp, q1v.unsqueeze(-1)).squeeze(-1) # shape (N, 3)
        q_out_flat      = torch.cat([scalar.unsqueeze(-1), vector], dim = -1) # shape (N, 4)
        q_out_flat      = TorchQuatMath.torch_q_norm(q_out_flat)
        q_out           = q_out_flat.reshape(original_shape + (4,))
        return q_out

# # Testing for TorchQuatMath
# import torch 
# q_single    = torch.tensor([1.0, 2.0, 3.0, 4.0])
# q2_single   = torch.tensor([5.0, 6.0, 7.0, 8.0])
# q_batch     = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
# q1_batch    = torch.tensor([[1.0, 2.0, 3.0, 4.0],
#                          [5.0, 6.0, 7.0, 8.0]])
# q2_batch    = torch.tensor([[9.0, 10.0, 11.0, 12.0],
#                          [13.0, 14.0, 15.0, 16.0]])
# test1       = TorchQuatMath.torch_q2trfm(q_single)
# test2       = TorchQuatMath.torch_q2trfm(q_batch)
# v_single    = torch.tensor([0.1, 0.2, 0.3])
# quat_s      = TorchQuatMath.torch_qexp_map(v_single)
# v_batch     = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
# quat_b      = TorchQuatMath.torch_qexp_map(v_batch)
# qmult_sing  = TorchQuatMath.torch_qmult_ham(q2_single, q_single)
# qmult_batch = TorchQuatMath.torch_qmult_ham(q2_batch, q1_batch)
# jax_test    = QuatMath.jax_q2trfm(q_batch[1].cpu().numpy())
# sc1         = test2[1].cpu().numpy() - jax_test
# sc2         = qmult_batch[1].cpu().numpy() - QuatMath.jax_qmulut_ham(q2_batch[1].cpu().numpy(), q1_batch[1].cpu().numpy())
# print(f'Quaternion to transformation matrix for single quaternion: {test1} with shape {test1.shape}')
# print(f'Quaternion to transformation matrix for batch of quaternions: {test2} with shape {test2.shape}')
# print(f'Quaternion from vector using exponential map for single vector: {quat_s} with shape {quat_s.shape}')
# print(f'Quaternion from vector using exponential map for batch of vectors: {quat_b} with shape {quat_b.shape}')
# print(f'Hamilton product of two quaternions for single quaternions: {qmult_sing} with shape {qmult_sing.shape}')
# print(f'Hamilton product of two quaternions for batch of quaternions: {qmult_batch} with shape {qmult_batch.shape}')
# print(f'JAX quaternion to transformation matrix for batch of quaternions: {jax_test} with shape {jax_test.shape}')
# print(f'Difference between PyTorch and JAX quaternion to transformation matrix: {sc1}')
# print(f'Difference between PyTorch and JAX quaternion Hamilton product: {sc2}')
# pdb.set_trace()



class Bearing:
    """
    A Python class that includes methods for going from pixel coordinates to bearing measurements
    """

    @staticmethod
    def compute_azimuth_elevation(kps_2D, K):
        """ 
        Compute azimuth and elevation angles from pixel coordinates (assumes kps_2d is in (x,y) format)

        Inputs: 
            kps_2D (np.array): (N, 2) array representing the 2D keypoints in the image plane
            K (np.array): 3x3 camera matrix

        Outputs:
                az_el (np.array): (N, 2) array representing the azimuth and elevation angles
                azel (np.array): (2*N,) array representing alternating azimuth and elevation angles
                    [az_1, el_1, az_2, el_2, ..., az_N, el_N]
        """
        c_x             = K[0, 2]
        c_y             = K[1, 2]
        f_x             = K[0, 0]
        f_y             = K[1, 1]
        
        kps_x   = kps_2D[:,0]
        kps_y   = kps_2D[:,1]
        x_n     = (kps_x - c_x) / f_x
        y_n     = (kps_y - c_y) / f_y
        # y_n     = (c_y - kps_y) / f_y, another way, need to test original bindings in KeypointRCNN_trainer/keypoint_rcnn  /validate_pnp_nls.py
        one_t   = np.array(1.0, dtype = np.float32)
        az      = np.arctan2(x_n, one_t)
        el      = np.arctan2(y_n, one_t)
        az_el   = np.stack( (az, el), axis = 1 )
        azel    = az_el.ravel()
        return az_el, azel
    
    @staticmethod
    def torch_compute_azimuth_elevation(kps_2D, K):
        """ 
        With pytorch, compute azimuth and elevation angles from pixel coordinates (assumes kps_2d is in (x,y) format)

        Inputs: 
        kps_2D (torch.Tensor): (N, 2) tensor representing the 2D keypoints in the image plane
        K (torch.Tensor): 3x3 camera matrix

        Outputs:
        az_el (torch.Tensor): (N, 2) tensor representing the azimuth and elevation angles
        azel (torch.Tensor): (2*N,) tensor representing alternating azimuth and elevation angles
            [az_1, el_1, az_2, el_2, ..., az_N, el_N]
        """
        import torch 
        c_x             = K[0, 2]
        c_y             = K[1, 2]
        f_x             = K[0, 0]
        f_y             = K[1, 1]
        
        kps_x   = kps_2D[:,0]
        kps_y   = kps_2D[:,1]
        x_n     = (kps_x - c_x) / f_x
        y_n     = (kps_y - c_y) / f_y
        # y_n     = (c_y - kps_y) / f_y, another way, need to test original bindings in KeypointRCNN_trainer/keypoint_rcnn  /validate_pnp_nls.py
        one_t   = torch.tensor(1.0, dtype = kps_2D.dtype, device = kps_2D.device)
        az      = torch.atan2(x_n, one_t)
        el      = torch.atan2(y_n, one_t)
        az_el   = torch.stack( (az, el), dim = 1 )
        azel    = az_el.view(-1)
        return az_el, azel
    
    @staticmethod
    def torch_batch_compute_azimuth_elevation(kps_2D, K):
        """
        With pytorch, compute azimuth and elevation angles from pixel coordinates (assumes kps_2d is in (x,y) format), works with batch processing

        Inputs:
        kps_2D (torch.Tensor): (B, N, 2) tensor representing the 2D keypoints in the image plane
        K (torch.Tensor): (B, 3, 3) tensor representing the camera matrix

        Outputs:
        az_el (torch.Tensor): (B, N, 2) tensor representing the azimuth and elevation angles
        azel (torch.Tensor): (B, 2*N) tensor representing alternating azimuth and elevation angles
            [az_1, el_1, az_2, el_2, ..., az_N, el_N]
        """
        import torch 
        c_x = K[:, 0, 2] # shape (B,)
        c_y = K[:, 1, 2] # shape (B,)
        f_x = K[:, 0, 0] # shape (B,)
        f_y = K[:, 1, 1] # shape (B,)
        
        kps_x   = kps_2D[:, :, 0] # shape (B, N)
        kps_y   = kps_2D[:, :, 1] # shape (B, N)
        x_n     = (kps_x - c_x.unsqueeze(1)) / f_x.unsqueeze(1) # shape (B, N)
        y_n     = (kps_y - c_y.unsqueeze(1)) / f_y.unsqueeze(1) # shape (B, N)
        one_t   = torch.tensor(1.0, dtype = kps_2D.dtype, device = kps_2D.device)
        az      = torch.atan2(x_n, one_t) # shape (B, N)
        el      = torch.atan2(y_n, one_t) # shape (B, N)
        az_el   = torch.stack( (az, el), dim = -1 ) # shape (B, N, 2)
        azel    = az_el.view(az_el.shape[0], -1) # shape (B, 2*N)
        return az_el, azel
    
    @staticmethod 
    def torch_batch_compute_kps_2D_pix(kps_2D_norm, K):
        """
        With PyTorch, compute the unnormalized pixel coordinates from normalized pixel coordinates
        This function assumes a camera matrix with principal point at the center of the image

        Inputs:
        kps_2D_norm (torch.Tensor): (B, N, 2) tensor representing the normalized 2D keypoints in the image plane
        K (torch.Tensor): (B, 3, 3) tensor representing the camera matrix
        """
        import torch 
        img_w       = 2 * K[:, 0, 2] # shape (B,)
        img_h       = 2 * K[:, 1, 2] # shape (B,)
        kps_2D_pix  = kps_2D_norm.clone()
        kps_2D_pix[..., 0] = kps_2D_norm[..., 0] * img_w.unsqueeze(1)
        kps_2D_pix[..., 1] = kps_2D_norm[..., 1] * img_h.unsqueeze(1)
        return kps_2D_pix
        
# # Testing
# import torch 
# kps_2D          = torch.rand(10, 2)
# K               = torch.rand(3, 3)
# kps_2D_batch    = torch.rand(5, 10, 2)
# K_batch         = torch.rand(5, 3, 3)
# az_el, azel     = Bearing.torch_compute_azimuth_elevation(kps_2D, K)
# az_el_batch, azel_batch = Bearing.torch_batch_compute_azimuth_elevation(kps_2D_batch, K_batch)
# kps_2D_pix    = Bearing.torch_batch_compute_kps_2D_pix(kps_2D_batch, K_batch)
# print(f'Shape of single azimuth and elevation angles: {az_el.shape}')
# print(f'Shape of single alternating azimuth and elevation angles: {azel.shape}')
# print(f'Shape of batch azimuth and elevation angles: {az_el_batch.shape}')
# print(f'Shape of batch alternating azimuth and elevation angles: {azel_batch.shape}')
# print(f'Shape of batch unnormalized pixel coordinates: {kps_2D_pix.shape}')
# kps_2D_norm     = torch.tensor([
#                             [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],   # Batch 1
#                             [[0.1, 0.2], [0.6, 0.7], [0.9, 0.8]]    # Batch 2
#                         ], dtype=torch.float32)
# real_K_batch    = torch.tensor([
#                                 [[500, 0, 320],
#                                 [0, 500, 240],
#                                 [0,   0,   1]],
#                                 [[600, 0, 400],
#                                 [0, 600, 300],
#                                 [0,   0,   1]]
#                             ], dtype=torch.float32)
# kps_2d_pix2     = Bearing.torch_batch_compute_kps_2D_pix(kps_2D_norm, real_K_batch)
# print(f'Normalized pixel coordinates: {kps_2D_norm}')
# print(f'Unnormalized pixel coordinates: {kps_2d_pix2}')
# print(f'Shape of unnormalized pixel coordinates: {kps_2d_pix2.shape}')
# pdb.set_trace()


class Camera:
    """ 
    A Python class that includes methods for camera processing 
    """

    @staticmethod
    def camera_matrix(Nu, Nv, sensor_width_mm, sensor_height_mm, focal_length_mm, ccx = None, ccy = None, square_pixels = True):
        """
        Write and calculate camera parameters and store in a JSON file

        Input:
            Nu (int): number of pixels in the horizontal direction
            Nv (int): number of pixels in the vertical direction
            sensor_width_mm (float): width of the sensor in mm
            sensor_height_mm (float): height of the sensor in mm
            focal_length_mm (float): focal length of the camera in mm
            ccx (int, optional): principal point x-coordinate (default is Nu / 2)
            ccy (int, optional): principal point y-coordinate (default is Nv / 2)
            square_pixels (bool, optional): whether pixels are square (default is True)
        
        Output:
            camera_matrix (np.array): 3x3 camera matrix

        Src: https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
        """

        # ppx             = (sensor_width_mm * 1e-3 ) / Nu # physical size of pixel in x (horizontal) direction; in meters
        # ppy             = (sensor_height_mm * 1e-3) / Nv # physical size of pixel in y (vertical) direction; in meters
        fx              = (focal_length_mm * Nu) / sensor_width_mm # focal length in pixels for x (horizontal) dimension
        fy              = (focal_length_mm * Nv) / sensor_height_mm # focal length in pixels for y (vertical) dimension

        # calculate principal point
        if ccx is None:
            ccx = Nu / 2 # principal point x-coordinate is middle of image when not provided
        if ccy is None:
            ccy = Nv / 2 # principal point y-coordinate is middle of image when not provided

        # create the camera matrix
        if square_pixels: # square pixel assumption
            camera_matrix   = [
                                [fx, 0, ccx],
                                [0, fx, ccy],
                                [0, 0, 1]
                            ]
        else:
            camera_matrix   = [
                                [fx, 0, ccx],
                                [0, fy, ccy],
                                [0, 0, 1]
                            ]
        
        return np.array(camera_matrix, dtype=np.float32)

class PnP:
    """ Class for solving for pose using PnP """

    @staticmethod
    def pnp_solve(
                    kps_3D
                    ,kps_2D
                    ,camera_matrix
                    ,camera_dist_coeffs = np.zeros(5, dtype = np.float32)
                    ,rvec = None
                    ,tvec = None
                    ,useExtrinsicGuess = False
        ):
        """ Solve for pose using PnP 
        
        Inputs:
            kps_3D (np.array): (N, 3) array representing the 3D keypoints in the object frame, in meters
            kps_2D (np.array): (N, 2) array representing the 2D keypoints in the image frame, in pixel coordinates
            camera_matrix (np.array): 3x3 camera intrinsic matrix
            camera_dist_coeffs (np.array): (5,) camera distortion coefficients
            rvec (np.array, optional): (3,) initial rotation vector
            tvec (np.array, optional): (3,) initial translation vector
            useExtrinsicGuess (bool, optional): whether to use the provided initial estimates for rotation and translation
        
        Outputs:
            tr (np.array): (3,) translation vector from the object frame to the camera frame
            quat (np.array): (4,) quaternion representing the rotation from the object frame to the camera frame
        """
        
        _, R_exp, tr    = cv2.solvePnP(
                                    kps_3D
                                    ,kps_2D
                                    ,camera_matrix
                                    ,camera_dist_coeffs
                                    ,rvec
                                    ,tvec
                                    ,useExtrinsicGuess
                                    ,flags = cv2.SOLVEPNP_EPNP
                                    )
        R_pr, _     = cv2.Rodrigues(R_exp) # convert rotation matrix to rotation vector
        # quat        = R.from_matrix(R_pr).as_quat(scalar_first = True) # convert rotation vector to quaternion, [qw qvec] format, right scalar first
        quat        = R.from_matrix(R_pr).as_quat() # some versions of scipy have scalar last as default with no ability to change in as_quat fcn 
        quat        = np.concatenate([quat[-1:], quat[:-1]]) # convert to right scalar first format
        tr          = np.squeeze(tr) # convert to numpy array
        
        return tr, quat
    
    @staticmethod
    def ransac_pnp_solve(
                            kps_3D
                            ,kps_2D
                            ,camera_matrix
                            ,camera_dist_coeffs = np.zeros(5, dtype = np.float32)
                            ,rvec = None
                            ,tvec = None
                            ,useExtrinsicGuess = False
                            ,iterations = 5000
                            ,reprojection_error = 8.0
                            ,confidence = 0.99
                            ,flags = cv2.SOLVEPNP_EPNP
                            ,inliers = None
        ):
        """ Solve for pose using PnP RANSAC 

        Inputs:
            kps_3D (np.array): (N, 3) array representing the 3D keypoints in the object frame, in meters
            kps_2D (np.array): (N, 2) array representing the 2D keypoints in the image frame, in pixel coordinates
            camera_matrix (np.array): 3x3 camera intrinsic matrix
            camera_dist_coeffs (np.array): (5,) camera distortion coefficients
            rvec (np.array, optional): (3,) initial rotation vector
            tvec (np.array, optional): (3,) initial translation vector
            useExtrinsicGuess (bool, optional): whether to use the provided initial estimates for rotation and translation
            iterations (int, optional): number of iterations
            reprojection_error (float, optional): maximum allowed reprojection error
            confidence (float, optional): confidence level
            flags (int, optional): flags for solvePnPRansac
            inliers (np.array, optional): inliers

        Outputs:
            q_rsf: quaternion in right scalar first format
            t: translation vector
            inliers: inlier indices
        """
    
        go, rot_vec, tr_vec, inliers    = cv2.solvePnPRansac(   
                                                                objectPoints = kps_3D
                                                                ,imagePoints = kps_2D      
                                                                ,cameraMatrix = camera_matrix
                                                                ,distCoeffs = camera_dist_coeffs
                                                                ,rvec = rvec
                                                                ,tvec = tvec
                                                                ,useExtrinsicGuess = useExtrinsicGuess
                                                                ,iterationsCount = iterations
                                                                ,reprojectionError = reprojection_error
                                                                ,confidence = confidence
                                                                ,inliers = inliers
                                                                ,flags  = flags
                                                            )
    
        if not go:
            # print('PnP RANSAC failed')
            return None
        
        R_pr, _ = cv2.Rodrigues(rot_vec)
        # quat    = R.from_matrix(R_pr).as_quat(scalar_first = True)
        quat    = R.from_matrix(R_pr).as_quat() # some versions of scipy have scalar last as default with no ability to change in as_quat fcn
        quat    = np.concatenate([quat[-1:], quat[:-1]]) # make right scalar first
        tr      = np.squeeze(tr_vec) # convert to numpy array
        # inliers are the indices of the inliers

        return tr, quat, inliers

class PoseError:
    """ Class for computing pose errors """

    @staticmethod
    def rmse_calc(gt, pred):
        """ Calculate the Root Mean Squared Error (RMSE) between the ground truth and predicted values """
        rmse   = np.sqrt( np.mean( (gt - pred) **2) )
        return rmse
    
    @staticmethod
    def norm_err_calc(gt, pred, norm_type = 2):
        """ Calculate the Lp norm between the ground truth and predicted values """
        norm_calc   = np.linalg.norm(gt - pred, ord = norm_type)
        return norm_calc

    @staticmethod
    def E_T(tr, tr_hat):
        """ This function computes 2norm of difference between two translation vectors """
        e_tr    = np.linalg.norm(tr-tr_hat, ord = 2) 
        return e_tr
    
    @staticmethod
    def E_TN(tr, tr_hat):
        """ This function computes normalized 2norm of difference between two translation vectors, the first vector is often the truth translation """
        err_tn  = PoseError.E_T(tr, tr_hat) / np.linalg.norm(tr, ord = 2)
        return err_tn

    @staticmethod
    def E_R(quat, quathat):
        """ This function computes rotational error by finding the angle of smallest rotation between the truth and estimated attitudes, robust form """
        quat        = quat / np.linalg.norm(quat)
        quathat     = quathat / np.linalg.norm(quathat)
        abs_dotprod = np.abs( np.dot(quat, quathat) )
        abs_dotprod = np.where(abs_dotprod > 1, np.abs(QuatMath.q_error_shu(quat, quathat)[0]), abs_dotprod)
        err_r       = 2 * np.arccos( abs_dotprod )
        return err_r
    
    def batch_E_T(tr, tr_hat):
        """ This function computes the 2norm of difference between two batches of translation vectors """
        err_t   = np.linalg.norm(tr - tr_hat, ord = 2, axis = 1)
        # shape of err_t is (N,)
        return err_t

    @staticmethod
    def batch_E_TN(tr, tr_hat):
        """ This function computes the 2norm of difference between two batches of translation vectors """
        err_tn  = np.linalg.norm(tr - tr_hat, ord = 2, axis = 1) / np.linalg.norm(tr, ord = 2, axis = 1)
        # shape of err_tn is (N,)
        return err_tn
    
    @staticmethod
    def batch_E_R(quat, quathat):
        """ This function computes the rotational error between two batches of quaternions (non-robust) """
        quat        = quat / np.linalg.norm(quat, axis = 1, keepdims = True)
        quathat     = quathat / np.linalg.norm(quathat, axis = 1, keepdims = True)
        abs_dotprod = np.abs( np.sum(quat * quathat, axis = 1) )
        err_r       = 2 * np.arccos( abs_dotprod )
        # shape of err_r is (N,)
        return err_r
    
    @staticmethod 
    def torch_batch_E_T(tr, tr_hat):
        """ This function computes the 2norm of difference between two batches of torch translation vectors """
        import torch
        err_t   = torch.norm(tr - tr_hat, dim = 1) 
        # shape of err_t is (N,)
        return err_t
    
    @staticmethod
    def torch_batch_E_TN(tr, tr_hat):
        """ This function computes the 2norm of difference between two batches of torch translation vectors """
        import torch
        err_tn  = torch.norm(tr - tr_hat, dim = 1) / torch.norm(tr, dim = 1)
        # shape of err_tn is (N,)
        return err_tn
    
    @staticmethod
    def torch_batch_E_R(quat, quathat):
        """ This function computes the rotational error between two batches of torch quaternions (non-robust) """
        import torch
        quat        = quat / torch.norm(quat, dim = 1, keepdim = True)
        quathat     = quathat / torch.norm(quathat, dim = 1, keepdim = True)
        abs_dotprod = torch.abs( torch.sum(quat * quathat, dim = 1) )
        err_r       = 2 * torch.acos( torch.clamp(abs_dotprod, -1, 1) )
        # shape of err_r is (N,)
        return err_r

    
    @staticmethod
    def jax_dense_svd_cov(J, eps = 1e-15):
        """ 
        Compute the Covariance of a NLS problem using the dense SVD method for the Jacobian of the residuals evaluated at best estimates
        Covariance is approximated as the pseudo-inverse of the Fisher Information Matrix 
        
        J = U * D * V^T  =>  J^T J = V * D^(2†) * V^T
        D^(2†) = diag(1 / sigma_i^2)
        (J^T J)† = V * D^(2†) * V^T

        Inputs:
        J (jnp.array): Jacobian matrix of the residuals evaluated at the estimated parameters
        eps (float): small value to avoid division by zero

        Ouputs:
        J_T_J_inv (jnp.array): covariance, which is approximated as the inverse of the Fisher Information Matrix

        SRC: http://ceres-solver.org/nnls_covariance.html, look for DENSE_SVD under CovarianceAlgorithmType Covariance::Options::algorithm_type
        """
        U, D, V_T   = jnp.linalg.svd(J, full_matrices = False)
        D_temp      = jnp.where(D > eps, 1 / (D**2), 0.0)
        D2_dagg     = jnp.diag(D_temp)
        J_T_J_inv   = V_T.T @ D2_dagg @ V_T
        
        return J_T_J_inv

class Projection:
    """ Pose information onto images """
    
    @staticmethod
    def project_keypoints(
                            q: np.ndarray
                            , r: np.ndarray
                            , K: np.ndarray
                            , keypoints: np.ndarray
                            , dist: np.ndarray = None
        ) -> np.ndarray:

        """
        Projecting 3D keypoints to 2D image coordinates

        Input:
            q: quaternion (4,) (np.ndarray)
            r: position (3,) (np.ndarray)
            K: camera intrinsic matrix (3,3) (np.ndarray)
            keypoints: 3D keypoints (N, 3) (np.ndarray)
            dist: distortion coefficients (5,) (np.ndarray), defaults to np.zeros(5) if None

        Output:
            points2D: 2D keypoints (N, 2) (np.ndarray)

        Src: https://github.com/tpark94/spnv2/blob/dbcf0de8813da56529bb7467a87c6cdacfc46d23/core/utils/postprocess.py#L78
        """

        if dist is None:
            dist        = np.zeros(5)
        
        if keypoints.shape[0] != 3:
            keypoints   = keypoints.T
        
        num_kps     = keypoints.shape[1]
        # convert keypoints to homogeneous coordinates
        # homogenous coordinates allow us to rpresent 3D points in 4D space
        # (x,y, z)  -> (x, y, z, 1)
        # do this for ease: (tranformed points) = M x (original points)
        ones_row    = np.ones((1, num_kps), dtype = keypoints.dtype)
        keypoints   = np.vstack((keypoints, ones_row)) # 4 x N
        tmat        = QuatMath.q2trfm(q) # transformation matrix
        # construct [R | t] pose matrix, shape (3, 4) where R is tmat.T
        pmat        = np.hstack( (tmat.T, r.reshape(3, 1)) )
        xyz         = pmat @ keypoints # 3 x 4 @ 4 x N = 3 x N
        
        # perspective division, this step is to convert from 3D to 2D
        x0          = xyz[0, :] / xyz[2, :]
        y0          = xyz[1, :] / xyz[2, :]

        dist0       = dist[0]
        dist1       = dist[1]
        dist2       = dist[2]
        dist3       = dist[3]
        dist4       = dist[4]

        # radial distortion
        r2          = x0**2 + y0**2
        cdist       = 1 + dist0*r2 + dist1*r2**2 + dist4*r2**3
        # tangential distortion
        xdist       = x0*cdist + 2*dist2*x0*y0 + dist3*(r2 + 2*x0**2)
        ydist       = y0*cdist + dist2*(r2 + 2*y0**2) + 2*dist3*x0*y0

        # apply camera matrix
        u           = K[0, 0]*xdist + K[0, 2]
        v           = K[1, 1]*ydist + K[1, 2]
        points2D    = np.vstack((u, v)).T

        return points2D
 
    @staticmethod
    def project_bbox_kps_array_2cv2np(
                                        img_fn_or_arr 
                                        ,box = None
                                        ,keypoints = None  
                                        ,box_color = (0, 0, 255) 
                                        ,keypoint_color = (0, 255, 0) 
                                        ,bgr_flag = True 
                                        ,origin_flag = False 
                                        ,origin_thickness = -1
                                        ,origin_color = (128, 0, 128)
                                        ,origin_size = 10
                                        ,circle_thickness = 2
                                        ,circle_size = 5
                                        ,box_thickness = 2
                                        ,box_label = None
                                        ,box_score = None
                                        ,box_label_pos = (10, -10)
                                        ,box_score_pos = (100, -10)
                                        ,label_color = (0, 128, 255)
                                        ,score_color = (255, 0, 0)
                                ):
        """
        Projects bounding boxes (optional) and keypoints (optional) onto an numpy array image that we read via a provided filename and returns a NumPy image array using OpenCV (cv2)

        Input:
            img_fn (absolute file path of the image or np.array): Image filepath or numpy BGR array of shape (H, W, 3)
            box (list): Bounding box coordinates in the format [xmin, ymin, xmax, ymax], pixel coordinates
            box_color (tuple): Color of the bounding boxes in BGR format, default is red
            keypoints (np.array, optional): Keypoints coordinates in the format (x, y), shape (N, 2), pixel coordinates
            keypoint_color (tuple): Color of the keypoints in BGR format, default is green
            bgr_flag (bool): Whether to return the image in BGR format, default is True
            origin_flag (bool): Whether to draw the origin (first keypoint) as a filled circle, default is False
            origin_thickness (int): Thickness of the origin circle, default is -1 (filled circle)
            origin_color (tuple): Color of the origin circle in BGR format, default is purple
            origin_size (int): Size of the origin circle, default is 10
            circle_thickness (int): Thickness of the keypoint circles, default is 2
            circle_size (int): Size of the keypoint circles, default is 5
            box_thickness (int): Thickness of the bounding box, default is 2
            box_label (str): Label to be drawn on the bounding box, default is None
            box_score (str): Score to be drawn on the bounding box, default is None
            box_label_pos (tuple): Position of the box label in relation to the top left corner of the box
            box_score_pos (tuple): Position of the box score in relation to the the top left of the box
            label_color (tuple): Color of the box labels in BGR format, default is orange
            score_color (tuple): Color of the box scores in BGR format, default is blue

        Output:
            img_np (np.array): NumPy image array with the bounding boxes and keypoints drawn, returns BGR or RGB np.array
        """
        if isinstance(img_fn_or_arr, np.ndarray): # requires exact import: import numpy as np
            img_arr = img_fn_or_arr
        else:
            img_arr= cv2.imread(img_fn_or_arr) # read image, # dimensions of the image are (H, W, 3)
        
        if len(img_arr.shape) == 2 or img_arr.shape[2] == 1:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2BGR)

        # img_np  = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR) # convert RGB to BGR for use with cv2
        img_np  = img_arr.copy()

        if box is not None:
            xmin, ymin, xmax, ymax = box 
            xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
            if box_label is not None:
                cv2.putText(img_np, box_label, (xmin + box_label_pos[0], ymin + box_label_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
                if box_score is not None:
                    cv2.putText(img_np, str(box_score), (xmin + box_score_pos[0], ymin + box_score_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

        if keypoints is not None:
            keypoints   = np.round(keypoints).astype(int)
            for count, kp in enumerate(keypoints):
                x, y    = kp.tolist()
                if count == 0 and origin_flag == True:
                    cv2.circle(img_np, (x, y), origin_size, origin_color, origin_thickness) # origin_type = -1 for filled circle
                else:
                    cv2.circle(img_np, (x, y), circle_size, keypoint_color, circle_thickness)
        if not bgr_flag:
            img_np  = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # convert BGR back to RGB if needed

        return img_np
    
class PoseDistro:
    """ Class containing methods to detail pose distributions of underlying data """
    
    @staticmethod
    def load_pose(filepath, pose_key = 'pose'):
        """ Helper function to load pose data from a JSON file """

        try: # load the pose data from the metadata file
            with open(filepath, 'r') as f:
                data = json.load(f)
                if pose_key in data.keys():
                    return data[pose_key]
                else:
                    print(f'Pose key {pose_key} not found in {filepath}')
                    return None
        except Exception as e: # catch and print any errors
            print(f'Error loading pose data from {filepath}: {e}')
            return None
    
    @staticmethod
    def concurrent_pose_hists(data_list, description, export_path, pose_key = 'pose', print_flag = False): 
        """
        Generate histograms for translation and pose values in collected metadata files using concurrent processing

        Inputs:
            data_list (list): list containing paths of metadata files
            export_path (str): the path to export the histograms to
            description (str): description of the metadata set
            data_ext (str): the extension of the metadata files
            pose_key (str): the key in the metadata files containing the pose values
        """
        hist_t_path = os.path.join(export_path, f'{description}_translation_histograms.png')
        hist_a_path = os.path.join(export_path, f'{description}_attitude_histograms.png')

        metadata_files  = sorted(data_list)
        translations    = []
        atts            = []

        # load json files concurrently
        with ThreadPoolExecutor() as executor:
            poses   = list( executor.map(lambda fp: PoseDistro.load_pose(fp, pose_key), metadata_files) )
            # executor.map returns a generator, so convert to list, lambda function to call load_pose with the correct arguments, metadata_files is the iterable
        

        # filter out any files where the pose key was missing or an error occurred.
        poses   = [p for p in poses if p is not None]
        if not poses:
            if print_flag:
                print('No valid pose data found in the metadata files.')
            return
        
        poses_array = np.array(poses)
        trs_array   = poses_array[:, 0:3]
        atts_array  = poses_array[:, 3:] # right scalar first quaternions
        num_imgs    = trs_array.shape[0]

        trs_norm    = np.linalg.norm(trs_array, axis = 1, ord = 2, keepdims = True)
        # add norm of translation to translation array
        trs_array_r = np.hstack( (trs_array, trs_norm) )
        # plot histograms for translation
        fig, axes   = plt.subplots(4, 1, figsize = (12, 12))
        labels      = ['x', 'y', 'z', 'r']
        for i, ax in enumerate(axes):
            ax.hist(trs_array_r[:, i], alpha = 0.7, edgecolor = 'black')
            ax.set_xlabel(f'{labels[i]} (meters)')
            ax.set_ylabel('Frequency')
        fig.suptitle(f'Translation Histograms for Imagery Set(s), {num_imgs} Images Processed')
        fig.tight_layout(rect = [0, 0, 1, 0.96])
        fig.savefig(hist_t_path)
        plt.close(fig)
        
        # this angle is the euler-axis angle
        euler_angle = np.degrees( 2 * np.arccos(atts_array[:, 0]) ) # degrees
        ########### using QuatMath ###########
        # bottleneck b/c QuatMath.rmat_to_euler_angles is not vectorized
        ypr_angles2 = np.array([QuatMath.rmat_to_euler_angles(QuatMath.q2rotm(q), sequence = 'zyx') for q in atts_array])
        ypra_angles = np.hstack( (ypr_angles2, euler_angle.reshape(-1, 1)) )
        ########### using QuatMath ###########
        ############# using SciPy ############
        # # normalize quats
        # atts_array  = atts_array / np.linalg.norm(atts_array, axis = 1, keepdims = True)
        # # use scipy to convert quaternions to Rotation objects
        # rotations   = R.from_quat(atts_array)
        # euler_angs  = rotations.as_euler('zyx', degrees = True)
        # ypra_angles = np.hstack( (euler_angs, euler_angle.reshape(-1, 1)) )
        ############# using SciPy ############
        # plot histograms for attitude
        fig, axes   = plt.subplots(4, 1, figsize = (12, 12))
        labels      = ['roll', 'pitch', 'yaw', 'angle of smallest rotation']
        for j, ax in enumerate(axes):
            ax.hist(ypra_angles[:, j], alpha = 0.7, edgecolor = 'black')
            ax.set_xlabel(f'{labels[j]} (degrees)')
            ax.set_ylabel('Frequency')
        fig.suptitle(f'Attitude Histograms for Imagery Set(s), {num_imgs} Images Processed')
        fig.tight_layout(rect = [0, 0, 1, 0.96])
        fig.savefig(hist_a_path)
        plt.close(fig)

        if print_flag:
            print(f'Translation Histogram path: {hist_t_path}')
            print(f'Attitude Histogram path: {hist_a_path}')
    
    @staticmethod
    def pose_hists(data_list, description, export_path, pose_key = 'pose', print_flag = False):
        """
        Generate histograms for translation and pose values in collected metadata files

        Inputs:
            data_list (list): list containing paths of metadata files
            export_path (str): the path to export the histograms to
            description (str): description of the metadata set
            data_ext (str): the extension of the metadata files
            pose_key (str): the key in the metadata files containing the pose values
        """

        hist_t_path = os.path.join(export_path, f'{description}_translation_histograms.png')
        hist_a_path = os.path.join(export_path, f'{description}_attitude_histograms.png')

        metadata_files  = sorted(data_list)
        translations    = []
        atts            = []

        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                data    = json.load(f)
                if pose_key in data.keys():
                    tr  = data[pose_key][0:3]
                    att = data[pose_key][3:]
                    translations.append(tr)
                    atts.append(att)
                else:
                    print(f"Translation or attitude key not found in {metadata_file}")
        trs_array   = np.array(translations)
        num_imgs    = trs_array.shape[0]
        trs_norm    = np.linalg.norm(trs_array, axis = 1, ord = 2)
        # add norm of translation to translation array
        trs_array_r = np.hstack( (trs_array, trs_norm.reshape(-1, 1)) )
        # Plot histograms for translation
        plt.figure(figsize = (12, 12))
        for i, label in enumerate(['x', 'y', 'z', 'r']):
            plt.subplot(4, 1, i + 1)
            plt.hist(trs_array_r[:, i], alpha = 0.7, edgecolor = 'black'
                    # , bins = 30 
                    )
            plt.xlabel(f'{label} (meters)')
            plt.ylabel('Frequency')
        plt.suptitle(f'Translation Histograms for Imagery Set(s), {num_imgs} Images Processed')
        plt.tight_layout(rect = [0, 0, 1, 0.96])
        plt.savefig(hist_t_path)
        
        atts_array  = np.array(atts) # right scalar first quaternion 
        # this angle is the euler-axis angle
        euler_angle = np.degrees( 2 * np.arccos(atts_array[:, 0]) ) # degrees

        ypr_angles2 = np.array([QuatMath.rmat_to_euler_angles(QuatMath.q2rotm(q), sequence = 'zyx') for q in atts_array])
        ypra_angles = np.hstack( (ypr_angles2, euler_angle.reshape(-1, 1)) )
    
        plt.figure(figsize = (12, 12))
        for j, label in enumerate(['roll', 'pitch', 'yaw', 'angle of smallest rotation']):
            plt.subplot(4, 1, j + 1)
            plt.hist(ypra_angles[:, j], alpha = 0.7, edgecolor = 'black'
                    # , bins = 30 
                    )
            plt.xlabel(f'{label} (degrees)')
            plt.ylabel('Frequency')
        plt.suptitle(f'Attitude Histograms for Imagery Set(s), {num_imgs} Images Processed')
        plt.tight_layout(rect = [0, 0, 1, 0.96])
        plt.savefig(hist_a_path)

        if print_flag:
            print(f'Translation Histogram path: {hist_t_path}')
            print(f'Attitude Histogram path: {hist_a_path}')

class InferUtils:
    """ A class containing methods for deployed inference for pose error models """
    
    @staticmethod
    def onnx_model_input_deets(model_path, printing_flag = True, logging_flag = False):
        """
        Takes in the path to an ONNX model and prints out the details of a model input

        Inputs:
        model_path: str, the path to the model
        printing_flag: bool, whether to print the details (default is True)
        logging_flag: bool, whether to log the details (default is False)
        """
        import onnxruntime
        sess    = onnxruntime.InferenceSession(model_path) # setup model
        
        model_str   = f'ONNX Model: {model_path}'
        model_str2  = f'Model Inputs and Outputs:'
        if printing_flag:
            print(model_str)
            print(model_str2)
        if logging_flag:
            logging.info(model_str)
            logging.info(model_str2)

        for input in sess.get_inputs():
            input_name  = input.name
            input_type  = input.type
            input_size  = input.shape
            input_n_str = f"Input Name: {input_name}"
            input_t_str = f"Type: {input_type}"
            input_s_str = f"Expected Shape: {input_size}"
            if printing_flag:
                print(input_n_str)
                print(input_t_str)
                print(input_s_str)
            if logging_flag:
                logging.info(input_n_str)
                logging.info(input_t_str)
                logging.info(input_s_str)

        for output in sess.get_outputs():
            output_name     = output.name
            output_type     = output.type
            output_size     = output.shape
            output_n_str    = f"Output Name: {output_name}"
            output_t_str    = f"Type: {output_type}"
            output_s_str    = f"Expected Shape: {output_size}"
            if printing_flag:
                print(output_n_str)
                print(output_t_str)
                print(output_s_str)
            if logging_flag:
                logging.info(output_n_str)
                logging.info(output_t_str)
                logging.info(output_s_str)
    
    @staticmethod
    def onnx_model_setup(model_path, ort_device = ['CUDAExecutionProvider']):
        """
        Creates ONNX Runtime model by taking in the path to an ONNX model and returns the session, input name and output names

        Inputs:
        model_path: str, the path to the model
        ort_device: list of str, the device to run the model on (default is ['CPUExecutionProvider'])
        -->  can also be ['CUDAExecutionProvider', 'CPUExecutionProvider'] --> use GPU if available then CPU
        
        Outputs:
        sess: ONNX Runtime InferenceSession object
        in_names: str, the input name for the model
        out_names: list of str, the output names for the model
        """
        import onnxruntime
        sess        = onnxruntime.InferenceSession(model_path, providers = ort_device)
        in_names    = sess.get_inputs()[0].name
        out_names   = [out.name for out in sess.get_outputs()]
        return sess, in_names, out_names 

    @staticmethod
    def ort_krcnn_inference(sess, in_names, out_names, img_input, output_keys = ['boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores']):
        """
        This function infers with an ONNX model using ONNX Runtime for Keypoint R-CNN

        Inputs:
        sess: ONNX Runtime InferenceSession object
        in_names: str, the input name for the model
        out_names: list of str, the output names for the model
        img_input: np.array, the image input
        output_keys: list of str, the keys for the model outputs (default is ['boxes', 'labels', 'scores', 'keypoints', 'keypoints_scores'])
    
        Outputs:
        output_dict: dict, the output dictionary that mimics the output of the model described by out_names; 
        returning numpy arrays
        """
        
        if isinstance(img_input, np.ndarray):
            img_np  = np.expand_dims(img_input, axis = 0)
            input   = {in_names: img_np} # add batch dimension when input is a numpy array
        else:
            raise ValueError('Input must be a numpy array or torch tensor')
        inference       = sess.run(out_names, input)
        output_dict     = {}
        for idx, val in enumerate(inference): # iterate over the outputs and add them to the output dictionary
            output_dict[output_keys[idx]]   = val

        return output_dict
    
    @staticmethod
    def ov2024_model_input_deets(xml_path, printing_flag = True, logging_flag = False):
        """
        This function takes in the path to an OpenVINO model and prints out the details of a model input

        Inputs:
        xml_path: str, the path to the model
        
        Output:
        None
        """
        import openvino as ov
        core    = ov.Core()
        model   = core.read_model(xml_path)
        
        model_str   = f'OpenVINO Model: {xml_path}'
        model_str2  = f'Model Inputs and Outputs:'
        if printing_flag:
            print(model_str)
            print(model_str2)
        if logging_flag:
            logging.info(model_str)
            logging.info(model_str2)

        for idx, model_input in enumerate(model.inputs):
            input_name  = model_input.get_any_name()
            input_type  = model_input.get_element_type()
            input_shape = model_input.get_partial_shape()
            input_i_str = f'Input #{idx}'
            input_n_str = f'Name: {input_name}'
            input_t_str = f'Element Type: {input_type}'
            input_s_str = f'Shape: {input_shape}'
            if printing_flag:
                print(input_i_str)
                print(input_n_str)
                print(input_t_str)
                print(input_s_str)
            if logging_flag:
                logging.info(input_i_str)
                logging.info(input_n_str)
                logging.info(input_t_str)
                logging.info(input_s_str)

            # print(f"  Input #{idx}")
            # print(f"    Name          : {input_name}")
            # print(f"    Element Type  : {input_type}")
            # print(f"    Shape         : {input_shape}\n")

        for idx, model_output in enumerate(model.outputs):
            output_name     = model_output.get_any_name()
            output_type     = model_output.get_element_type()
            output_shape    = model_output.get_partial_shape()
            output_i_str    = f'Output #{idx}'
            output_n_str    = f'Name: {output_name}'
            output_t_str    = f'Element Type: {output_type}'
            output_s_str    = f'Shape: {output_shape}'
            if printing_flag:
                print(output_i_str)
                print(output_n_str)
                print(output_t_str)
                print(output_s_str)
            if logging_flag:
                logging.info(output_i_str)
                logging.info(output_n_str)
                logging.info(output_t_str)
                logging.info(output_s_str)
            # print(f"  Output #{idx}")
            # print(f"    Name          : {output_name}")
            # print(f"    Element Type  : {output_type}")
            # print(f"    Shape         : {output_shape}\n")
    
    @staticmethod
    def ov2024_onnx_or_IR_compile(model_path, device = 'AUTO'):
        """
        This function compiles an ONNX or IR model for inference with OpenVINO 2024

        Input:
        model_path: str, the path to the model, can be an ONNX or IR model
        device: str, the device to run the model on (default is 'AUTO'), only 'GPU' if using Intel Integrated GPUs
        
        Output:
        compiled_model: OpenVINO model, the compiled OpenVINO model
        """
        import openvino as ov
        name                    = os.path.basename(model_path) # get the name of the model
        model_name, model_ext   = os.path.splitext(name) # get the name and extension of the model
        if model_ext == '.onnx': # if the model is an ONNX model
            ov_onnx_model   = ov.convert_model(model_path)
            compiled_model  = ov.compile_model(ov_onnx_model, device_name = device)
        
        elif model_ext == '.xml' or model_ext == '.bin': # if the model is an IR model
            core            = ov.Core()
            ov_ir_model     = core.read_model(model_path)
            compiled_model  = ov.compile_model(ov_ir_model, device_name = device)

        return compiled_model
    
    @staticmethod
    def openvino_model_setup(model_path, output_keys, device = 'AUTO'):
        """
        This function sets up the OpenVINO model for inference

        Inputs:
            model_path: str, the path to the model
            output_keys: list of str, the keys for the model outputs
            device: str, the device to run the model on (default is 'AUTO'), only 'GPU' if using Intel Integrated GPUs
        
        Outputs:
            compiled_model: OpenVINO model, the compiled OpenVINO model
            ov_outpts: dict, the output layers of the model
            keys: list, the keys of the output layers
        """
        compiled_model  = InferUtils.ov2024_onnx_or_IR_compile(model_path, device = device)
        cm_outputs      = {}
        for idx, output_key in enumerate(output_keys):
            output_idx              = compiled_model.output(idx)
            cm_outputs[output_key]  = output_idx

        return compiled_model, cm_outputs

    @staticmethod
    def ov_krcnn_inference(compiled_model, cm_outputs, img_input):
        """
        This function takes in an image input (either a numpy array or torch tensor) and runs inference with the provided Mask R-CNN OpenVINO compiled model
        
        Inputs:
            compiled_model: OpenVINO model, the compiled OpenVINO model
            cm_outputs: dict, the compiled model outputs
            img_input: np.array, the image input
        
        Outputs:
            output_dict: dict, the output dictionary that mimics the output of the model described by cm_outputs; 
            returning numpy arrays
        """
        if isinstance(img_input, np.ndarray):
            input   = np.expand_dims(img_input, axis = 0)
        else:
            raise ValueError('Input must be a numpy array or torch tensor')
        inference       = compiled_model(input)
        output_dict     = {}
        for key, value in cm_outputs.items():
            output_dict[key]    = inference[value]
        
        return output_dict
    
    @staticmethod
    def cv2_pad_to_square( img, pad_color = (0, 0, 0) ):
        """
        This functions takes a cv2 image array (uint8 numpy array) and pads it to a square using cv2.copyMakeBorder

        Input:
        img: np.array, the image array
        pad_color: tuple, the color to pad the image with
        
        Output: 
        pad_img: np.array, the padded image array
        or 
        img: np.array, the original image array if it is already a square
        """
        img_h, img_w = img.shape[:2]
        if img_h == img_w:
            return img
        else:
            max_dim             = max(img_h, img_w)
            horizontal_padding  = (max_dim - img_w) // 2
            vertical_padding    = (max_dim - img_h) // 2
            pad_img             = cv2.copyMakeBorder(
                                                    img
                                                    ,top = vertical_padding
                                                    ,bottom = vertical_padding
                                                    ,left = horizontal_padding
                                                    ,right = horizontal_padding
                                                    ,borderType = cv2.BORDER_CONSTANT
                                                    ,value = pad_color
                                                    )
            return pad_img 
    
    @staticmethod
    def cv2_preprocess_img_np( 
                            img_fp_or_arr
                            ,resize_tuple = (512, 512)
                            ,imagenet_norm = False
                            ,pad_color = (0, 0, 0)
                            ,return_bgr = False 
                        ):
        """
        This function takes an image filepath and using cv2 preprocesses them for inference
        Inference is designed to work with Keypoint RCNN, legacy bounding bounding box and keypoint regression networks, and spnv2 architecture 
        Normalization can be set to imagenet normalization or or standard 0 to 1 normalization
        A flag is included for inferencing on grayscale images

        Input:
        img_fp (str): the image filepath
        resize_tuple (tuple): the dimensions to resize the images provided as (height, width)
        imagenet_norm (bool): whether to use imagenet normalization
        pad_color (tuple): the color to pad the image with
        return_bgr (bool): whether to return the BGR image array that is padded, resized, but not normalized

        Output:
        img_rgb_chw (np.array): the preprocessed image array, which is a CxHxW RGB image array that is normalized by either imagenet or 0 to 1 normalization

        """
        if imagenet_norm:
            mean    = np.array([0.485, 0.456, 0.406])
            std     = np.array([0.229, 0.224, 0.225])
        else:
            mean    = np.array([0, 0, 0])
            std     = np.array([1, 1, 1])

        # read in RBG image and return BGR image array
        # flags = cv2.IMREAD_COLOR, is the default flag that will return a BGR image array
        if isinstance(img_fp_or_arr, np.ndarray):
            img_bgr = img_fp_or_arr
        else:
            img_bgr     = cv2.imread(img_fp_or_arr, flags = cv2.IMREAD_COLOR) # returns uint8 numpy array that is Height X Width X Channel with values from 0 to 255
        img_bgr     = InferUtils.cv2_pad_to_square(img_bgr, pad_color = pad_color) # pad to square first before resizing, default pad_color input is black (0,0,0)
        img_bgr     = cv2.resize(src = img_bgr, dsize = (resize_tuple[1], resize_tuple[0])) # dsize is (width, height), image is still Height X Width X Channel
        img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert to RGB for albumentations, expands back to three channels
        img_rgb_n   = ( (img_rgb / 255.0) - mean ) / std # normalize image
        img_rgb_chw = np.transpose(img_rgb_n, (2, 0, 1)) # convert to CHW format
        
        if return_bgr:
            return img_rgb_chw, img_bgr
        else:
            return img_rgb_chw
        

class Plotting:
    """ Class containing methods for plotting truth, predictions, and errors """

    @staticmethod
    def subplot_singlecol_truth_est(
                                        time : np.ndarray
                                        , gt : np.ndarray
                                        , est : np.ndarray
                                        , output_path : str
                                        , plot_num : int
                                        , plt_dict : Optional[Dict[str, Any]] = None # means that the type is either Dict[str, Any] or None
                                ):

        """
        This function plots the ground truth and estimated values for a given set of data within a single subplot

        Inputs:
        time (np.array): the time values, N x 1 array
        gt (np.array): the ground truth values, N x M array
        est (np.array): the estimated values, N x M array
        outputh_path (str): the path to save the plot
        plot_num (int): the plot number
        plt_dict (dict): the dictionary containing the plot details
        """

        # check if the number of components in the ground truth and estimated values are the same
        num_est_components  = est.shape[1]
        num_gt_components   = gt.shape[1]
        num_comps           = num_est_components
        if num_est_components != num_gt_components:
            raise ValueError('The number of estimated and ground truth components must be the same')

        est_timesteps       = est.shape[0]
        gt_timesteps        = gt.shape[0]
        provided_timesteps  = time.shape[0]
        if est_timesteps != gt_timesteps or est_timesteps != provided_timesteps or gt_timesteps != provided_timesteps:
            raise ValueError('The number of timesteps in the ground truth and estimated values must be the same')

        # check if the dictionary is provided 
        if plt_dict is None:
            plt_dict    = {
                            'labels' : ['X', 'Y', 'Z']
                            , 'title' : 'Ground Truth vs. Estimated Translation'
                            , 'legend' : ['Truth', 'Estimate']
                            , 'fig_size' : (10, 8)
                            , 'filename' : 'truth_vs_estimate.png'
                            , 'x_label' : 'Time (seconds)' 
                            , 'y_label' : 'Translation (meters)'
                            , 'gt_style' : '-'
                            , 'est_style' : '--'
                            , 'gt_color' : 'blue'
                            , 'est_color' : 'red'
                        }
            
        # extract the dictionary values if dictionary is provided as input
        labels      = plt_dict['labels']
        title       = plt_dict['title']
        legend      = plt_dict['legend']
        figsize     = plt_dict['fig_size']
        filename    = plt_dict['filename']
        xlabel      = plt_dict['x_label']
        ylabel      = plt_dict['y_label']
        gt_line     = plt_dict['gt_style']
        est_line    = plt_dict['est_style']
        gt_color    = plt_dict['gt_color']
        est_color   = plt_dict['est_color']

        # create output path
        outpath     = os.path.join(output_path, filename)
        # create a color cycle for the plot 
        # color_cycle         = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        # create component cycle for the plot, cycles through the labels
        component_cycle     = itertools.cycle(labels)
        fig     = plt.figure(plot_num, figsize = figsize)
        axes    = fig.subplots(nrows = num_comps, ncols = 1) # create subplots for a single column
        # wrap in a list if only one component to ensure iterability
        if num_comps == 1:
            axes = [axes]

        for ii, ax in enumerate(axes):
            # color       = next(color_cycle)
            component   = next(component_cycle)

            ax.plot(time, gt[:, ii], color = gt_color, linestyle = gt_line , label = f'{legend[0]}')
            ax.plot(time, est[:, ii], color = est_color, linestyle = est_line, label = f'{legend[1]}')


            ax.set_xlabel(f'{xlabel}')
            ax.set_ylabel(f'{component} {ylabel}')
            ax.legend()

        fig.suptitle(title)
        plt.tight_layout(rect = [0, 0, 1, 0.95])
        plt.savefig(outpath)
        plt.show(block = False) 
        # this is a non-blocking call, so the program will continue to run after the plot is displayed

    @staticmethod
    def subplot_singlecol_err_3sig(
                                            time : np.ndarray
                                            , err: np.ndarray
                                            , std : np.ndarray
                                            , output_path : str
                                            , plot_num : int
                                            , plt_dict : Optional[Dict[str, Any]] = None
                                        ):

        """
        This function plots the error in ground truth and estimated values for a given set of data within a single subplot, providing 3sigma error

        Inputs:
        time (np.array): the time values, N x 1 array
        err (np.array): the error values, N x M array, where N is the number of samples and M is the number of components
        std (np.array): the std values, N x M arrays, where N is the number of samples and M is the number of components, 
            M represents the diagonal of the square root of the covariance matrices
        outputh_path (str): the path to save the plot
        plot_num (int): the plot number
        plt_dict (dict): the dictionary containing the plot details
        """

        # check if the number of components in the ground truth and estimated values are the same
        num_est_components  = err.shape[1]
        num_gt_components   = std.shape[1]
        num_comps           = num_est_components
        if num_est_components != num_gt_components:
            raise ValueError('The number of estimated and ground truth components must be the same')
        
        err_timesteps       = err.shape[0]
        std_timesteps       = std.shape[0]
        provided_timesteps  = time.shape[0]
        if err_timesteps != std_timesteps or err_timesteps != provided_timesteps or std_timesteps != provided_timesteps:
            raise ValueError('The number of timesteps in the error and std values must be the same')

        # check if the dictionary is provided 
        if plt_dict is None:
            plt_dict    = {
                            'labels' : ['X', 'Y', 'Z']
                            , 'title' : "Error with Filter's 3$\sigma$ Covariance Bounds"
                            , 'legend' : ['Error', "3$\sigma$"]
                            , 'fig_size' : (10, 8)
                            , 'filename' : 'err_3sigma.png'
                            , 'x_label' : 'Time (seconds)' 
                            , 'y_label' : 'Translation (meters)'
                            , 'err_style' : '--'
                            , 'cov_style' : '-'
                            , 'err_style' : 'blue'
                            , 'cov_style' : 'red'
                        }
            
        # extract the dictionary values if dictionary is provided as input
        labels      = plt_dict['labels']
        title       = plt_dict['title']
        legend      = plt_dict['legend']
        figsize     = plt_dict['fig_size']
        filename    = plt_dict['filename']
        xlabel      = plt_dict['x_label']
        ylabel      = plt_dict['y_label']
        err_line    = plt_dict['err_style']
        cov_line    = plt_dict['cov_style']
        err_color   = plt_dict['err_color']
        cov_color   = plt_dict['cov_color']

        # create output path
        outpath         = os.path.join(output_path, filename)
        # create component cycle for the plot, cycles through the labels
        component_cycle = itertools.cycle(labels)
        fig     = plt.figure(plot_num, figsize = figsize)
        axes    = fig.subplots(nrows = num_comps, ncols = 1) # create subplots for a single column
        # wrap in a list if only one component to ensure iterability
        if num_comps == 1:
            axes = [axes]

        for ii, ax in enumerate(axes):
            # color       = next(color_cycle)
            component   = next(component_cycle)

            ax.plot(time, err[:, ii], color = err_color, linestyle = err_line , label = f'+{legend[0]}')
            ax.plot(time, 3 *std[:, ii], color = cov_color, linestyle = cov_line, label = f'-{legend[1]}')
            ax.plot(time, -3 *std[:, ii], color = cov_color, linestyle = cov_line, label = f'-{legend[1]}')


            ax.set_xlabel(f'{xlabel}')
            ax.set_ylabel(f'{component} {ylabel}')
            ax.legend()

        fig.suptitle(title)
        plt.tight_layout(rect = [0, 0, 1, 0.95])
        plt.savefig(outpath)
        plt.show(block = False) 
        # this is a non-blocking call, so the program will continue to run after the plot is displayed
