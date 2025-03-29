""" 
Module for the MEKF class, MEKF is based on:
1) Real-Time Image-Based Relative Pose Estimation and Filtering for Spacecraft Applications (Kaki et al. 2023)
2) https://github.com/siddarthkaki/pose_terrier
"""

# Imports
import numpy as np
import os, yaml, csv
from datetime import datetime
from scipy.linalg import expm
import pdb 

# Local Imports
from theodwyn.navigations.pose_utils import setup_JAX
jax, jnp    = setup_JAX.setup_safely()
from theodwyn.navigations.pose_utils import custom_Quaternion
from theodwyn.navigations.pose_utils import QuatMath # using sscp_R3()
from theodwyn.navigations.measurement import jax_solve_pose_exmap_reint_parallel
from theodwyn.navigations.measurement import jax_solve_pose_local_reint_parallel
from theodwyn.navigations.measurement import Pose_Direct_LM, Pose_Local_LM

try:
    import theodwyn.navigations.ceres_pose_python as ceres_pose_python
except ImportError:
    print( "[WARNING]: ceres_pose_python file not found in theodwyn.navigations " )

class MEKF_ppt_Dynamics:
    """ 
    Callable functions (static methods) for pose terrier MEKF dynamics, kinematics, and measurement models
    State Components:
    quaterion states: δg = [\delta_{g_x}, \delta_{g_y}, \delta_{g_z}]^T
    angular velocity states: ω = [\omega_x, \omega_y, \omega_z]^T
    angular acceleration states: α = [\alpha_x, \alpha_y, \alpha_z]^T
    position states: x_p  = [x, x_dot, x_ddot, y, y_dot, y_ddot, z, z_dot, z_ddot]^T
    Full state: x = [δg, ω, α, x_p]^T
    """

    @staticmethod
    def F_i_position_dwpa(dt):
        """ 
        Function to calculate the state transition matrix for each component of the position states
        Applies to the position, velocity, and acceleration states: [x, x_dot, x_ddot], (3x3) matrix

        Common constant-acceleration model for position states where p, v, and a are the position, velocity, and acceleration states, respectively
        w is indepent and identically distributed (i.i.d) Gaussian noise with zero mean and covariance Qpos; w ~ N(0, Qpos)

        p_{k+1} = p_k + v_k*dt + 0.5*a_k*dt^2
        v_{k+1} = v_k + a_k*dt
        a_{k+1} = a_k + w_{k}
        
        a_{k+2} = a_{k+1} + w_{k+1}         = a_k + w_{k} + w_{k+1}
        v_{k+2} = v_{k+1} + a_{k+1}*dt      = v_k + a_k*dt + (a_k + w_{k})*dt = v_k + 2*a_k*dt + w_{k}*dt
        p_{k+2} = p_{k+1} + v_{k+1}*dt + 0.5*a_{k+1}*dt^2 = p_k + v_k*dt + 0.5*a_k*dt^2 + v_k*dt + 0.5*(a_k + w_{k})*dt^2 = p_k + 2*v_k*dt + 0.5*a_k*dt^2 + w_{k}*dt^2
        """
        F_i_p   = np.array([
                        [1, dt, .5*(dt**2)],
                        [0, 1, dt],
                        [0, 0, 1]
                            ])
        return F_i_p
    
    @staticmethod
    def F_position_dwpa(dt):
        """ 
        Function to calculate F (discretized state transition matrix) matrix for 3rd order discrete-time Weiner Process Acceleration (DWPA)
        Applies to the position, velocity, and acceleration states for each component: 
        [x, x_dot, x_ddot, y, y_dot, y_ddot, z, z_dot, z_ddot], (9x9) matrix
        """
        Fprime  = MEKF_ppt_Dynamics.F_i_position_dwpa(dt)
        F_pos   = np.kron(np.eye(3), Fprime)
        return F_pos
    
    @staticmethod
    def Q_i_position_dwpa(dt, sigma_v):
        """ 
        Function to help calculate noise model (discretized process noise covariance matrix) for discrete-time Weiner Process Acceleration (DWPA)
        This model describes the process noise for a single component of the state vector, e.g., [x, x_dot, x_ddot], (3x3) matrix
        
        p_{k+1} = p_k + v_k*dt + 0.5*a_k*dt^2
        v_{k+1} = v_k + a_k*dt
        a_{k+1} = a_k + w_{k}

        
        a_{k+2} = a_{k+1} + w_{k+1}         = a_k + w_{k} + w_{k+1}
        v_{k+2} = v_{k+1} + a_{k+1}*dt      = v_k + a_k*dt + (a_k + w_{k})*dt = v_k + 2*a_k*dt + w_{k}*dt
        p_{k+2} = p_{k+1} + v_{k+1}*dt + 0.5*a_{k+1}*dt^2 = p_k + v_k*dt + 0.5*a_k*dt^2 + v_k*dt + 0.5*(a_k + w_{k})*dt^2 = p_k + 2*v_k*dt + 0.5*a_k*dt^2 + w_{k}*dt^2

        Therefore, Γ^T = [0.5*dt^2, dt, 1], and Q_{k}
        x_{k+1} = F_i_position_dwpa*x_k + Γ*w_k
        Assume E[w_k*w_k^T] = σ_v^2
        Q_{k}   = E[Γ*w_k*w_k^T*Γ^T] = σ_v^2 * (Γ * Γ^T)
        """
        Q_i_p   = (sigma_v**2) * np.array([
                            [.25*(dt**4), .5*(dt**3), .5*(dt**2)],
                            [.5*(dt**3), dt**2, dt],
                            [.5*(dt**2), dt, 1]
                            ])
        return Q_i_p
    
    @staticmethod
    def Q_position_dwpa(dt, sigma_v):
        """ 
        Function to calculate Q (discretized process noise covariance matrix) for discrete-time Weiner Process Acceleration (DWPA)
        Applies to the position, velocity, and acceleration states for each component: 
        [x, x_dot, x_ddot, y, y_dot, y_ddot, z, z_dot, z_ddot], (9x9) matrix
        """
        Qprime  = MEKF_ppt_Dynamics.Q_i_position_dwpa(dt, sigma_v)
        Q_pos   = np.kron(np.eye(3), Qprime)
        return Q_pos
    
    @staticmethod
    def A_quat(dt, omega_hat):
        """ Discrete Quaterion Propagation Matrix"""
        Omega       = np.linalg.norm(omega_hat, ord = 2)
        if Omega == 0:
            return np.eye(4)
        omega_unit  = omega_hat / Omega

        omega_4x4           = np.zeros([4, 4])
        omega_4x4[0, 1:]    = -omega_unit
        omega_4x4[1:, 0]    = omega_unit
        omega_4x4[1:, 1:]   = -QuatMath.sscp_R3(omega_hat)
        
        theta       = (1/2) * Omega * dt 
        A_quat      = (np.cos(theta) * np.eye(4)) + (np.sin(theta) * omega_4x4)    
        return A_quat

    @staticmethod
    def H_meas_pose():
        """ 
        Measurement model for pose measurement entering the 18 state pose terrier MEKF
        Pose affects the attitude quaterion and (x, y, z) position states
        """
        I3      = np.eye(3)
        O3x6    = np.zeros([3, 6])
        O3      = np.zeros([3, 3])
        # wrong Hmat, paper has this typo 
        # Hmat    = np.block([
        #                         [ I3, O3x6, O3, O3x6],
        #                         [O3, O3x6, I3, O3x6]
        #                     ]) 
        O1x3    = np.zeros((1, 3))  # 1x3 zero matrix
        O1x6    = np.zeros((1, 6))  # 1x6 zero matrix
        onevec  = np.array([1, 0, 0])  # 1x3 vector [1, 0, 0]

        Hmat    = np.block([
                            [ I3, O3x6, O3, O3x6 ],
                            [ O1x3, O1x6, onevec, O1x3, O1x3 ],
                            [ O1x3, O1x6, O1x3, onevec, O1x3 ],
                            [ O1x3, O1x6, O1x3, O1x3, onevec ] 
                        ])
        return Hmat

    @staticmethod
    def A_angle_states(tau, omega):
        """ Discrete-time Gibbs-style Quaternion, Angular Velocity, and Angular Acceleration Propagation Matrix (9x9); not a state transition matrix """
        I3          = np.eye(3)
        omega_sscp  = QuatMath.sscp_R3(omega)
        A_ang_states            = np.zeros([9, 9])
        A_ang_states[0:3, 0:3]  = -omega_sscp
        A_ang_states[0:3, 3:6]  = I3
        A_ang_states[3:6, 6:]   = I3
        A_ang_states[6:, 6:]    = -(1/tau) * I3
        return A_ang_states

    @staticmethod
    def Q_ang_accel(dt, tau, Q_psd):
        """ Process noise covariance matrix for angular acceleration states """
        Q_alpha   = Q_psd * (1 - np.exp( -2 * (dt/tau) )) * np.eye(3)
        return Q_alpha

    @staticmethod
    def Q_k(sigma_v, tau, Q_alpha, Q_position_dwpa):
        """ 
        Process noise covariance matrix for angular and translation states
        Inclues: attitude, angular velocity, and angular acceleration & position, velocity, and acceleration
        """
        Q           = np.zeros([18, 18])
        Q[6:9, 6:9] = sigma_v**2 * (tau/2) * Q_alpha
        Q[9:, 9:]   = Q_position_dwpa
        return Q

    @staticmethod
    def F_k(dt, A_ang_states, F_position_dwpa):
        """ Full State transition matrix for the 18 state pose terrier MEKF """
        F           = np.zeros([18, 18])
        F[0:9, 0:9] = expm(A_ang_states * dt)
        F[9:, 9:]   = F_position_dwpa
        return F
    
    @staticmethod 
    def R_k(sigma_meas):
        """ Measurement noise covariance matrix for pose measurements, 6x6 matrix """
        R   = sigma_meas**2 * np.eye(6)
        return R
    
    @staticmethod 
    def nls_direct(pose0, az_el, kps_3D, bearing_meas_std_rad, rCamVec, max_iter = 50, num_inits = 10):
        """ Measurement model function to solve nonlinear least squares problem for pose using exponential map for attitude """
        pose0_jax   = jnp.array(pose0)
        # check dimensions of attitude
        tr0_jax     = pose0_jax[:3]
        att0_jax    = pose0_jax[3:]
        att_size    = att0_jax.shape[0]
        if att_size == 4:
            quat0_jax   = QuatMath.jax_quat_to_exp_map(att0_jax)
        elif att_size == 3:
            quat0_jax   = att0_jax
        # else: # will error here if att_size is not 3 or 4
        #     raise ValueError('Invalid attitude size')
        pose0_jax   = jnp.concatenate([tr0_jax, quat0_jax])
        az_el_jax   = jnp.array(az_el)
        kps_3D_jax  = jnp.array(kps_3D)
        rCamVec_jax = jnp.array(rCamVec)
        pose_jax, theta_jax, cost   = jax_solve_pose_exmap_reint_parallel(
                                                                            pose0_jax, 
                                                                            az_el_jax, 
                                                                            kps_3D_jax, 
                                                                            bearing_meas_std_rad, 
                                                                            rCamVec_jax,
                                                                            max_iter, 
                                                                            num_inits
                                                                            )
        jax_problem = Pose_Direct_LM(pose0_jax, az_el_jax, kps_3D_jax, bearing_meas_std_rad, rCamVec_jax)
        cov_jax     = QuatMath.covar_block_switch(jax_problem.compute_cov(theta_jax))
        return pose_jax, cov_jax

    @staticmethod 
    def nls_local(pose0, az_el, kps_3D, bearing_meas_std_rad, rCamVec, max_iter = 50, num_inits = 10):
        """ Measurement model function to solve nonlinear least squares problem for pose using a local quaternion update (tangent plane)"""
        pose0_jax   = jnp.array(pose0)
        az_el_jax   = jnp.array(az_el)
        kps_3D_jax  = jnp.array(kps_3D)
        rCamVec_jax = jnp.array(rCamVec)
        pose_jax, theta_jax, cost   = jax_solve_pose_local_reint_parallel(
                                                                            pose0_jax, 
                                                                            az_el_jax, 
                                                                            kps_3D_jax, 
                                                                            bearing_meas_std_rad, 
                                                                            rCamVec_jax,
                                                                            max_iter,
                                                                            num_inits
                                                                            )
        jax_problem = Pose_Local_LM(pose0_jax, az_el_jax, kps_3D_jax, bearing_meas_std_rad, rCamVec_jax)
        cov_jax     = QuatMath.covar_block_switch(jax_problem.compute_cov(theta_jax))
        return pose_jax, cov_jax
    
    @staticmethod
    def nls_cpp(pose0, az_el, kps_3D, bearing_meas_std_rad, rCamVec, num_inits = 10):
        """ Measurement model function to solve nonlinear least squares problem for pose using C++ ceres solver """
        num_kps = kps_3D.shape[0]
        yVec    = az_el.reshape(2*num_kps, 1)
        pos0    = pose0[:3].reshape(3, 1)
        quat0   = pose0[3:].reshape(4, 1)
        rCamVec = rCamVec.reshape(3, 1)
        rFeaMat = kps_3D
        cs      = ceres_pose_python.PoseSolver.SolvePoseReinitParallel(pos0, quat0, yVec, rCamVec, rFeaMat, bearing_meas_std_rad, num_inits) 
        pos     = cs.pose.pos
        quat    = cs.pose.quat
        pose    = np.concatenate([pos, quat])
        covar   = cs.cov_pose
        return pose, covar


class MEKF_ppt:
    """
    Class for the Multiplicative Extended Kalman Filter (MEKF) algorithm
    """
    def __init__(self, dt, process_noise_std, measurement_noise_std, max_flip_deg = 45, tau = 1, Qpsd = 25):
        """ 
        Initialize the MEKF object 

        Input:
            dt (float): time step
            process_noise_std (float): standard deviation of the Weiner Process Acceleration (WPA) noise
            measurement_noise_std (float): standard deviation of the measurement noise
            max_flip_deg (float): maximum flip angle in degrees
            tau (float): time constant for discrete-time first-order Gauss–Markov process
            Qpsd (float): is the power spectral density of each element of the underlying continuous-time model of the discrete noise term
        """
        self.dt         = dt
        self.sig_v      = process_noise_std
        self.tau        = tau
        self.Qpsd       = Qpsd
        self.sig_meas   = measurement_noise_std
        self.max_flip   = max_flip_deg
        self.I3         = np.eye(3)
        self.I4         = np.eye(4)
        self.O3         = np.zeros([3, 3])
        self.O3x6       = np.zeros([3, 6])


        self.F_pos      = MEKF_ppt_Dynamics.F_position_dwpa(self.dt)
        self.Q_pos      = MEKF_ppt_Dynamics.Q_position_dwpa(self.dt, self.sig_v)
        self.Q_alpha    = MEKF_ppt_Dynamics.Q_ang_accel(self.dt, self.tau, self.Qpsd)
        self.H          = MEKF_ppt_Dynamics.H_meas_pose()
        self.Q_k        = MEKF_ppt_Dynamics.Q_k(self.sig_v, self.tau, self.Q_alpha, self.Q_pos)
        self.R_k        = MEKF_ppt_Dynamics.R_k(self.sig_meas) # initial R_k
        
        self.num_states         = self.Q_k.shape[0]
        self.num_pos_states     = self.F_pos.shape[0]
        self.num_att_states     = 9
        # self.num_att_states     = self.F_pos.shape[0]
        self.num_meas_states    = self.H.shape[0]
        
        self.gidx_start         = 0
        self.gidx_end           = 3
        self.gidx               = np.arange(0, 3) # gibbs vector index
        self.oidx               = np.arange(3, 6) # angular velocity index
        # self.pidx               = np.arange(self.num_att_states, self.num_att_states + 3) # position index
        # self.ngidx              = tuple(np.arange(3, self.num_states)) # index for state except delta gibbs vector
        self.ngidx              = np.arange(3, self.num_states) # index for state except delta gibbs vector

        self.idx_pos_x          = 9     # Indexing for position state x
        self.idx_pos_y          = 12    # Indexing for position state y
        self.idx_pos_z          = 15    # Indexing for position state z

        self.idx_pos_x_no_gvec  = 6     # Indexing for position state x when excluding delta gibbs vector
        self.idx_pos_y_no_gvec  = 9     # Indexing for position state y when excluding delta gibbs vector
        self.idx_pos_z_no_gvec  = 12    # Indexing for position state z when excluding delta gibbs vector


        self.position_est       = np.zeros(3)
        self.omega_est          = np.zeros(3)
        self.global_quat_est    = custom_Quaternion().npy # identity quaternion
        self.delta_gibbs_est    = np.zeros(3)
        self.state_est          = np.zeros(self.num_states)
        self.covar_est          = np.zeros([self.num_states, self.num_states])
        self.processed_meas     = False
        self.current_time       = 0.0

        
    def set_initial_state_covar(self, quat0, omega0, alpha0, tr0, covar0):
        """ Function to set the initial state and covariance estimates """
        
        self.position_est       = tr0 # position (x, y, z)
        self.omega_est          = omega0
        self.global_quat_est    = quat0
        self.state_est[3:6]     = omega0
        self.state_est[6:9]     = alpha0

        ## Edited and indexed by Edward. You can change the indexing if you want. Anand.
        self.state_est[self.idx_pos_x]      = tr0[0] 
        self.state_est[self.idx_pos_y]      = tr0[1]
        self.state_est[self.idx_pos_z]      = tr0[2]

        self.covar_est          = covar0


    def time_update(self):
        """ Function to update the state estimate using the dynamics model """    

        omega_est_k     = self.omega_est
        # omega_est_k     = self.state_est[3:6]
        quat_est_k      = self.global_quat_est
        state_est_k     = self.state_est
        covar_est_k     = self.covar_est

        # grab Q_k, F_k
        Q_k             = self.Q_k
        A_att_k         = MEKF_ppt_Dynamics.A_angle_states(self.tau, omega_est_k)     
        F_k             = MEKF_ppt_Dynamics.F_k(self.dt, A_att_k, self.F_pos)

        # propagate the quaternion estimate
        A_k             = MEKF_ppt_Dynamics.A_quat(self.dt, omega_est_k)
        quat_est_kp1    = A_k @ quat_est_k
        quat_est_kp1    = QuatMath.q_norm(quat_est_kp1)

        # propagate covariance estimate
        P_bar_kp1                   = F_k @ covar_est_k @ F_k.T + Q_k
        state_est_bar_kp1_no_gvec   = F_k[np.ix_(self.ngidx, self.ngidx)] @ state_est_k[self.ngidx]


        # update object states
        self.omega_est              = state_est_bar_kp1_no_gvec[0:3] # omega_est_bar_kp1
        self.position_est           = np.array([state_est_bar_kp1_no_gvec[self.idx_pos_x_no_gvec], state_est_bar_kp1_no_gvec[self.idx_pos_y_no_gvec], state_est_bar_kp1_no_gvec[self.idx_pos_z_no_gvec]])# position_est_bar_kp1
        self.global_quat_est        = quat_est_kp1 # global quaternion estimate after discrete-time propagation
        self.state_est[self.ngidx]  = state_est_bar_kp1_no_gvec # update state estiamte after discrete-time propagation, exclude delta gibbs state
        self.covar_est              = P_bar_kp1 # covariance estimate
        self.current_time           += self.dt

    def measurement_update(self, quat_meas, position_meas, R_k = None):
        """ Function to update the state estimate using the measurement model """
        
        omega_est_bar_kp1   = self.omega_est
        quat_est_bar_kp1    = self.global_quat_est
        state_est_bar_kp1   = self.state_est
        covar_est_bar_kp1   = self.covar_est

        # grab H_k, R_k
        H_k     = self.H 
        if R_k is None:
            R_k = self.R_k

        # covariance update
        S_kp1   = H_k @ covar_est_bar_kp1 @ H_k.T + R_k # innovation covariance, Pinn in Kaki 2023 paper
        K_kp1   = covar_est_bar_kp1 @ H_k.T @ np.linalg.inv(S_kp1) # Kalman gain
        I       = np.eye(self.num_states)
        P_kp1   = (I - K_kp1 @ H_k) @ covar_est_bar_kp1 @ (I - K_kp1 @ H_k).T + K_kp1 @ R_k @ K_kp1.T # updated covariance estimate, Joseph form

        # measurement update
        delta_quat          = QuatMath.q_error_shu(quat_meas, quat_est_bar_kp1)
        attitude_innovation = 2 * QuatMath.quat2gibbs(delta_quat)
        position_innovation = position_meas - self.position_est
        meas_innovation     = np.concatenate([attitude_innovation, position_innovation])
        
        delta_x                 = K_kp1 @ meas_innovation
        delta_gibbs_est_kp1     = delta_x[0:3]
        self.delta_gibbs_est    = delta_gibbs_est_kp1

        # reset 
        delta_quat_temp     = custom_Quaternion().npy
        ########## check this ##########
        delta_quat_temp[1:] = .5 * delta_gibbs_est_kp1
        # could use exponential map to ensure unit quaternion but may be more expensive
        # delta_quat_temp     = QuatMath.q_norm(delta_quat_temp) # commented out, generally not needed to normalize at this step
        quat_est_kp1        = QuatMath.qmult_shu(delta_quat_temp, quat_est_bar_kp1)
        ########## check this ##########

        # robustness checks 
        mean_attitude_inn   = np.mean(attitude_innovation)
        mean_position_inn   = np.mean(position_innovation)
        attiude_inn_std     = np.sqrt( np.trace(S_kp1[0:3, 0:3]) / 3 )
        position_inn_std    = np.sqrt( np.trace(S_kp1[3:, 3:]) / 3 )
        delta_quat          = QuatMath.q_error_shu(quat_est_bar_kp1, quat_est_kp1)
        dangle_rad          = 2.0 * np.arccos(delta_quat[0])
        dangle_deg          = np.degrees(dangle_rad)

        # robustness logic, whether to reject measurement or not 
        if np.abs(mean_attitude_inn) > 3.0 * attiude_inn_std:
            print(f'Rejected measurement due to attitude innovation: | {np.degrees(mean_attitude_inn)} | > 3 * {np.degrees(attiude_inn_std)}')
        elif abs(dangle_deg) > self.max_flip:
            print(f'Rejected measurement due to attitude flip: | {dangle_deg} | > {self.max_flip}')
        elif np.abs(mean_position_inn) > 3.0 * position_inn_std:
            print(f'Rejected measurement due to position innovation: | {mean_position_inn} | > 3 * {position_inn_std}')
        else:
            state_est_kp1   = state_est_bar_kp1 + delta_x

            # update object states
            self.position_est           = np.array([state_est_kp1[self.idx_pos_x], state_est_kp1[self.idx_pos_y], state_est_kp1[self.idx_pos_z]])# position_est_bar_kp1
            self.omega_est              = state_est_kp1[self.oidx] # omega_est_bar_kp1
            self.global_quat_est        = quat_est_kp1 # global quaternion estimate after discrete-time propagation
            self.state_est              = state_est_kp1 # update state estimate after discrete-time propagation, exclude delta Gibbs state
            self.covar_est              = P_kp1 # covariance estimate
            self.processed_meas         = True

    def mekf_reset(self):
        """ Function to reset the MEKF object """
        self.delta_gibbs_est        = np.zeros(3)
        self.state_est[self.gidx]   = np.zeros(3)
        self.processed_meas         = True

    def write_config_yaml(self, yaml_file_fn = 'mekf_config.yaml'):
        """ Function to write the MEKF initialization configuration to a yaml file; typically called once """
    
        config_data = {
                        'dt'                    : float(self.dt),
                        'process_noise_std'     : float(self.sig_v),
                        'measurement_noise_std' : float(self.sig_meas),
                        'max_flip_deg'          : float(self.max_flip),
                        'tau'                   : float(self.tau),
                        'Qpsd'                  : float(self.Qpsd),
                        'initial_position'      : self.position_est.tolist(),
                        'initial_omega'         : self.omega_est.tolist(),
                        'initial_quaternion'    : self.global_quat_est.tolist(),
                        'initial_state_est'     : self.state_est.tolist(),
                        'initial_covar_diag'    : np.diag( self.covar_est ).tolist()
                    }

        with open(yaml_file_fn, 'w') as yf:
            yaml.dump(config_data, yf, default_flow_style = False)
        print(f'Wrote MEKF configuration to {yaml_file_fn}')

    def log_mekf_est_csv(self, csv_file = 'mekf_estimates.csv'):
        """ Function to log the MEKF estimates and covariances to a csv file """

        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                header  = [
                                'time'
                                    , 'position_x', 'position_y', 'position_z'
                                , 'omega_x', 'omega_y', 'omega_z'
                                , 'quat_w', 'quat_x', 'quat_y', 'quat_z'
                            ]
                writer.writerow(header)
