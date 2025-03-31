""" Classes and functions for calculating measurements keypoint regression data """

# Imports
from scipy.optimize import least_squares, minimize
import pdb

# Local Imports
from theodwyn.navigations.pose_utils import setup_JAX
jax, jnp    = setup_JAX.setup_safely()
from theodwyn.navigations.pose_utils import QuatMath
from theodwyn.navigations.pose_utils import TorchQuatMath

# Note: calls to fcns that contain .jax here will ultimately fail b/c jax is a dummy module when it is not installed

# # For debugging
# jax.config.update('jax_debug_nans', True)
# jax.config.update('jax_debug_infs', True)

###################### Functions that will use jaxopt to directly solve pose via exponential map parameters #######################
@jax.jit
def jax_exmap_residuals_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam = jnp.array([0.0, 0.0, 0.0])): 
    """ 
    Computes residuals for Bearing measurements with a exponential map representation of the quaternion

    Inputs:
    N is the number of keypoints
    pose0 (jnp.array): (6,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
        resid_weighted (jnp.array): (2*N,) array representing the weighted residuals for the bearing measurements of form
        [resid_az_1, resid_el_1, resid_az_2, resid_el_2, ..., resid_az_N, resid_el_N]


    for SolvePose(pos0, quat0, yVec, rCamVec, rFeaMat, bearing_mes_std) from pose_terrier (not MeasResidCostFunctorQuat())
    pos0 is translation portion of pose0
    quat0 is quaternion portion of pose0
    yVec is az_el_2D but need to reshape to (2*N, 1) with alternating azimuth and elevation angles
    rCamVec is rvec_cam but need to reshape to (3, 1)
    rFeaMat is kps_3D (no reshaping is needed)
    bearing_meas_std is bearing_meas_std

    Src: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf w/ journal paper at : https://www.tandfonline.com/doi/epdf/10.1080/10867651.1998.10487493?needAccess=true
    """
    
    # bearing_meas_std_rad    = float( bearing_meas_std * (jnp.pi/180) )
    # extract position and quaternion from pose0
    rvec    = pose0[:3] # (3,), relative position vector from chaser to target body-fixed frame origins in chaser body-fixed frame
    rtang   = pose0[3:] # (3,), exponential map parameters representing attitude transformation from chaser to target body-fixed frame
    
    quat_c2t= QuatMath.jax_qexp_map(rtang) # quaterion from the exponential map parameters, representing attitude transformation from chaser to target body-fixed frame
    # quat_c2t= QuatMath.robust_jax_qexp_map(rtang)
    quat_t2c= QuatMath.jax_q_conj(quat_c2t) # quaternion representing attitude transformation from target to chaser body-fixed frame
    trfMat  = QuatMath.jax_q2trfm(quat_t2c) # maybe move outside of the function?
    
    # transform keypoints from target frame to "camera" frame
    # rvec - rvec_cam => shift from chaser origin to camera, then
    # plus transformation (passive rotation) * the 3D keypoints
    # shape after multiplication: (3, N)
    r_c_i       = (rvec - rvec_cam)[:, None] + trfMat @ kps_3D.T
    pred_az     = jnp.arctan2(r_c_i[0], r_c_i[2])
    pred_el     = jnp.arctan2(r_c_i[1], r_c_i[2])
    # compute residual in azimuth
    resid_az    = az_el_2D[:, 0] - pred_az
    # compute residual in elevation
    resid_el    = az_el_2D[:, 1] - pred_el
    # concatenate into a single 1D array of size 2*N with alternating azimuth and elevation residuals
    resid       = jnp.column_stack([resid_az, resid_el]).ravel()

    # weight residuals by inverse of measurement standard deviation
    # not squared because we are using the residuals for optimization
    resid_weighted  = resid / bearing_meas_std_rad
    return resid_weighted

@jax.jit
def jax_exmap_cost_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """ 
    Computes cost, sum of squares of weighted residual, for Bearing measurements with a exponential map representation of the quaternion

    Inputs:
    N is the number of keypoints
    pose0 (jnp.array): (6,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
        cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    resid_weighted  = jax_exmap_residuals_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    cost            = jnp.sum(resid_weighted**2)
    return cost

@jax.jit
def cost_and_grad(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """ Computes the cost and gradient for the cost function and has been jitted for optimization """
    cost_val, grad_val  = jax.value_and_grad(jax_exmap_cost_unconstrained)(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return cost_val, grad_val

class Pose_Direct_LM:
    def __init__(self, pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
        self.pose0              = pose
        self.az_el_2D           = az_el_2D
        self.kps_3D             = kps_3D
        self.bearing_meas_std   = bearing_meas_std_rad
        self.rvec_cam           = rvec_cam
        self._cost_and_grad     = jax.value_and_grad(self.cost_fun)
        self._jac_fn            = jax.jacfwd(self.residual_fun)
    
    def residual_fun(self, pose):
        """ Compute the measurement residuals """
        resid   = jax_exmap_residuals_unconstrained(pose, self.az_el_2D, self.kps_3D, self.bearing_meas_std, self.rvec_cam)
        return resid

    def cost_fun(self, pose):
        """ Compute the cost function for the optimization """
        resid   = self.residual_fun(pose)
        cost    = jnp.sum(resid**2)
        return cost

    def compute_jacobian(self, pose):
        """ Compute the Jacobian for the cost function """
        return self._jac_fn(pose)
    
    def compute_gradient(self, pose):
        """ Compute the gradient for the cost function """
        cost, grad  = self._cost_and_grad(pose)
        return cost, grad
    
    def compute_cov(self, pose):
        """ Compute the covariance matrix for the pose optimization """
        Jac     = self.compute_jacobian(pose)
        Hessian = Jac.T @ Jac
        cov     = jnp.linalg.inv(Hessian)
        return cov

def solve_pose_exmap_jaxopt_lm(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter = 50):
    """
    Solves for the relative pose between the chaser and target body-fixed frames using the jaxopt Levenberg-Marquardt optimizer with a local tangent update (local parameterization)

    Inputs:
    pose0 (jnp.array): (6,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
    max_iter (int): maximum number of iterations for the optimization

    Outputs: 
    final_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    final_6pose (jnp.array): (6,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    final_cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    import jaxopt

    problem         = Pose_Direct_LM(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    solver          = jaxopt.LevenbergMarquardt(residual_fun  = problem.residual_fun, maxiter = max_iter)
    result_obj      = solver.run(pose0)
    final_6pose     = result_obj.params
    final_quat      = QuatMath.jax_qexp_map(final_6pose[3:])
    final_pose      = jnp.concatenate([final_6pose[:3], final_quat])
    final_cost      = jax_exmap_cost_unconstrained(final_6pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return final_pose, final_6pose, final_cost


def jax_solve_pose_exmap_reint_parallel(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter = 50, n_init = 10):
    """
    Optimizes the relative pose between the chaser and target body-fixed frames using multiple random re-initializations in parallel

    Inputs:
    pose0 (jnp.array): (6,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
    learning_rate (float): step size for the gradient descent optimization
    max_iter (int): maximum number of iterations for the optimization
    n_init (int): number of re-initializations

    Outputs:
    best_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    best_pose_theta (jnp.array): (6,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
    best_cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    keys    = jax.random.split(jax.random.PRNGKey(0), n_init)

    def init_guess(i, key):
        rtang0      = jnp.where(i == 0, pose0[3:], QuatMath.jax_quat_to_exp_map(QuatMath.jax_random_quat(key)))
        new_pose    = jnp.concatenate([pose0[:3], rtang0])
        return new_pose
        
    poses_init      = jnp.stack([init_guess(i, key) for i, key in enumerate(keys)], axis = 0)
    opt_vmap        = jax.jit( jax.vmap(lambda p0: solve_pose_exmap_jaxopt_lm(p0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter)) ) 
    poses, poses_theta, costs   = opt_vmap(poses_init)
    best_idx        = jnp.argmin(costs)
    best_pose       = poses[best_idx]
    best_pose_theta = poses_theta[best_idx]
    best_cost       = costs[best_idx]
    return best_pose, best_pose_theta, best_cost
###################### Functions that will use jaxopt to directly solve pose via exponential map parameters #######################




################################### Functions that will use jaxopt to do a local tangent update ###################################
@jax.jit
def jax_pose_residuals_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam = jnp.array([0.0, 0.0, 0.0])): 
    """ 
    Computes residuals for Bearing measurements with with pose vector consisting of translation and quaternion

    Inputs:
    N is the number of keypoints
    pose0 (jnp.array): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
        resid_weighted (jnp.array): (2*N,) array representing the weighted residuals for the bearing measurements of form
        [resid_az_1, resid_el_1, resid_az_2, resid_el_2, ..., resid_az_N, resid_el_N]


    for SolvePose(pos0, quat0, yVec, rCamVec, rFeaMat, bearing_mes_std) from pose_terrier (not MeasResidCostFunctorQuat())
    pos0 is translation portion of pose0
    quat0 is quaternion portion of pose0
    yVec is az_el_2D but need to reshape to (2*N, 1) with alternating azimuth and elevation angles
    rCamVec is rvec_cam but need to reshape to (3, 1)
    rFeaMat is kps_3D (no reshaping is needed)
    bearing_meas_std is bearing_meas_std

    Src: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf w/ journal paper at : https://www.tandfonline.com/doi/epdf/10.1080/10867651.1998.10487493?needAccess=true
    """
    eps     = 0
    # eps     = 1e-8 # small value to avoid division by zero
    # bearing_meas_std_rad    = float( bearing_meas_std * (jnp.pi/180) )
    # extract position and quaternion from pose0
    rvec        = pose0[:3] # (3,), relative position vector from chaser to target body-fixed frame origins in chaser body-fixed frame
    quat_c2t    = pose0[3:] # (4,), quaternion representing attitude transformation from chaser to target body-fixed frame
    quat_t2c    = QuatMath.jax_q_conj(quat_c2t) # quaternion representing attitude transformation from target to chaser body-fixed frame
    
    # trfMat  = QuatMath.jax_q2trfm(quat_c2t).T # same 
    trfMat      = QuatMath.jax_q2trfm(quat_t2c) 

    # transform keypoints from target frame to "camera" frame
    # rvec - rvec_cam => shift from chaser origin to camera, then
    # plus transformation (passive rotation) * the 3D keypoints
    # shape after multiplication: (3, N)
    r_c_i       = (rvec - rvec_cam)[:, None] + trfMat @ kps_3D.T
    pred_az     = jnp.arctan2(r_c_i[0], r_c_i[2] + eps)
    pred_el     = jnp.arctan2(r_c_i[1], r_c_i[2] + eps)
    # compute residual in azimuth
    resid_az    = az_el_2D[:, 0] - pred_az
    # compute residual in elevation
    resid_el    = az_el_2D[:, 1] - pred_el
    # concatenate into a single 1D array of size 2*N with alternating azimuth and elevation residuals
    resid       = jnp.column_stack([resid_az, resid_el]).ravel()

    # weight residuals by inverse of measurement standard deviation
    # not squared because we are using the residuals for optimization
    resid_weighted  = resid / bearing_meas_std_rad
    return resid_weighted

@jax.jit
def jax_pose_cost_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """
    Computes cost for residuals for Bearing measurements with with pose vector consisting of translation and quaternion

    Inputs:
    N is the number of keypoints
    pose0 (jnp.array): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
        cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    resid_weighted  = jax_pose_residuals_unconstrained(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    cost            = jnp.sum(resid_weighted**2)
    return cost

@jax.jit
def jax_pose_tangent_update(pose_7, delta_6, max_theta = jnp.pi):
    """
    Updates the pose using the tangent update delta (a 6-vector)

    Inputs:
    pose_7 (jnp.array): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z] 
    delta_6 (jnp.array): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    max_theta (float): maximum angle for the rotation update, only pi b/c we are using the exponential map

    Outputs:
    new_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    """
    eps         = 1e-8
    tr_base     = pose_7[:3]
    quat_base   = pose_7[3:]
    delta_tr    = delta_6[:3]
    delta_theta = delta_6[3:]
    angle       = jnp.linalg.norm(delta_theta)
    # jax.debug.print("    angle of delta_theta= {}", angle)
    safe_angle  = jnp.where(angle < eps, eps, angle)
    scaled_dt   = delta_theta * (max_theta/safe_angle)
    delta_theta = jnp.where(angle > max_theta, scaled_dt, delta_theta)
    new_t       = tr_base + delta_tr
    q_delta     = QuatMath.jax_qexp_map(delta_theta)
    # q_delta     = QuatMath.robust_jax_qexp_map(delta_theta)
    # jax.debug.print("    q_delta= {}", q_delta)
    new_q       = QuatMath.jax_qmulut_ham(q_delta, quat_base)
    new_pose    = jnp.concatenate([new_t, new_q])
    return new_pose

@jax.jit
def jax_pose_local_update_residual(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """
    Calculates the residuals for the bearing measurements after a local update to the pose

    Inputs:
    pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    del_pose (jnp.array): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
    resid (jnp.array): (2*N,) array representing the weighted residuals for the bearing measurements
    """
    new_pose    = jax_pose_tangent_update(pose0, del_pose)
    resid       = jax_pose_residuals_unconstrained(new_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return resid

@jax.jit
def jax_pose_covariance(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """
    Computes the covariance matrix for the pose optimization

    Inputs:
    pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    del_pose (jnp.array): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
    cov6x6 (jnp.array): (6, 6) array representing the covariance matrix for the pose optimization
    """ 

    def residuals_local(delta):
        new_pose    = jax_pose_tangent_update(pose0, delta)
        resid       = jax_pose_residuals_unconstrained(new_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
        return resid
    
    J_fn            = jax.jacfwd(residuals_local)
    # resid_final     = residuals_local(del_pose)
    J_final         = J_fn(del_pose)
    Hessian         = J_final.T @ J_final
    cov6x6          = jnp.linalg.inv(Hessian) # Fisher Information Matrix

    return cov6x6

class Pose_Local_LM:
    """
    Holds the 'base pose' and data, and defines a residual_fun for jaxopt; the solver's parameters are the local 6D tangent increments.
    """
    def __init__(self, pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
        self.base_pose              = pose0
        self.az_el_2D               = az_el_2D
        self.kps_3D                 = kps_3D
        self.bearing_meas_std_rad   = bearing_meas_std_rad
        self.rvec_cam               = rvec_cam
        
        # build the cost function and its gradient
        self._cost_and_grad         = jax.value_and_grad(self.cost_fun)
        # build jitted Jacobian function
        self._jac_fn                = jax.jacfwd(self.residual_fun)

    def residual_fun(self, del_pose):
        """ Lift the 6D tangent update into a full 7D pose on the quaternion manifold and compute the measurement residual """
        new_pose    = jax_pose_tangent_update(self.base_pose, del_pose)
        # # DEBUG: print or check intermediate values
        # jax.debug.print("Inside residual_fun:")
        # jax.debug.print("  base_pose = {}", self.base_pose)
        # jax.debug.print("  del_pose  = {}", del_pose)
        # jax.debug.print("  new_pose  = {}", new_pose)
        resid       = jax_pose_residuals_unconstrained(new_pose, self.az_el_2D, self.kps_3D, self.bearing_meas_std_rad, self.rvec_cam)
        return resid
    
    def cost_fun(self, del_pose):
        """ Compute the cost function for the optimization """
        resid       = self.residual_fun(del_pose)
        cost        = jnp.sum(resid**2)
        return cost
    
    def compute_gradient(self, del_pose):
        """ Compute the gradient for the cost function """
        cost, grad  = self._cost_and_grad(del_pose)
        return cost, grad
    
    def compute_Jacobian(self, del_pose):
        """ Compute the Jacobian for the cost function """
        return self._jac_fn(del_pose)
    
    def compute_cov(self, del_pose):
        """ Compute the covariance matrix for the pose optimization """
        Jac     = self.compute_Jacobian(del_pose)
        Hessian = Jac.T @ Jac
        cov     = jnp.linalg.inv(Hessian)
        return cov


def solve_pose_local_jaxopt_lm(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter = 50):
    """
    Solves for the relative pose between the chaser and target body-fixed frames using the jaxopt Levenberg-Marquardt optimizer with a local tangent update (local parameterization)

    Inputs:
    pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
    max_iter (int): maximum number of iterations for the optimization

    Outputs: 
    final_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    final_cost (float): sum of squares of weighted residuals for the bearing measurements
    num_iters (int): number of iterations taken by the solver
    """
    import jaxopt 

    init_delta      = 1e-6 * jnp.ones(6)  # need to give this a non-zero initial guess in order for the solver to work
    problem         = Pose_Local_LM(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    # sc1             = problem.compute_gradient(init_delta)
    # sc2             = problem.compute_Jacobian(init_delta)
    solver          = jaxopt.LevenbergMarquardt(residual_fun  = problem.residual_fun, maxiter = max_iter)
    result_obj      = solver.run(init_delta)
    final_del_pose  = result_obj.params
    # num_iters       = result_obj.state.iter_num.item()
    final_pose      = jax_pose_tangent_update(pose0, final_del_pose)
    final_cost      = jax_pose_cost_unconstrained(final_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    # sc3a            = problem.compute_gradient(final_del_pose)
    # sc3b            = sc3a - result_obj.state.gradient
    # sc4a            = problem.compute_Jacobian(final_del_pose)
    # sc4b            = jnp.linalg.inv(sc4a.T @ sc4a )
    # sc5             = problem.residual_fun(final_del_pose) - result_obj.state.residual
    # pdb.set_trace()
    return final_pose, final_del_pose, final_cost

def jax_solve_pose_local_reint_parallel(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter = 50, n_init = 10):
    """
    Optimizes the relative pose between the chaser and target body-fixed frames using multiple random re-initializations in parallel

    Inputs:
        pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
        az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
        kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
        bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
        rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
        learning_rate (float): step size for the gradient descent optimization
        max_iter (int): maximum number of iterations for the optimization
        n_init (int): number of re-initializations

    Outputs:
        optimized_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]

    """
    keys    = jax.random.split(jax.random.PRNGKey(0), n_init)

    def init_guess(i, key):
        quat0       = jnp.where(i == 0, pose0[3:], QuatMath.jax_random_quat(key))
        new_pose    = jnp.concatenate([pose0[:3], quat0])
        return new_pose
    
    poses_init      = jnp.stack([init_guess(i, key) for i, key in enumerate(keys)], axis = 0)

    opt_vmap        = jax.jit( jax.vmap(lambda p0: solve_pose_local_jaxopt_lm(p0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter)) ) 
    poses, del_poses, costs = opt_vmap(poses_init)
    best_idx        = jnp.argmin(costs)
    best_pose       = poses[best_idx]
    best_del_pose   = del_poses[best_idx]
    best_cost       = costs[best_idx]
    # best_iteration  = iterations[best_idx]
    # best_resid_sc   = jax_pose_residuals_unconstrained(best_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    # best_cost_sc    = jax_pose_cost_unconstrained(best_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    # best_pose_sc    = jax_pose_tangent_update(pose0, best_del_pose)
    # best_quat_sc    = best_pose_sc[3:]
    # pdb.set_trace()
    return best_pose, best_del_pose, best_cost

################################### Functions that will use jaxopt to do a local tangent update ###################################
#### NOTE TO SELF: Implement dense SVD for the Jacobian matrix to get the covariance matrix


############################# Streamlined Functions that will use jaxopt to do a local tangent update #############################
@jax.jit
def jax_stream_pose_resids_cost(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam = jnp.array([0.0, 0.0, 0.0])): 
    """ 
    Computes residuals and costs for Bearing measurements with with pose vector consisting of translation and quaternion

    Inputs:
    N is the number of keypoints
    pose0 (jnp.array): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
    resid_weighted (jnp.array): (2*N,) array representing the weighted residuals for the bearing measurements of form
        [resid_az_1, resid_el_1, resid_az_2, resid_el_2, ..., resid_az_N, resid_el_N]
    cost (float): sum of squares of weighted residuals for the bearing measurements



    for SolvePose(pos0, quat0, yVec, rCamVec, rFeaMat, bearing_mes_std) from pose_terrier (not MeasResidCostFunctorQuat())
    pos0 is translation portion of pose0
    quat0 is quaternion portion of pose0
    yVec is az_el_2D but need to reshape to (2*N, 1) with alternating azimuth and elevation angles
    rCamVec is rvec_cam but need to reshape to (3, 1)
    rFeaMat is kps_3D (no reshaping is needed)
    bearing_meas_std is bearing_meas_std

    Src: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf w/ journal paper at : https://www.tandfonline.com/doi/epdf/10.1080/10867651.1998.10487493?needAccess=true
    """
    # eps         = 0
    eps         = 1e-8 # small value to avoid division by zero
    # extract position and quaternion from pose0
    rvec        = pose0[:3] # (3,), relative position vector from chaser to target body-fixed frame origins in chaser body-fixed frame
    quat_c2t    = pose0[3:] # (4,), quaternion representing attitude transformation from chaser to target body-fixed frame
    quat_t2c    = QuatMath.jax_q_conj(quat_c2t) # quaternion representing attitude transformation from target to chaser body-fixed frame
    
    # trfMat  = QuatMath.jax_q2trfm(quat_c2t).T # same 
    trfMat      = QuatMath.jax_q2trfm(quat_t2c) 

    # transform keypoints from target frame to "camera" frame
    # rvec - rvec_cam => shift from chaser origin to camera, then
    # plus transformation (passive rotation) * the 3D keypoints
    # shape after multiplication: (3, N)
    r_c_i       = (rvec - rvec_cam)[:, None] + trfMat @ kps_3D.T
    pred_az     = jnp.arctan2(r_c_i[0], r_c_i[2] + eps)
    pred_el     = jnp.arctan2(r_c_i[1], r_c_i[2] + eps)
    # compute residual in azimuth
    resid_az    = az_el_2D[:, 0] - pred_az
    # compute residual in elevation
    resid_el    = az_el_2D[:, 1] - pred_el
    # concatenate into a single 1D array of size 2*N with alternating azimuth and elevation residuals
    resid       = jnp.column_stack([resid_az, resid_el]).ravel()

    # weight residuals by inverse of measurement standard deviation
    # not squared because we are using the residuals for optimization
    resid_weighted  = resid / bearing_meas_std_rad
    cost            = jnp.sum(resid_weighted**2)
    return resid_weighted, cost

@jax.jit
def jax_stream_pose_tangent_update(pose_7, delta_6, max_theta = jnp.pi):
    """
    Updates the pose using the tangent update delta (a 6-vector)

    Inputs:
    pose_7 (jnp.array): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z] 
    delta_6 (jnp.array): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    max_theta (float): maximum angle for the rotation update, only pi b/c we are using the exponential map

    Outputs:
    new_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    """
    eps         = 1e-8
    tr_base     = pose_7[:3]
    quat_base   = pose_7[3:]
    delta_tr    = delta_6[:3]
    delta_theta = delta_6[3:]
    angle       = jnp.linalg.norm(delta_theta)
    # jax.debug.print("    angle of delta_theta= {}", angle)
    safe_angle  = jnp.where(angle < eps, eps, angle)
    scaled_dt   = delta_theta * (max_theta/safe_angle)
    delta_theta = jnp.where(angle > max_theta, scaled_dt, delta_theta)
    new_t       = tr_base + delta_tr
    q_delta     = QuatMath.jax_qexp_map(delta_theta)
    # q_delta     = QuatMath.robust_jax_qexp_map(delta_theta)
    # jax.debug.print("    q_delta= {}", q_delta)
    new_q       = QuatMath.jax_qmulut_ham(q_delta, quat_base)
    new_pose    = jnp.concatenate([new_t, new_q])
    return new_pose

@jax.jit
def jax_stream_pose_local_update_residual(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
    """
    Calculates the residuals for the bearing measurements after a local update to the pose

    Inputs:
    pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    del_pose (jnp.array): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
    resid (jnp.array): (2*N,) array representing the weighted residuals for the bearing measurements
    """
    new_pose    = jax_stream_pose_tangent_update(pose0, del_pose)
    resid, _    = jax_stream_pose_resids_cost(new_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return resid

def stream_solve_pose_local_jaxopt_lm(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_iter = 50):
    """
    Solves for the relative pose between the chaser and target body-fixed frames using the jaxopt Levenberg-Marquardt optimizer with a local tangent update (local parameterization)

    Inputs:
    pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
    max_iter (int): maximum number of iterations for the optimization

    Outputs: 
    final_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    final_cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    import jaxopt
    def lm_residual_fun(del_pose):
        return jax_stream_pose_local_update_residual(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    
    init_delta      = 1e-6 * jnp.ones(6)  # need to give this a non-zero initial guess in order for the solver to work
    solver          = jaxopt.LevenbergMarquardt(residual_fun  = lm_residual_fun, maxiter = max_iter)
    result_obj      = solver.run(init_delta)
    final_del_pose  = result_obj.params
    final_pose      = jax_stream_pose_tangent_update(pose0, final_del_pose)
    _, final_cost   = jax_stream_pose_resids_cost(final_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return final_pose, final_cost 
############################# Streamlined Functions that will use jaxopt to do a local tangent update #############################



############################# Streamlined Functions that will use pytorch to do a local tangent update ############################

def torch_pose_resids_cost(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam): 
    """ 
    Computes residuals and costs for Bearing measurements with with pose vector consisting of translation and quaternion

    Inputs:
    N is the number of keypoints`
    pose0 (torch.Tensor): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (torch.Tensor): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (torch.Tensor): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (torch.Tensor): (3,) array representing the position of the camera in the chaser body-fixed frame

    Outputs:
    resid_weighted (torch.Tensor): (2*N,) array representing the weighted residuals for the bearing measurements of form
        [resid_az_1, resid_el_1, resid_az_2, resid_el_2, ..., resid_az_N, resid_el_N]
    cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    import torch
    # rvec_cam = torch.zeros(3, device = pose0.device, dtype = pose0.dtype)
    # eps         = 0
    eps         = 1e-8 # small value to avoid division by zero
    # extract position and quaternion from pose0
    rvec        = pose0[:3] # (3,), relative position vector from chaser to target body-fixed frame origins in chaser body-fixed frame
    quat_c2t    = pose0[3:] # (4,), quaternion representing attitude transformation from chaser to target body-fixed frame
    quat_t2c    = TorchQuatMath.torch_q_conj(quat_c2t) # quaternion representing attitude transformation from target to chaser body-fixed frame
    
    trfMat      = TorchQuatMath.torch_q2trfm(quat_t2c) # shape: (3, 3)
 
    # transform keypoints from target frame to "camera" frame
    # rvec - rvec_cam => shift from chaser origin to camera, then
    # plus transformation (passive rotation) * the 3D keypoints
    # shape after multiplication: (3, N)
    # r_c_i       = (rvec - rvec_cam)[:, None] + trfMat @ kps_3D.T
    shifted     = (rvec - rvec_cam).unsqueeze(-1) # (3, 1)
    r_c_i       = shifted + trfMat @ kps_3D.transpose(0, 1) # (3, N)
    pred_az     = torch.arctan2(r_c_i[0], r_c_i[2] + eps)
    pred_el     = torch.arctan2(r_c_i[1], r_c_i[2] + eps)
    # compute residual in azimuth
    resid_az    = az_el_2D[:, 0] - pred_az
    # compute residual in elevation
    resid_el    = az_el_2D[:, 1] - pred_el
    # concatenate into a single 1D array of size 2*N with alternating azimuth and elevation residuals
    resid       = torch.stack([resid_az, resid_el], dim = -1).reshape(-1)

    # weight residuals by inverse of measurement standard deviation
    # not squared because we are using the residuals for optimization
    resid_weighted  = resid / bearing_meas_std_rad
    cost            = (resid_weighted**2).sum()
    return resid_weighted, cost


def torch_pose_tangent_update(pose_7, delta_6, max_theta):
    """
    Updates the pose using the tangent update delta (a 6-vector)

    Inputs:
    pose_7 (torch.Tensor): (7,) array representing the relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    delta_6 (torch.Tensor): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    max_theta (float): maximum angle for the rotation update, only pi b/c we are using the exponential map

    Outputs:
    new_pose (torch.Tensor): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q _x, q_y, q_z]
    """
    import torch
    # max_theta   = torch.pi
    eps         = 1e-8
    tr_base     = pose_7[:3]
    quat_base   = pose_7[3:]
    delta_tr    = delta_6[:3]
    delta_theta = delta_6[3:]
    angle       = torch.linalg.norm(delta_theta)
    safe_angle  = torch.where(angle < eps, eps, angle)
    scaled_dt   = delta_theta * (max_theta/safe_angle)
    delta_theta = torch.where(angle > max_theta, scaled_dt, delta_theta)
    new_t       = tr_base + delta_tr
    q_delta     = TorchQuatMath.torch_qexp_map(delta_theta)
    new_q       = TorchQuatMath.torch_qmult_ham(q_delta, quat_base)
    new_q       = TorchQuatMath.torch_q_norm(new_q)
    new_pose    = torch.concatenate([new_t, new_q])
    return new_pose

def torch_pose_local_update_residual(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_theta):
    """
    Calculates the residuals for the bearing measurements after a local update to the pose

    Inputs:
    pose0 (torch.Tensor): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    del_pose (torch.Tensor): (6,) array representing the tangent update; [delta_r_x, delta_r_y, delta_r_z, delta_theta_x, delta_theta_y, delta_theta_z]
    az_el_2D (torch.Tensor): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (torch.Tensor): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (torch.Tensor): (3,) array representing the position of the camera in the chaser body-fixed frame
    max_theta (float): maximum angle for the rotation update, only pi b/c we are using the exponential map

    Outputs:
    resid (torch.Tensor): (2*N,) array representing the weighted residuals for the bearing measurements
    """
    import torch
    new_pose    = torch_pose_tangent_update(pose0, del_pose, max_theta)
    resid, _    = torch_pose_resids_cost(new_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return resid 

def torch_solve_pose_local_torch_adam(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_theta, max_iter = 5, lr = 1e-3):
    """
    """
    import torch
    # need to give this a non-zero initial guess in order for the solver to work
    delta_6     = torch.nn.Parameter(1e-6 * torch.ones(6, dtype = pose0.dtype, device = pose0.device))
    opt         = torch.optim.Adam([delta_6], lr = lr)

    for iteration in range(max_iter):
        opt.zero_grad()
        resid   = torch_pose_local_update_residual(pose0, delta_6, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_theta)
        cost    = (resid**2).sum()
        cost.backward(retain_graph = True)
        opt.step()

    final_pose      = torch_pose_tangent_update(pose0, delta_6, max_theta)
    _, final_cost   = torch_pose_resids_cost(final_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return final_pose, final_cost


def torch_solve_pose_local_custom_lm(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_theta, max_iter = 5):
    """
    Naive Levenberg-Marquardt optimizer for the pose optimization problem within pytorch and using a local tangent update

    Inputs:
    pose0 (torch.Tensor): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    az_el_2D (torch.Tensor): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
    kps_3D (torch.Tensor): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
    bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
    rvec_cam (torch.Tensor): (3,) array representing the position of the camera in the chaser body-fixed frame
    max_theta (float): maximum angle for the rotation update, only pi b/c we are using the exponential map
    max_iter (int): maximum number of iterations for the optimization

    Outputs:
    final_pose (torch.Tensor): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
    final_cost (float): sum of squares of weighted residuals for the bearing measurements
    """
    import torch
    from torch.autograd.functional import jacobian
    def lm_residual_fun(del_pose):
        return torch_pose_local_update_residual(pose0, del_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, max_theta)
    
    delta_6         = 1e-6 * torch.ones(6, dtype = pose0.dtype, device = pose0.device)  # need to give this a non-zero initial guess in order for the solver to work
    resid           = lm_residual_fun(delta_6)
    prev_cost       = (resid**2).sum()
    lam             = 1e-3
    best_delta      = delta_6.clone()
    best_cost       = prev_cost.clone()
    for iteration in range(max_iter):
        def wrapped_delta_6(x):
            return lm_residual_fun(x)
        J   = jacobian(wrapped_delta_6, delta_6, create_graph = True)
        # J is (2*N, 6)
        # standard Levengerg-Marquardt update is 
        # delta = (J^T J + lam I)^{-1} * J^T  * r

        r_vec       = resid # flatten to (2N,)
        # normal eqns
        J_tr        = J.transpose(0, 1) # (6, 2*N)
        A           = J_tr @ J + lam * torch.eye(6, dtype = pose0.dtype, device = pose0.device)
        b           = J_tr @ r_vec
        # solve step
        delta_step  = torch.linalg.solve(A, b)
        # eval candidate 
        candidate   = delta_6 - delta_step
        reisd_new   = lm_residual_fun(candidate)
        cost_new    = (reisd_new**2).sum()

        # check if cost is better
        if cost_new < prev_cost:
            delta_6     = candidate
            prev_cost   = cost_new
            resid       = reisd_new
            lam         = lam * 0.2
            if cost_new < best_cost:
                best_cost   = cost_new
                best_delta  = delta_6.clone()
        else:
            lam         = lam * 5.0
            if lam > 1e10:
                break
    
    final_pose      = torch_pose_tangent_update(pose0, best_delta, max_theta)
    _, final_cost   = torch_pose_resids_cost(final_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
    return final_pose, final_cost 
############################# Streamlined Functions that will use pytorch to do a local tangent update ############################


# # Testing
# import torch
# n_kps           = 40
# torch_pose0     = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype = torch.float64)
# torch_del6      = torch.tensor([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6], dtype = torch.float64)
# max_theta       = torch.pi
# torch_az_el_2D  = torch.rand( (n_kps, 2), dtype = torch.float64) 
# torch_kps_3D    = torch.rand( (n_kps, 3), dtype = torch.float64)
# torch_bearing_meas_std_rad = 1e-3
# torch_rvec_cam  = torch.tensor([0.0, 0.0, 0.0], dtype = torch.float64)
# torch_resids, torch_cost = torch_pose_resids_cost(torch_pose0, torch_az_el_2D, torch_kps_3D, torch_bearing_meas_std_rad, torch_rvec_cam)
# torch_new_pose  = torch_pose_tangent_update(torch_pose0, torch_del6, max_theta)
# torch_local     = torch_pose_local_update_residual(torch_pose0, torch_del6, torch_az_el_2D, torch_kps_3D, torch_bearing_meas_std_rad, torch_rvec_cam, max_theta)
# print(f'torch_resids = {torch_resids} with shape {torch_resids.shape}')
# print(f'torch_cost = {torch_cost} with shape {torch_cost.shape}')
# print(f'torch_new_pose = {torch_new_pose} with shape {torch_new_pose.shape}')
# print(f'torch_local = {torch_local} with shape {torch_local.shape}')
# final_pose, final_cost  = torch_solve_pose_local_custom_lm(torch_pose0, torch_az_el_2D, torch_kps_3D, torch_bearing_meas_std_rad, torch_rvec_cam, max_theta)
# print(f'final_pose = {final_pose} with shape {final_pose.shape}')
# print(f'final_cost = {final_cost} with shape {final_cost.shape}')
# final_pose2, final_cost2    = torch_solve_pose_local_torch_adam(torch_pose0, torch_az_el_2D, torch_kps_3D, torch_bearing_meas_std_rad, torch_rvec_cam, max_theta)
# print(f'final_pose2 = {final_pose2} with shape {final_pose2.shape}')
# pdb.set_trace()





################### Functions that will use jaxopt to do a local tangent update using a simple gradient update ####################
# def jax_update_rotation(rtang, delta_theta):
#     """
#     Updates the rotation parameters using the error perturbation
#     """
#     q_curr      = QuatMath.jax_qexp_map(rtang)
#     q_delta     = QuatMath.jax_qexp_map(delta_theta)
#     q_new       = QuatMath.jax_qmulut_ham(q_delta, q_curr)
#     rtang_new   = QuatMath.jax_quat_to_exp_map(q_new)
#     return rtang_new

# def jax_quaternion_update(quat, delta):
#     """ Update quaterion using tangent update delta (a 3-vector) """
#     dq = QuatMath.jax_qexp_map(delta)
#     return QuatMath.jax_qmulut_ham(dq, quat)

# def jax_project_to_tangent(quat, grad_q):
#     """ 
#     Project the gradient grad_q (4-vector) onto the tangent space of quaternion q
#     For a unit quaternion, the tangent space consists of all 4-vectors orthogonal to q 
#     """
#     return grad_q - jnp.dot(grad_q, quat) * quat

# def jax_solve_pose_expmap_manifold(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, learning_rate = 1e-3, max_iter = 10000):
#     """
#     Solves for the relative pose between the chaser and target body-fixed frames using the exponential map representation of the quaternion and small updates

#     Inputs:
#         pose0 (jnp.array): (6,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
#         az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
#         kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
#         bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
#         rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
#         learning_rate (float): step size for the gradient descent optimization
#         max_iter (int): maximum number of iterations for the optimization

#     Outputs:
#         optimized_pose (jnp.array): (6,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, theta_x, theta_y, theta_z]
#     """
#     # pose            = pose0
#     def iteration(i, state):
#         pose        = state
#         cost, grad  = cost_and_grad(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
#         new_tr      = pose[:3] - learning_rate * grad[:3]
#         new_rot     = jax_update_rotation(pose[3:], -learning_rate * grad[3:])
#         new_pose    = jnp.concatenate([new_tr, new_rot])
#         return new_pose
    
#     optimized_pose  = jax.lax.fori_loop(0, max_iter, iteration, pose0)
#     return optimized_pose


# @jax.jit
# def jax_pose_cost_and_grad(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam):
#     """ Computes the cost and gradient for the cost function and has been jitted for optimization """
#     cost_val, grad_val  = jax.value_and_grad(jax_pose_cost_unconstrained)(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
#     return cost_val, grad_val

# @jax.jit
# def jax_solve_pose_manifold(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, learning_rate = 1e-3, max_iter = 200):
#     """
#     Solves for the relative pose between the chaser and target body-fixed frames using small quaterion updates

#     Inputs:
#         pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
#         az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
#         kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
#         bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
#         rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
#         learning_rate (float): step size for the gradient descent optimization
#         max_iter (int): maximum number of iterations for the optimization

#     Outputs:
#         optimized_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
#     """

#     def iteration(i, state):
#         pose        = state
#         cost, grad  = jax_pose_cost_and_grad(pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam)
#         tr          = pose[:3]
#         quat        = pose[3:]
#         tr_grad     = grad[:3]
#         quat_grad   = grad[3:]
#         new_tr      = tr - learning_rate * tr_grad
#         tang_grad   = jax_project_to_tangent(quat, quat_grad)
#         tang_update = -learning_rate * tang_grad
#         new_quat    = jax_quaternion_update(quat, tang_update)
#         new_pose    = jnp.concatenate([new_tr, new_quat])
#         return new_pose
    
#     optimized_pose  = jax.lax.fori_loop(0, max_iter, iteration, pose0)
#     final_cost, _   = jax_pose_cost_and_grad(optimized_pose, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam) 
#     return optimized_pose, final_cost

# def jax_solve_pose_reint_parallel(pose0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, learning_rate = 1e-2, max_iter = 200, n_init = 100):
#     """
#     Optimizes the relative pose between the chaser and target body-fixed frames using multiple random re-initializations in parallel

#     Inputs:
#         pose0 (jnp.array): (7,) array representing the initial relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]
#         az_el_2D (jnp.array): (N, 2) measurement array representing the azimuth and elevation angles in the image plane
#         kps_3D (jnp.array): (N, 3) array representing the 3D keypoints in the target body-fixed frame w/ (x, y, z) format
#         bearing_meas_std_rad (float): standard deviation of the bearing measurements (radians), tuning parameter for the cost function
#         rvec_cam (jnp.array): (3,) array representing the position of the camera in the chaser body-fixed frame
#         learning_rate (float): step size for the gradient descent optimization
#         max_iter (int): maximum number of iterations for the optimization
#         n_init (int): number of re-initializations

#     Outputs:
#         optimized_pose (jnp.array): (7,) array representing the updated relative pose between the chaser and target body-fixed frames; [r_x, r_y, r_z, q_w, q_x, q_y, q_z]

#     """
#     keys    = jax.random.split(jax.random.PRNGKey(0), n_init)

#     def init_guess(i, key):
#         quat0       = jnp.where(i == 0, pose0[3:], QuatMath.jax_random_quat(key))
#         new_pose    = jnp.concatenate([pose0[:3], quat0])
#         return new_pose
    
#     poses_init  = jnp.stack([init_guess(i, key) for i, key in enumerate(keys)], axis = 0)

#     opt_vmap        = jax.jit( jax.vmap(lambda p0: jax_solve_pose_manifold(p0, az_el_2D, kps_3D, bearing_meas_std_rad, rvec_cam, learning_rate, max_iter) ) )
#     poses, costs    = opt_vmap(poses_init)

#     best_idx        = jnp.argmin(costs)
#     best_pose       = poses[best_idx]
#     best_cost       = costs[best_idx]
#     return best_pose, best_cost
################### Functions that will use jaxopt to do a local tangent update using a simple gradient update ####################