import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    V, om = u[0], u[1]
    x_prev, y_prev, th_prev = xvec[0], xvec[1], xvec[2]
    
    if abs(om) < EPSILON_OMEGA:
        # use limit of omega goes to 0
        # compute g
        th = th_prev + om * dt
        sin_t = np.sin(th_prev) + np.sin(th)
        cos_t = np.cos(th_prev) + np.cos(th)
        x = x_prev + 0.5 * V * cos_t * dt
        y = y_prev + 0.5 * V * sin_t * dt
        # compute Gx -> (3, 3)
        Gx = [[1, 0, -0.5 * V * sin_t * dt],
              [0, 1,  0.5 * V * cos_t * dt],
              [0, 0, 1]]
        # compute Gu -> (2, 3)
        Gu = [[0.5 * cos_t * dt, -0.5 * V * np.sin(th) * dt * dt],
              [0.5 * sin_t * dt,  0.5 * V * np.cos(th) * dt * dt],
              [0, dt]]

    else:
        # regular
        th = th_prev + om * dt
        # pre-calculate
        sin_th = np.sin(th)
        sin_th_p = np.sin(th_prev)
        cos_th = np.cos(th)
        cos_th_p = np.cos(th_prev)
        om_inv = 1.0 / om
        V_om_inv = V * om_inv
        # compute g
        x = (x_prev) + V_om_inv * (sin_th - sin_th_p)
        y = (y_prev) + V_om_inv * (cos_th_p - cos_th)
        # compute Gx -> (3, 3)
        Gx = [[1, 0, V_om_inv * (cos_th - cos_th_p)],
              [0, 1, V_om_inv * (sin_th - sin_th_p)],
              [0, 0, 1]]
        # comput Gu -> (2, 3)
        Gu = [[om_inv * (sin_th - sin_th_p),
               V * (om_inv**2 * (sin_th_p - sin_th) + om_inv * cos_th * dt)],
              [om_inv * (cos_th_p - cos_th),
               V * (om_inv**2 * (cos_th - cos_th_p) + om_inv * sin_th * dt)],
              [0, dt]]

    g = np.array([x, y, th])
    Gx = np.array(Gx)
    Gu = np.array(Gu)
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ######1#### Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), 
    #       a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base 
    #                   in world frame (x_base, y_base, th_base)
    
    # calculate pose of camera in world frame
    # t = np.linalg.norm([x, tf_base_to_camera])
    x_world, y_world, th_world = x[0], x[1], x[2] 
    R = [
        [np.cos(th_world), -np.sin(th_world), 0],
        [np.sin(th_world), np.cos(th_world), 0],
        [0, 0, 1]
    ]

    cam_pose = x + np.dot(R, tf_base_to_camera)
    x_cam, y_cam, th_cam = cam_pose[0], cam_pose[1], cam_pose[2]

    h = np.array([alpha - th_cam,
                  r - x_cam * np.cos(alpha) - y_cam * np.sin(alpha)])
    x_base_cam, y_base_cam, _ = tf_base_to_camera
    temp =  np.cos(alpha) * (np.sin(th_world) * x_base_cam + y_base_cam * np.cos(th_world))
    temp += np.sin(alpha) * (-np.cos(th_world) * x_base_cam + np.sin(th_world) * y_base_cam)

    Hx = np.array([[0, 0, -1],
                   [-np.cos(alpha), -np.sin(alpha), temp]])
    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
