import warnings

import numpy as np
from scipy import odr
from sklearn.linear_model import LinearRegression


def estimate_trajectory_interaction_amount_using_current_to_future_velocity_differences(trajectories, dt=8):
    """
    Function to estimate the amount of the interaction in the given trajectories.
    We use the per-step curvature definition from equation 6 in https://arxiv.org/pdf/2111.14820.pdf
    and average the per-step curvature to get the interaction estimate.

    @param trajectories: np.array of shape (n_agents, n_timesteps, n_coordinates); e.g., (6,20,2)
    @param dt: The delta t parameter, see the definition of the interaction estimate
               in equation 6 in https://arxiv.org/pdf/2111.14820.pdf
    @return: np.array of shape (n_agents), each value corresponding to the estimated trajectory interaction amount
    """
    n_agents, n_timesteps, n_coordinates = trajectories.shape

    current_speed = trajectories[:, 1:, :] - trajectories[:, :-1, :]
    per_timestep_interaction = np.linalg.norm(current_speed[:, dt:, :] - current_speed[:, :-dt, :], ord=2, axis=-1)
    assert per_timestep_interaction.shape == (n_agents, n_timesteps - dt - 1)

    trajectory_interaction = per_timestep_interaction.mean(axis=-1)
    assert trajectory_interaction.shape == (n_agents,)

    return trajectory_interaction


def estimate_future_trajectory_curvature_using_linear_regression(trajectories, future_start_timestep):
    """
    Function to estimate the curvature of given trajectories using the $R^2$ value of a linear regression fitted
    to the future trajectory. Assumes there are only two coordinates: x and y.
    Nota bene: this trajectory curvature estimate is useless since it is not rotation invariant.

    @param trajectories: np.array of shape (n_agents, n_timesteps, n_coordinates); e.g., (6,20,2)
    @param future_start_timestep: The timestep at which the future starts. E.g., if the trajectory is 20 timesteps long,
        and 8 timesteps correspond to the past (i.e., observations are trajectories[:,:8,:])
        and the remaining 12 to the future (i.e., trajectories[:,8:,:]),
        then one would provide 8 as the future_start_timestep.
    @return: np.array of shape (n_agents), each value corresponding to the estimated
    trajectory curvature
    """
    warnings.warn('Warning: R^2 value of linear regression is '
                  'a useless curvature estimate for trajectories '
                  'since it is not rotation invariant. Rotating '
                  'a trajectory could give unreasonably different '
                  'results, because the errors are computed as a '
                  '"vertical" distance from the point to the '
                  'line evaluated at the x coordinate of the point. '
                  'We recommand using total least squares '
                  '(i.e., orthogonal distance regression) instead.')

    n_agents, n_timesteps, n_coordinates = trajectories.shape
    assert n_coordinates == 2

    trajectory_curvature_list = []
    for agent_idx in range(n_agents):
        X = trajectories[agent_idx, future_start_timestep:, 0:1]
        y = trajectories[agent_idx, future_start_timestep:, 1]
        r2_score = LinearRegression().fit(X, y).score(X, y)
        trajectory_curvature_list.append(r2_score)

    trajectory_curvature = np.array(trajectory_curvature_list)
    assert trajectory_curvature.shape == (n_agents,)

    return trajectory_curvature


def estimate_future_trajectory_curvature_by_linearly_extrapolating_past(trajectories, future_start_timestep,
                                                                        reduce_fn=np.mean):
    """
    Function to estimate the curvature of given trajectories by computing the square distance
    between future trajectory points and a total least square regression line fitted to the past trajectory points.
    The square distances are then reduced using the given function, e.g., `np.mean` or `np.max`.
    Assumes there are only two coordinates: x and y.
    If the value is close to 0.0, then there is not much deviation
    of future trajectory points from the linear extrapolation of the past.
    It is an estimate of the curvature only in the sense that the measures how bad a linear approximation is.
    Formally, the estimate does not model the curvature per se in any way.
    Also, the order of the trajectory points does not matter (i.e., time is not taken into account).

    @param trajectories: np.array of shape (n_agents, n_timesteps, n_coordinates); e.g., (6,20,2)
    @param future_start_timestep: The timestep at which the future starts. E.g., if the trajectory is 20 timesteps long,
                                  and 8 timesteps correspond to the past (i.e., observations are trajectories[:,:8,:])
                                  and the remaining 12 to the future (i.e., trajectories[:,8:,:]),
                                  then one would provide 8 as the future_start_timestep.
    @param reduce_fn: Function called on top of the 1D np.array of squared distances for each trajectory.
                      Used to reduce the squred distances into one number representing the curvature.
    @return: np.array of shape (n_agents), each value corresponding to the estimated
    trajectory curvature
    """
    n_agents, n_timesteps, n_coordinates = trajectories.shape
    assert n_coordinates == 2

    def linear_odf_f(betas, x):
        a, b = betas
        return a * x + b

    odr_model = odr.Model(linear_odf_f)

    def distance_from_line_to_point(betas, points2d):
        a, b = betas
        p1 = np.array([0, a * 0 + b])
        p2 = np.array([1, a * 1 + b])
        return np.abs(np.cross(p2 - p1, points2d - p1)) / np.linalg.norm(p2 - p1)

    trajectory_curvature_list = []
    for agent_idx in range(n_agents):
        x_past = trajectories[agent_idx, :future_start_timestep, 0]
        y_past = trajectories[agent_idx, :future_start_timestep, 1]

        odr_data = odr.Data(x_past, y_past)
        ordinal_distance_reg = odr.ODR(odr_data, odr_model, beta0=[1.0, 0.])
        out = ordinal_distance_reg.run()

        xy_future = trajectories[agent_idx, future_start_timestep:, :]
        future_odr_square_error = distance_from_line_to_point(out.beta, xy_future) ** 2
        curvature = reduce_fn(future_odr_square_error)

        trajectory_curvature_list.append(curvature)

    trajectory_curvature = np.array(trajectory_curvature_list)
    assert trajectory_curvature.shape == (n_agents,)

    return trajectory_curvature
