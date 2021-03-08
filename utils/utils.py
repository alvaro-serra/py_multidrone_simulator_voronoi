import numpy as np

def predictStateConstantV(state_now, dt, N):
    state_pred = np.zeros((6,N))
    state_pred[:,0:1] = state_now.copy()
    xpred = state_now.copy()
    F = np.array([[1, 0, 0, dt, 0, 0],
                 [0, 1, 0, 0, dt, 0],
                 [0, 0, 1, 0, 0, dt],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]])

    for iStage in range(1,N):
        xpred = np.matmul(F,xpred)
        state_pred[:,iStage:iStage+1] = xpred

    return state_pred


def predictQuadPathFromCom(quad_traj_com, quad_state_now, time_step_now, dt, error_pos_tol):
    # predict other quad traj based on last communicated trajectory

    # get info from communicated trajectory
    N = int(np.shape(quad_traj_com)[1]) # quad_traj_com_ is 7xN
    time_step_comm = quad_traj_com[0,0]
    quad_traj_pred = np.zeros((6,N)) # first vector is the first prediction step (NOT CURRENT STATE)

    dN_elapsed = int(time_step_now - time_step_comm)
    dN_left = int(N -dN_elapsed)
    if dN_elapsed <= N and dN_elapsed > 0 and dN_left > 0:
        # choose the comm pos
        state_from_comm = quad_traj_com[1:7, int(dN_elapsed)-1:int(dN_elapsed)]
        error_pos = np.linalg.norm(state_from_comm[0:3] - quad_state_now[0:3])
        if error_pos < error_pos_tol: # use some com info
            ifAbandonCom = 0          # not abandon the com info
            # use part of the com traj
            quad_traj_pred[:,0:dN_left] = quad_traj_com[1:7, dN_elapsed:N]
            # the left uses constant v for prediction
            quad_traj_pred[:, dN_left-1:N] = predictStateConstantV(quad_traj_com[1:7,N-1:N], dt,dN_elapsed+1)

        else:
            ifAbandonCom = 1 # abandon the com info

    else:
        ifAbandonCom = 1 # com is too old

    if ifAbandonCom == 1: # if the com is not useful
        quad_traj_pred = predictStateConstantV(quad_state_now, dt, N) # Last term is the number of timesteps

    return quad_traj_pred, ifAbandonCom