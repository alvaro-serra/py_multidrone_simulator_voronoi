import numpy as np
import numpy.matlib as matlib
#from solvers.solver.basic.Basic_Forces_11_20_50.FORCESNLPsolver_basic_11_20_50_py import FORCESNLPsolver_basic_11_20_50_solve as solver
from solvers.solver.basic.Basic_Forces_11_20_50 import FORCESNLPsolver_basic_11_20_50_py
from integrators.RK2 import rk2a_onestep as RK2
from dynamics import bebop_dynamics as dynamics
import warnings
warnings.simplefilter("always")

import time


# Problem setup for basic mav collision avoidance
# Used for generating FORCES PRO solvers and control node of the mav
# Define system physical parameters, problem dimensions and indexing here
def basic_setup(nQuad, nDynObs):

    # System physical parameters
    pr = {
        # speed limit, hard constraint
        "state":{
            "maxVx": 3.0,
            "maxVy": 3.0,
            "maxVz": 1.0
        },
        "input": {
            "maxRoll": np.deg2rad(20),
            "maxPitch": np.deg2rad(20),
            "maxVz": 1.0,
            "maxYawRate": np.deg2rad(90)
        },
    }

    # Problem dimensions, check matlab implementation for further explanation
    model = {
        "nDynObs": nDynObs,
        "nQuad": nQuad,
        "nObs": nDynObs + nQuad - 1,
        "nParamPerObs": 8,
        "N": 20,
        "dt": 0.05,
        "nvar": 15,
        "neq": 9
    }

    model["nh"] = 3 + model["nObs"] # number of inequality constraints
    model["nin"] = 4
    model["nslack"] = 2
    model["npar"] = 18 + model["nObs"]*model["nParamPerObs"]

    # Indexing, not changable when running
    index = {
        # in stage vector, each stage
        "z":{
            "all": list(range(model["nvar"])),
            "inputs": list(range(0,4)),
            "slack": list(range(4,6)),
            "pos": list(range(6,9)),
            "vel": list(range(9,12)),
            "euler": list(range(12,15)),
        },
        # in state vector, each stage
        "x": {
            "all": list(range(model["neq"])),
            "pos": list(range(0,3)),
            "vel": list(range(3,6)),
            "euler": list(range(6,9))
        },
        # in parameter vector, problem, each stage
        "p": {
            "all": list(range(0,model["npar"])),
            "envDim": list(range(0,3)),
            "startPos": list(range(3,7)),
            "wayPoint": list(range(7,11)),
            "size": list(range(11,14)),
            "weights": list(range(14,18))
        }
    }

    if model["nObs"] >= 1:
        idxBegin = index["p"]["weights"][-1] + 1
        auxarray = np.reshape(np.array(range(idxBegin,idxBegin+model["nParamPerObs"]*model["nObs"])),(model["nObs"],model["nParamPerObs"])).T
        #index["p"]["obsParam"] = [ list(nparray) for nparray in auxarray]
        index["p"]["obsParam"] = auxarray
        # index inside for each moving obstacle
        index["p"]["obs"] = {
            "pos": list(range(0, 3)),
            "size": list(range(3, 6)),
            "coll": list(range(6, 8))
        }

    return pr, model, index


def scn_circle(nQuad, radius):
    # quad initial and end positionis (yaw)
    quadStartPos = np.zeros((4,nQuad))
    quadStartVel = np.zeros((3,nQuad))
    quadEndPos = np.zeros((4,nQuad))

    angle = 2.0*np.pi / nQuad
    for iQuad in range(nQuad):
        #initial
        angle_i = np.deg2rad(0) + angle*(iQuad)
        quadStartPos[0:2,iQuad:iQuad+1] = np.array([[radius*np.cos(angle_i)],[radius*np.sin(angle_i)]])
        quadStartPos[2,iQuad] = 1.6  # flying height
        quadStartPos[3, iQuad] = np.deg2rad(0)  #yaw

        # end
        quadEndPos[0:2,iQuad:iQuad+1] = -quadStartPos[0:2,iQuad:iQuad+1]
        quadEndPos[2,iQuad] = quadStartPos[2,iQuad]
        quadEndPos[3,iQuad] = quadStartPos[3,iQuad]

    return quadStartPos, quadStartVel, quadEndPos


def predictStateConstantV(state_now, dt, N):
    state_pred = np.zeros((6,N))
    state_pred[:,0:1] = state_now
    xpred = state_now
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
    N = np.shape(quad_traj_com)[1] # quad_traj_com_ is 7xN
    time_step_comm = quad_traj_com[0,0]
    quad_traj_pred = np.zeros((6,N)) # first vector is the first prediction step (NOT CURRENT STATE)

    dN_elapsed = time_step_now - time_step_comm
    dN_left = N -dN_elapsed
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


def scn_random(nQuad, xDim, yDim, zDim):

    ## quad initial and end positions (yaw)
    quadStartPos = np.zeros((4,nQuad))
    quadStartVel = np.zeros((3,nQuad))
    quadEndPos = np.zeros((4,nQuad))

    ## generate random starting and end position
    resolution = 0.6
    # discrete the workspace
    xL = xDim[1] - xDim[0]
    yL = yDim[1] - yDim[1]
    zL = zDim[2] - zDim[2]

    # number of discrete points in each dimension
    xN = np.floor(xL/resolution)
    yN = np.floor(yL/resolution)
    zN = np.floor(zL/0.6)

    #non-repeating integer sequence
    xidx_rand = np.random.permutation(xN)
    yidx_rand = np.random.permutation(yN)

    # random starting and end pos
    for iQuad in range(nQuad):
        xN_i = xidx_rand[iQuad]
        yN_i = yidx_rand[iQuad]
        zN_i = np.random.randint(zN-1)
        quadStartPos[0, iQuad] = xDim[0] + resolution*xN_i
        quadStartPos[1, iQuad] = yDim[0] + resolution*yN_i
        quadStartPos[2, iQuad] = zDim[0] + resolution*zN_i
        #quadStartPos[2, iQuad] = 2.0

    xidx_rand = np.random.permutation(xN)
    yidx_rand = np.random.permutation(yN)
    #zidx_rand = np.random.permutation(zN)

    # random starting and end pos
    for iQuad in range(nQuad):
        xN_i = xidx_rand(iQuad)
        yN_i = yidx_rand(iQuad)
        zN_i = np.random.randint(zN-1)

        quadEndPos[0, iQuad] = xDim[0] + resolution*xN_i
        quadEndPos[1, iQuad] = yDim[0] + resolution*yN_i
        quadEndPos[2, iQuad] = zDim[0] + resolution*zN_i
        # quadEndPos[2, iQuad] = 2.0

    return quadStartPos, quadStartVel, quadEndPos


def scn_circle_random(nQuad, R_min, R_max):

    # quad initial and end positions (yaw)
    quadStartPos = np.zeros((4, nQuad))
    quadStartVel = np.zeros((3, nQuad))
    quadEndPos = np.zeros((4, nQuad))

    angle = 2.0*np.pi / nQuad
    ang_gap = 0.45*angle

    for iQuad in range(nQuad):
        #initial
        angle_c = np.deg2rad(0) + angle*(iQuad-1)
        angle_min = angle_c - ang_gap
        angle_max = angle_c + ang_gap
        angle_i = angle_min + (angle_max - angle_min)*np.random.random()
        R_i = R_min + (R_max - R_min)*np.random.random()
        quadStartPos[0:2, iQuad] = np.array([R_i*np.cos(angle_i), R_i*np.sin(angle_i)])
        quadStartPos[2, iQuad] = 1.6 # flying height
        quadStartPos[3, iQuad] = np.deg2rad(0) # yaw

        # end
        quadEndPos[0:2, iQuad] = -quadStartPos[0:2, iQuad]
        quadEndPos[2, iQuad] = quadStartPos[2, iQuad]
        quadEndPos[3, iQuad] = quadStartPos[3, iQuad]

    return quadStartPos, quadStartVel, quadEndPos


class pyCDrone():
    def __init__(self, quadID, quadExpID, cfg, pr, model, index):
        ##### var construction #####
        # timer
        self.time_global_ = 0
        self.time_step_global_ = 0

        #mode
        self.modeCoor_ = 0

        # real state
        self.pos_real_ = np.zeros((3,1))
        self.vel_real_ = np.zeros((3,1))
        self.euler_real_ = np.zeros((3,1))

        # estimated state
        self.pos_est_cov_ = np.eye(3)
        self.vel_est_cov_ = np.eye(3)
        self.euler_est_cov_ = np.eye(3)

        # running para
        self.quad_goal_ = np.zeros((4,1))
        self.mpc_coll_ = np.zeros((2,2))
        self.mpc_weights_ = np.zeros((4,2))

        #mpc plan
        self.mpc_pAll_ = None
        self.mpc_exitflag_ = None
        self.mpc_info_ = None
        self.mpc_Xk_ = None
        self.mpc_ZK_ = None
        self.mpc_Zk2_ = None
        self.mpc_ZPlan_ = None



        ##### initialization #####
        self.id_ = quadID
        self.exp_id_ = quadExpID
        self.cfg_ = cfg

        self.size_ = cfg["quad"]["size"]

        self.dt_ = model["dt"]
        self.N_ = model["N"]

        self.nQuad_ = model["nQuad"]
        self.nDynObs_ = model["nDynObs"]

        self.nvar_ = model["nvar"]
        self.npar_ = model["npar"]

        self.index_ = index

        self.maxRoll_ = pr["input"]["maxRoll"]
        self.maxPitch_ = pr["input"]["maxPitch"]
        self.maxVz_ = pr["input"]["maxVz"]
        self.maxYawRate_ = pr["input"]["maxYawRate"]

        self.quad_path_ = np.zeros((3, self.N_, self.nQuad_))
        self.quad_pathcov_ = np.zeros((6, self.N_, self.nQuad_))

        self.obs_path_ = np.zeros((3, self.N_, self.nDynObs_))
        self.obs_pathcov_ = np.zeros((6, self.N_, self.nDynObs_))

        for jObs in range(self.nDynObs_):
            for iStage in range(self.N_):
                self.obs_path_[2, iStage, jObs] = -2

        self.mpc_Path_ = np.zeros((3, self.N_))
        self.mpc_PathCov_ = np.zeros((6, self.N_))
        self.mpc_traj_ = np.zeros((7, self.N_))

        self.pred_path_ = np.zeros((3, self.N_))
        self.pred_pathcov_ = np.zeros((6, self.N_))

        self.quad_traj_com_ = np.zeros((7, self.N_, self.nQuad_)) #com iter, pos, vel
        self.quad_traj_pred_ = np.zeros((6, self.N_, self.nQuad_))
        self.quad_traj_pred_tol_ = 0.1

    def initializeMPC(self, x_start, mpc_plan):
        # initialize the initial conditions for the MPC solver with
        # x_start and mpc_plan, only used when necessary.
        self.mpc_Xk_ = x_start
        self.mpc_ZPlan_ = mpc_plan
        self.mpc_Path_ = mpc_plan[self.index_["z"]["pos"],:]

    #def initializeROS(self): Not needed for the moment
    # Initialize ROS publishers and subscribers for the quadrotor
    #subs to mocap raw data / bebop 2 est pos,vel,orient / predicted path of moving obstacles
    #pubs to mpc control input cmd_vel, Twist


    #def getObservedSystemState(self) #Not really used at the moment
    #Get measured real-time position and attitude of the drone

    def getEstimatedSystemState(self):
        #always simulated
        assert self.cfg_["modeSim"]
        # set estimated state the same as real one
        self.pos_est_ = self.pos_real_
        self.vel_est_ = self.vel_real_
        self.euler_est_ = self.euler_real_

        # add noise if necessary ( NOT NECESSARY, only with chance constraints )

    #def getObsPredictedPath(self):
    # Get predicted path of all moving obstacles
    # This function takes a long time

    #def getObsPredictedPathCov(self):
    # Get predicted path of all moving obstacles
    # This function takes a long time

    def setOnlineParameters(self):
        # Set the real-time parameter vector
        # pAll include parameters for all N stage

        # prepare parameters
        envDim = self.cfg_["ws"]
        startPos = np.concatenate([self.pos_est_, [self.euler_est_[2]]], 0)
        wayPoint = self.quad_goal_
        egoSize = self.size_
        weightStage = self.mpc_weights_[:,0]
        weightN = self.mpc_weights_[:,1]
        quadSize = self.cfg_["quad"]["size"]
        obsSize = self.cfg_["obs"]["size"]
        quadColl = self.mpc_coll_[0:2,0:1] #lambda, buffer (sigmoid function)
        obsColl = self.mpc_coll_[0:2,1:2] #lambda, buffer
        quadPath = self.quad_path_
        obsPath = self.obs_path_

        # all stage parameters
        pStage = np.zeros((self.npar_, 1))
        self.mpc_pAll_ = matlib.repmat(pStage, self.N_, 1)
        for iStage in range(0,self.N_):
            #general parameter
            pStage[self.index_["p"]["envDim"]] = envDim
            pStage[self.index_["p"]["startPos"]] = startPos
            pStage[self.index_["p"]["wayPoint"]] = wayPoint
            pStage[self.index_["p"]["size"]] = egoSize
            pStage[self.index_["p"]["weights"], 0] = weightStage
            # obstacle information, including other quadrotors
            # and moving obstacles, set other quad first
            idx = 0
            for iQuad in range(self.nQuad_):
                if iQuad == self.id_:
                    continue
                else:
                    pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["pos"], idx]] = quadPath[:, iStage,iQuad:iQuad+1]
                    pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["size"], idx]] = quadSize
                    pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["coll"], idx]] = quadColl
                    idx = idx + 1

            for jObs in range(self.nDynObs_):
                pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["pos"],idx]] = obsPath[:, iStage, jObs:jObs+1]
                pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["size"], idx]] = obsSize
                pStage[self.index_["p"]["obsParam"][self.index_["p"]["obs"]["coll"], idx]] = obsColl
                idx = idx + 1

            # change the last stage cost term weights
            if iStage == self.N_-1:
                pStage[self.index_["p"]["weights"], 0] = weightN

            # insert into the all stage parameter
            self.mpc_pAll_[self.npar_ * iStage : self.npar_ * (iStage+1)] = pStage


    #def setOnlineParametersCov(self): # NOT NECESSARY, ONLY WHEN CONSIDERING CHANCE CONSTRAINTS
        # Set the real-time parameter vector
        # pAll include parameters for all N stage
        ## prepare parameters


    def solveMPC(self): #Might be some issues with the shape of the vectors
        # Calling the solver to solve the mpc for collision avoidance
        problem ={}
        problem["all_parameters"] = self.mpc_pAll_

        #set initial conditions
        self.mpc_Xk_ = np.concatenate([self.pos_est_, self.vel_est_, self.euler_est_], 0)
        problem["xinit"] = self.mpc_Xk_

        #prepare initial guess
        #self.mpc_exitflag_ = 0 # for debugging
        if self.mpc_exitflag_ == 1: # last step mpc feasible
            x0_temp = np.reshape(np.concatenate([self.mpc_ZPlan_[:,1:self.N_], self.mpc_ZPlan_[:,(self.N_-1):self.N_]], axis=1).T,(self.N_*self.nvar_,1))

        else: # last step mpc infeasible
            x0_temp_stage = np.zeros((self.nvar_,1))
            x0_temp_stage[self.index_["z"]["pos"]+self.index_["z"]["vel"]+self.index_["z"]["euler"]] = self.mpc_Xk_
            x0_temp = matlib.repmat(x0_temp_stage, self.N_, 1)

        problem["x0"] = x0_temp
        #problem["num_of_threads"] = 1

        # call the NLP solver
        #OUTPUT, EXITFLAG, INFO = solver(problem)
        OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_basic_11_20_50_py.FORCESNLPsolver_basic_11_20_50_solve(problem)

        # store solving information
        self.mpc_exitflag_ = EXITFLAG
        self.mpc_info_ = INFO

        # store output
        for iStage in range(self.N_):
            self.mpc_ZPlan_[:,iStage] = OUTPUT["x{0:0=2d}".format(iStage+1)]
            self.mpc_Path_[:,iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"],iStage]
            self.mpc_traj_[0,iStage] = self.time_step_global_
            self.mpc_traj_[1:7, iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"]+self.index_["z"]["vel"], iStage]

        self.mpc_Zk_ = self.mpc_ZPlan_[:,0:1]
        self.mpc_Zk2_ = self.mpc_ZPlan_[:,1:2]


        # check the exitflag and get optimal control input
        if EXITFLAG == 0:
            warnings.warn("MPC: Max iterations reached!")
        elif EXITFLAG == -4:
            warnings.warn("MPC: Wrong number of inequalities input to solver!")
        elif EXITFLAG == -5:
            warnings.warn("MPC: Error occured during matrix factorization!")
        elif EXITFLAG == -6:
            warnings.warn("MPC: NaN or INF occured during functions evaluations!")
        elif EXITFLAG == -7:
            warnings.warn("MPC: Infeasible! The solver could not proceed!")
        elif EXITFLAG == -10:
            warnings.warn("MPC: NaN or INF occured during evaluation of functions and derivatives!")
        elif EXITFLAG == -11:
            warnings.warn("MPC: Invalid values in problem parameters!")
        elif EXITFLAG == -100:
            warnings.warn("MPC: License error!")


        if EXITFLAG == 1:
            #if mpc solved successfully
            self.u_mpc_ = self.mpc_Zk_[self.index_["z"]["inputs"]]
        else:
            # if infeasible
            self.u_mpc_ = -0.0 * self.u_mpc_

        # transform u, check the using dynamics model before doing this!
        yaw = self.euler_est_[2]
        self.u_body_ = self.u_mpc_
        self.u_body_[0] = self.u_mpc_[1]*np.sin(yaw) + self.u_mpc_[0]*np.cos(yaw) #TODO: clarify with Hai
        self.u_body_[1] = self.u_mpc_[1]*np.cos(yaw) + self.u_mpc_[0]*np.sin(yaw) #u_mpc global --> here transform to local
                                                                            # this is only useful if performing real experiments

    def step(self):
        # send and execute the control command
        #TODO: we assume we are in simulation, build loop for real experiments

        # simulate one step in simple simulation mode
        # current state and control
        xNow = np.concatenate([self.pos_real_, self.vel_real_, self.euler_real_],0)
        u = self.u_mpc_ # use u_mpc in simulation --> no need to transform to local

        # integrate one step
        xNext = RK2(xNow, u, [0,self.dt_])


        #update the implicit real state
        self.pos_real_ = xNext[self.index_["x"]["pos"]]
        self.vel_real_ = xNext[self.index_["x"]["vel"]]
        self.euler_real_ = xNext[self.index_["x"]["euler"]]

    def propagateStateCov(self): # NOT NECESSARY, ONLY WHEN CONSIDERING CHANCE CONSTRAINTS
        #Propagate uncertainty covariance along the path
        ## model parameters
        g = 9.81
        kD_x = 0.25
        kD_y = 0.33
        tau_vz = 0.3367
        tau_phi = 0.2368
        tau_theta = 0.2318

        #current state uncertainty covariance
        S0 = np.zeros((9,9))
        S0[0:3, 0:3] = self.pos_est_cov_
        S0[3:6,3:6] = self.vel_est_cov_
        S0[6:9,6:9] = self.euler_est_cov_

        #uncertainty propagation
        S_Now = S0
        for iStage in range(self.N_):
            #store path cov
            self.mpc_PathCov_[:, iStage] = np.array([S_Now[0,0], S_Now[1,1], S_Now[2,2], S_Now[0,1], S_Now[1,2],S_Now[0,2]])
            # state transition matrix
            F_Now = np.zeros((9,9))
            phi_Now = self.mpc_ZPlan_[self.index_["z"]["euler"][0], iStage]
            theta_Now = self.mpc_ZPlan_[self.index_["z"]["euler"][1], iStage]
            F_Now[0,0] = 1
            F_Now[0,3] = self.dt_
            F_Now[1,1] = 1
            F_Now[1,4] = self.dt_
            F_Now[2,2] = 1
            F_Now[2,5] = self.dt_
            F_Now[3,3] = 1 - self.dt_*kD_x
            F_Now[3,7] = g*self.dt_/(np.cos(theta_Now))**2
            F_Now[4,4] = 1-self.dt_*kD_y
            F_Now[4,6] = -g*self.dt_ / (np.cos(phi_Now))**2
            F_Now[5,5] = 1-self.dt_/tau_vz
            F_Now[6,6] = 1-self.dt_/tau_phi
            F_Now[7,7] = 1-self.dt_/tau_theta
            F_Now[8,8] = 1
            # uncertainty propagation
            S_Next = np.matmul(F_Now,np.matmul(S_Now,F_Now.T))
                # S_Next = S_now # for debugging
            # set next to now
            S_Now = S_Next

    def predictPathConstantV(self):
        # Predict quad path based on constant velocity assumption
        self.pred_path_[:,0:1] = self.pos_est_
        self.pred_pathcov_[:, 1] = np.array([self.pos_est_cov_[0,0], self.pos_est_cov_[1,1],
                                             self.pos_est_cov_[2,2], self.pos_est_cov_[0,1],
                                             self.pos_est_cov_[1,2], self.pos_est_cov_[0,2]])
        xpred = np.concatenate([self.pos_est_, self.vel_est_],0)
        aux0 = np.concatenate([self.pos_est_cov_, np.zeros((3,3))], 1)
        aux1 = np.concatenate([np.zeros((3,3)), self.vel_est_cov_], 1)
        Ppred = np.concatenate([aux0,aux1], 0)

        F = np.array([[1, 0, 0, self.dt_, 0, 0],
                      [0, 1, 0, 0, self.dt_, 0],
                      [0, 0, 1, 0, 0, self.dt_],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        for iStage in range(1, self.N_):
            xpred = np.matmul(F,xpred)
            Ppred = np.matmul(F,np.matmul(Ppred,F.T))
            self.pred_path_[:, iStage] = xpred[0:3, 0]
            self.pred_pathcov_[:, iStage] = np.array([Ppred[0,0],
                                                      Ppred[1,1],
                                                      Ppred[2,2],
                                                      Ppred[0,1],
                                                      Ppred[1,2],
                                                      Ppred[0,2]])



class pySystem():
    def __init__(self, nQuad, nDynObs, cfg, pr, model, index):

        #declaration of variables
        # timer
        self.time_global_ = 0
        self.time_step_global_ = 0

        ##initialization constructor
        self.nQuad_ = nQuad
        self.nDynObs_ = nDynObs
        self.cfg_ = cfg
        self.dt_ = model["dt"]
        self.N_ = model["N"]
        self.index_ = index
        self.model_ = model

        self.MultiQuad_ = [pyCDrone(iQuad, iQuad, cfg, pr, model, index) for iQuad in range(nQuad)]

        self.MultiDynObs_ = []
        #self.MultiDynObs_ = [pyCDynObs(jObs, cfg, pr, model, index) for jObs in range(nDynObs)] # TODO code pyCDynObs

        self.GraphicCom_ = None
        #self.GraphicCom_ = pyCGraphicCom(true, cfg, nQuad, nDynObs, model["N"]) # TODO pyCGraphycCom

        self.multi_quad_state_ = np.zeros((9, model["nQuad"]))
        self.multi_quad_goal_ = cfg["quad"]["goal"]
        self.multi_quad_input_ = np.zeros((4, model["nQuad"]))
        self.multi_quad_slack_ = np.zeros((2, model["nQuad"]))
        self.multi_quad_mpc_path_ = np.zeros((3, model["N"], model["nQuad"]))
        self.multi_quad_mpc_pathcov_ = np.zeros((6, model["N"], model["nQuad"]))

        self.multi_quad_prep_path_ = np.zeros((3, model["N"], model["nQuad"]))
        self.multi_quad_prep_pathcov_ = np.zeros((6, model["N"], model["nQuad"]))

        self.multi_obs_state_ = np.zeros((6, model["nDynObs"]))
        self.multi_obs_path_ = np.zeros((3, model["N"], model["nDynObs"]))

        self.para_mpc_coll_ = np.concatenate([cfg["quad"]["coll"], cfg["obs"]["coll"]], 1)
        self.para_mpc_weights = np.concatenate([cfg["weightStage"], cfg["weightN"]], 1)

        self.multi_quad_coor_path_ = np.zeros((3, model["N"], model["nQuad"]))
        self.multi_quad_coor_pathcov_ = np.zeros((6, model["N"], model["nQuad"]))

        self.multi_quad_comm_mtx_ = np.zeros((model["nQuad"],model["nQuad"]))

        self.set_evaluation_ = 0

    #def simDynObsMotion(self):
    # publish dyn obs path in simulation mode

    def multiQuadMpcSimStep(self):
        # sequential mpc control and sim one step for the system
        for iQuad in range(self.nQuad_):
            # get estimated state of the ego quad
            self.MultiQuad_[iQuad].getEstimatedSystemState()

            # set configuration parameters
            self.MultiQuad_[iQuad].quad_goal_ = self.multi_quad_goal_[0:4,iQuad:iQuad+1]
            self.MultiQuad_[iQuad].mpc_coll_ = self.para_mpc_coll_
            self.MultiQuad_[iQuad].mpc_weights_ = self.para_mpc_weights

            # get predicted obstacles path
            #self.MultiQuad_[iQuad].getObsPredictedPath() # TODO get obstacles predicted path inside pyCDrone

            # for each quad, get path of other quads
            if self.MultiQuad_[iQuad].modeCoor_ == -1:  #centralized prioritized planning
                self.MultiQuad_[iQuad].quad_path_[:,:,:] = self.multi_quad_coor_path_[:,:,0:iQuad]
            elif self.MultiQuad_[iQuad].modeCoor_== 0:   #centralized sequential planning
                self.MultiQuad_[iQuad].quad_path_[:,:,:] = self.multi_quad_coor_path_[:,:,:]

            else:
                # consider communication
                for iTemp in range(self.nQuad_):
                    if self.multi_quad_comm_mtx_[iQuad, iTemp] == 1: # i requests from j
                        #update the comm info
                        self.MultiQuad_[iQuad].quad_traj_com_[:,:,iTemp] = self.MultiQuad_[iTemp].mpc_traj_  # last comm info
                        #get path info for motion planning
                        self.MultiQuad_[iQuad].quad_path_[:,:,iTemp] = self.multi_quad_mpc_path_[:,:,iTemp]  # all actual quad paths
                                                                                                            # stored here, new (N:N+1)
                                                                                                            #transition is considered to follow constant vel

                    else:
                        #predict other quad based on last comm info and their current state
                        self.MultiQuad_[iQuad].quad_traj_pred_[:,:,iTemp], ifabandon = predictQuadPathFromCom(self.MultiQuad_[iQuad].quad_traj_com_[:,:,iTemp],
                                                                                                              self.multi_quad_state_[0:6,iTemp:iTemp+1],
                                                                                                              self.MultiQuad_[iQuad].time_step_global_,
                                                                                                              self.MultiQuad_[iQuad].dt_,
                                                                                                              self.MultiQuad_[iQuad].quad_traj_pred_tol_) #
                        # get the path info for motion planning
                        self.MultiQuad_[iQuad].quad_path_[:,:,iTemp] = self.MultiQuad_[iQuad].quad_traj_pred_[0:3,:,iTemp]
                        if self.set_evaluation_ == 0:
                            #ignore this path
                            self.MultiQuad_[iQuad].quad_path_[2:3,:,iTemp] = -10*np.ones((1,self.N_))


        #aux1 = time.time()
        ##### part to parallelize #####
        for iQuad in range(self.nQuad_):
            # set online parameters for the MPC
            self.MultiQuad_[iQuad].setOnlineParameters()

            # solve the mpc problem
            self.MultiQuad_[iQuad].solveMPC()
        #print("solving time:", time.time()-aux1)



        for iQuad in range(self.nQuad_):
            # send and execute the control command
            self.MultiQuad_[iQuad].step()
            self.MultiQuad_[iQuad].time_step_global_ += 1

            # communicate the planned mpc path only in centralized planning
            if self.MultiQuad_[iQuad].modeCoor_ == 0 or self.MultiQuad_[iQuad].modeCoor_==-1: # sequential or prioritized
                self.multi_quad_coor_path_[:,:,iQuad] = self.MultiQuad_[iQuad].mpc_Path_

        self.time_step_global_ += 1


    def multiQuadComm(self):
        # for communication with the central system and allow debugging / message passing
        # Quad.pred_path refers to constant velocity predictions
        for iQuad in range(self.nQuad_):

            # path prediction using constant v
            self.MultiQuad_[iQuad].predictPathConstantV()
            self.multi_quad_prep_path_[:,:,iQuad] = self.MultiQuad_[iQuad].pred_path_

            self.multi_quad_mpc_path_[:,0:self.N_-1, iQuad] = self.MultiQuad_[iQuad].mpc_Path_[:,1:self.N_]
            self.multi_quad_mpc_path_[:,self.N_-1:self.N_,iQuad] = self.MultiQuad_[iQuad].mpc_Path_[:,self.N_-1:self.N_] +\
                                                                   self.MultiQuad_[iQuad].mpc_ZPlan_[self.index_["z"]["vel"],self.N_-1:self.N_]*self.dt_

            # the following part is not used when using learned comm. policies
            if self.MultiQuad_[iQuad].modeCoor_ == 1: # path communication (distributed)
                self.multi_quad_coor_path_ = self.multi_quad_mpc_path_

            elif self.MultiQuad_[iQuad].modeCoor_== 2: # path prediction based on constant v
                self.multi_quad_coor_path_ = self.multi_quad_prep_path_


    def getSystemState(self):
        # store system state

        #quad
        for iQuad in range(self.nQuad_):
            self.multi_quad_state_[:,iQuad:iQuad+1] = np.concatenate([self.MultiQuad_[iQuad].pos_real_,
                                                        self.MultiQuad_[iQuad].vel_real_,
                                                        self.MultiQuad_[iQuad].euler_real_],0)
            self.multi_quad_input_[:,iQuad:iQuad+1] = self.MultiQuad_[iQuad].u_body_
            self.multi_quad_slack_[:,iQuad:iQuad+1] = 10*self.MultiQuad_[iQuad].mpc_Zk_[self.index_["z"]["slack"]]
            self.multi_quad_mpc_path_[:,:,iQuad] = self.MultiQuad_[iQuad].mpc_Path_

        #obs
        self.multi_obs_path_ = self.MultiQuad_[self.nQuad_-1].obs_path_
        self.multi_obs_state_[0:3, :] = self.multi_obs_path_[0:3,1,:]
        self.multi_obs_state_[3:6,:] = (self.multi_obs_path_[0:3,1,:]-self.multi_obs_path_[0:3,0,:]) / self.dt_


    #def commWithVisualGui(self): # TODO 1st priority
        # communicate to gui for visualization

    #def createSystemSrvServer(self): # UNNECESSARY, system is directly called from python
        # create a service server

    def stepMultiAgent(self, comm_vector):

        #retrive comm info
        comm_mtx = np.reshape(comm_vector, (self.nQuad_, self.nQuad_))
        self.multi_quad_comm_mtx_ = comm_mtx

        # determine if evaluation environment
        self.set_evaluation_ = self.multi_quad_comm_mtx_[1,1]

        # set quad initial positions and goals
        if self.multi_quad_comm_mtx_[0, 0] == -1:
            self.resetScenario()
        elif self.multi_quad_comm_mtx_[0, 0] == -2:
            self.randomScenario()
        elif self.multi_quad_comm_mtx_[0, 0] == -3:
            self.randomSwapScenario()
        elif self.multi_quad_comm_mtx_[0, 0] == -4:
            self.rotateScenario()
        elif self.multi_quad_comm_mtx_[0, 0] == -5:
            self.circleRandomScenario()

        # communication (message passing inside the system)
        self.multiQuadComm()

        #planning & step
        self.multiQuadMpcSimStep()

        #system states
        self.getSystemState()

        #return
        respData = np.concatenate([self.multi_quad_state_, self.multi_quad_goal_], axis = 0) # TODO: check that some are not concatenates
        flattened_state_goal = respData.T.flatten()

        # optional, comm to visualize
        # TODO visualization with gui

        return flattened_state_goal


    def resetScenario(self):
        # reset the scenario, including quad initial state and goal

        # reset initial state
        rand_idx = np.random.permutation(self.nQuad_) # randomize initial positions
        rand_idx = np.arange(0,self.nQuad_)
        for iQuad in range(nQuad):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3,0] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].vel_real_[0:3,0] = self.cfg_["quadStartVel"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3,1))
            self.MultiQuad_[iQuad].euler_real_[2] = self.cfg_["quadStartPos"][3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # goal
            self.multi_quad_goal_[:,iQuad] = self.cfg_["quadEndPos"][:, rand_idx[iQuad]]

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_, self.MultiQuad_[iQuad].euler_real_], axis = 0)
            z_start = np.zeros((self.model_["nvar"],1))
            z_start[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = x_start
            mpc_plan = matlib.repmat(z_start,1,self.model_["N"])
            # initialize MPC
            self.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)
            # reset belief of other quad trajectories
            for iStage in range(self.model_["N"]):
                self.multi_quad_mpc_path_[:,iStage,iQuad] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
                #.multi_quad_mpc_pathcov_ = np.array([[self.cfg_["quad"]["noise"]["pos"][0, 0]], # TODO useful for chance ctraints, specify in cfg
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][2, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 2]]])

        self.multi_quad_prep_path_ = self.multi_quad_mpc_path_
        #self.multi_quad_prep_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before
        self.multi_quad_coor_path_ = self.multi_quad_mpc_path_
        #self.multi_quad_coor_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before


    def randomScenario(self):
        # random set the scenario, including quad initial state and goal


        xDim = np.array([-self.cfg_["ws"][0]+self.cfg_["quad"]["size"][0], self.cfg_["ws"][0]-self.cfg_["quad"]["size"][0]])
        yDim = np.array(
            [-self.cfg_["ws"][1] + self.cfg_["quad"]["size"][1], self.cfg_["ws"][1] - self.cfg_["quad"]["size"][1]])
        zDim = np.array(
            [self.cfg_["quad"]["size"][2], self.cfg_["ws"][2] - self.cfg_["quad"]["size"][2]])

        quadStartPos, quadStartVel, quadEndPos = scn_random(self.nQuad_, xDim, yDim, zDim)

        self.multi_quad_goal_ = quadEndPos

        for iQuad in range(nQuad):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3] = quadStartPos[0:3, iQuad]
            self.MultiQuad_[iQuad].vel_real_[0:3] = quadStartVel[0:3, iQuad]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
            self.MultiQuad_[iQuad].euler_real_[2] = quadStartPos[3, iQuad]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # goal
            self.multi_quad_goal_[:, iQuad] = self.cfg_["quadEndPos"][:, iQuad]

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_,
                                      self.MultiQuad_[iQuad].euler_real_], axis=0)
            z_start = np.zeros((self.model_["nvar"], 1))
            z_start[self.index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
            mpc_plan = matlib.repmat(z_start, 1, self.model_["N"])
            # initialize MPC
            self.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)
            # reset belief of other quad trajectories
            for iStage in range(self.model_["N"]):
                self.multi_quad_mpc_path_[:, iStage, iQuad] = self.cfg_["quadStartPos"][0:3, iQuad]
                # .multi_quad_mpc_pathcov_ = np.array([[self.cfg_["quad"]["noise"]["pos"][0, 0]], # TODO useful for chance ctraints, specify in cfg
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][2, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 2]]])

        self.multi_quad_prep_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_prep_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before
        self.multi_quad_coor_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_coor_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before



    def randomSwapScenario(self):
        # random set the scenario, including quad initial state and random swap pairs of them


        xDim = np.array(
            [-self.cfg_["ws"][0] + self.cfg_["quad"]["size"][0], self.cfg_["ws"][0] - self.cfg_["quad"]["size"][0]])
        yDim = np.array(
            [-self.cfg_["ws"][1] + self.cfg_["quad"]["size"][1], self.cfg_["ws"][1] - self.cfg_["quad"]["size"][1]])
        zDim = np.array(
            [self.cfg_["quad"]["size"][2], self.cfg_["ws"][2] - self.cfg_["quad"]["size"][2]])

        quadStartPos, quadStartVel, quadEndPos = scn_random(self.nQuad_, xDim, yDim, zDim)

        #self.multi_quad_goal_ = quadEndPos

        for iQuad in range(nQuad):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3] = quadStartPos[0:3, iQuad]
            self.MultiQuad_[iQuad].vel_real_[0:3] = quadStartVel[0:3, iQuad]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
            self.MultiQuad_[iQuad].euler_real_[2] = quadStartPos[3, iQuad]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # goal
            self.multi_quad_goal_[:, iQuad] = self.cfg_["quadEndPos"][:, iQuad]

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_,
                                      self.MultiQuad_[iQuad].euler_real_], axis=0)
            z_start = np.zeros((self.model_["nvar"], 1))
            z_start[self.index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
            mpc_plan = matlib.repmat(z_start, 1, self.model_["N"])
            # initialize MPC
            self.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)
            # reset belief of other quad trajectories
            for iStage in range(self.model_["N"]):
                self.multi_quad_mpc_path_[:, iStage, iQuad] = self.cfg_["quadStartPos"][0:3, iQuad]
                # .multi_quad_mpc_pathcov_ = np.array([[self.cfg_["quad"]["noise"]["pos"][0, 0]], # TODO useful for chance ctraints, specify in cfg
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][2, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 2]]])


        # random swapping pairs of quads
        num_pair = np.floor(self.nQuad_/2)  # number of pairs
        rand_idx = np.random.permutation(self.nQuad_)  # randomize index

        for iPair in range(num_pair):
            self.multi_quad_goal_[:, rand_idx[2*iPair-1]] = quadStartPos[:, rand_idx[2*iPair]]
            self.multi_quad_goal_[:, rand_idx[2*iPair]] = quadStartPos[:, rand_idx[2*iPair-1]]

        self.multi_quad_prep_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_prep_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before
        self.multi_quad_coor_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_coor_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before


    def rotateScenario(self):
        # the scenario of rotation, including quad initial state and goal

        # reset initial state
        rand_idx = np.random.permutation(self.nQuad_)  # randomize index
        for iQuad in range(nQuad):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].vel_real_[0:3] = self.cfg_["quadStartVel"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
            self.MultiQuad_[iQuad].euler_real_[2] = self.cfg_["quadStartPos"][3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_,
                                      self.MultiQuad_[iQuad].euler_real_], axis=0)
            z_start = np.zeros((self.model_["nvar"], 1))
            z_start[self.index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
            mpc_plan = matlib.repmat(z_start, 1, self.model_["N"])
            # initialize MPC
            self.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)
            # reset belief of other quad trajectories
            for iStage in range(self.model_["N"]):
                self.multi_quad_mpc_path_[:, iStage, iQuad] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
                # .multi_quad_mpc_pathcov_ = np.array([[self.cfg_["quad"]["noise"]["pos"][0, 0]], # TODO useful for chance ctraints, specify in cfg
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][2, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 2]]])

        # set goal
        self.multi_quad_goal_ = np.zeros((4,self.nQuad_))
        dir_rand = np.random.random()
        if dir_rand >= 0.5:
            dir = 1
        else:
            dir = -1

        for iQuad in range(nQuad):
            goal_idx = rand_idx[iQuad] + dir*1
            if goal_idx >= self.nQuad_:
                goal_idx = 0
            elif goal_idx < 0:
                goal_idx = self.nQuad_-1

            self.multi_quad_goal_[:, iQuad] = self.cfg_["quadStartPos"][:, goal_idx]

        self.multi_quad_prep_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_prep_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before
        self.multi_quad_coor_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_coor_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before



    def circleRandomScenario(self):
        # random set the scenario, including quad initial state and goal

        quadStartPos, quadStartVel, quadEndPos = scn_circle_random(self.nQuad_, 2.8, 5.4)

        self.multi_quad_goal_ = quadEndPos

        rand_idx = np.random.permutation(self.nQuad_)

        for iQuad in range(self.nQuad_):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3] = quadStartPos[0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].vel_real_[0:3] = quadStartVel[0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
            self.MultiQuad_[iQuad].euler_real_[2] = quadStartPos[3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # goal
            self.multi_quad_goal_[:, iQuad] = quadEndPos[:, rand_idx[iQuad]]

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_,
                                      self.MultiQuad_[iQuad].euler_real_], axis=0)
            z_start = np.zeros((self.model_["nvar"], 1))
            z_start[self.index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
            mpc_plan = matlib.repmat(z_start, 1, self.model_["N"])
            # initialize MPC
            self.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)
            # reset belief of other quad trajectories
            for iStage in range(self.model_["N"]):
                self.multi_quad_mpc_path_[:, iStage, iQuad] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
                # .multi_quad_mpc_pathcov_ = np.array([[self.cfg_["quad"]["noise"]["pos"][0, 0]], # TODO useful for chance ctraints, specify in cfg
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][2, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 1]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][1, 2]],
                #                                         [self.cfg_["quad"]["noise"]["pos"][0, 2]]])

        self.multi_quad_prep_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_prep_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before
        self.multi_quad_coor_path_ = self.multi_quad_mpc_path_
        # self.multi_quad_coor_pathcov_ = self.multi_quad_mpc_pathcov_ # Same as before



if __name__ == '__main__':
    ## Initialization
    nQuad = 12
    nDynObs = 0
    srv_idx = 1  # Not necessary

    #>>> initialize_func >>>>
    application = "basic"
    #getNewSolver = 0 # NOT NECESSARY IF GENERATED FROM MATLAB
    quadExpID = list(range(100))

    # Load problem setup
    #if application == "basic": # It is always basic for the moment
    pr, model, index = basic_setup(nQuad, nDynObs)

    quadStartPos, quadStartVel, quadEndPos = scn_circle(model["nQuad"], 4.0)

    cfg = {
        #running mode
        "quadStartPos": quadStartPos,
        "quadStartVel": quadStartVel,
        "quadEndPos":   quadEndPos,
        "application":  "basic",
        "modeSim":  1,
        "modeCoor": 2,

        # environment boundary, [xmax, ymax, zmax]
        "ws":   np.array([[6.0, 6.0, 3.0]]).T, # m

        "quad":{
            # goal
            "goal": quadEndPos, #[quadEndPos],

            # drone size, collision avoidance parameters
            "size": np.array([[0.3, 0.3, 0.5]]).T,
            "coll": np.array([[10, 1.2, 0.03]]).T,
        }
    }

    # stage weights
    wS = {
        "wp": 0.0,
        "input": 0.1,
        "coll": 0.2,
        "slack": 1e4,
    }
    cfg["weightStage"] = np.array([[wS["wp"]], [wS["input"]], [wS["coll"]], [wS["slack"]]])

    # terminal weights
    wN = {
        "wp":   10,
        "input":    0.0,
        "coll": 0.2,
        "slack":    1e4,
    }
    cfg["weightN"] = np.array([[wN["wp"]], [wN["input"]], [wN["coll"]], [wN["slack"]]])

    # moving obstacles
    cfg["obs"] = {
        "size": np.array([[0.5, 0.5, 0.9]]).T,  #[a, b, c]
        "coll": np.array([[10, 1.2, 0.03]]).T  # lambda, buffer, delta
    }

    # communication with gui
    cfg["ifCommWithGui"] = 0
    cfg["setParaGui"] = 0
    cfg["ifShowQuadHead"] = 1
    cfg["ifShowQuadSize"] = 1
    cfg["ifShowQuadGoal"] = 0
    cfg["ifShowQuadPath"] = 1
    cfg["ifShowQuadCov"] = 0
    cfg["ifShowQuadPathCov"] = 0

    ## Extra running configuration for chance constrained collision avoidance --> NOT NEEDED--> CAN USE MATLAB
    #<<<initialize_func<<<

    # Not necessary for the moment --> can be done through matlab
    #if getNewSolver:
    #    mpc_generator_basic

    # Create multi-robot system
    System = pySystem(nQuad, nDynObs, cfg, pr, model, index)

    # Initialization quad simulated initial state and mpc plan
    for iQuad in range(model["nQuad"]):
        # initialize ros --> NOT NEEDED
        # coordination mode
        System.MultiQuad_[iQuad].modeCoor_ = cfg["modeCoor"]
        # initial state
        System.MultiQuad_[iQuad].pos_real_[0:3] = quadStartPos[0:3, iQuad:iQuad+1]
        System.MultiQuad_[iQuad].vel_real_[0:3] = quadStartVel[0:3, iQuad:iQuad+1]
        System.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3,1))
        System.MultiQuad_[iQuad].euler_real_[2] = quadStartPos[3, iQuad:iQuad+1]
        # for mpc
        x_start = np.concatenate([System.MultiQuad_[iQuad].pos_real_, System.MultiQuad_[iQuad].vel_real_, System.MultiQuad_[iQuad].euler_real_], axis = 0)
        z_start = np.zeros((model["nvar"],1))
        z_start[index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
        mpc_plan = matlib.repmat(z_start, 1, model["N"]) #
        #initialize mpc
        System.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan) 

        # to avoid whom in prioritized planning --> NOT NECESSARY, CHECK MATLAB IMPLEMENTATION IF WE WANT TO ADAPT

#################################### Until here the fundamentally necessary############################################
    # TODO: write testing code --> create a system initializator # DO THIS AS FIRST THING IN THE MORNING - FIRST CHECK WHAT KIND OF MESSAGE IS PASSED AS ACTION
    # (i, j) --> robot i requests from robot j its traj. intention
    n_action = np.ones((nQuad,nQuad)) - np.eye(nQuad)
    n_action[0,0] = -1
    sent_action = n_action.flatten()

    aux2 = time.time()
    for i in range(100):
        print("step:",i)
        aux1 = time.time()
        System.stepMultiAgent(sent_action)
        n_action = np.ones((nQuad,nQuad)) - np.eye(nQuad)
        sent_action = n_action.flatten()
        print("solving time:", time.time() - aux1)

    print("everything's over")
    print("time:", time.time() - aux2)

####################
    # Quad pathcov initialization --> UNNECESSARY (chance constraints)
    """
    for iQuad in range(model["nQuad"]):
        System.multi_quad_state_[0:3,iQuad] = quadStartPos[0:3, iQuad]
        for iStage in range(model["N"]):
            System.multi_quad_mpc_path_[:, iStage, iQuad] = quadStartPos[0:3, iQuad]
            System.multi_quad_mpc_pathcov_[:, iStage, iQuad] = [cfg["quad"]["noise"]["pos"][]]
    """

    # Set moving obstacle objects in simulation mode --> UNNECESSARY (no moving obstacles other than drones)

    # TODO:Initialization graphic communicator
    # initialize ROS
    # set default quad and obs size

    # Create the server --> UNNECESSARY (this is what we want to avoid)

    #









