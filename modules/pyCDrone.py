
import time
import numpy as np
import numpy.matlib as matlib
import warnings
warnings.simplefilter("always")

from solvers.solver.basic.Basic_Forces_11_20_50.FORCESNLPsolver_basic_11_20_50_py import FORCESNLPsolver_basic_11_20_50_solve as solver
#from solvers.solver.basic.Basic_Forces_11_20_50 import FORCESNLPsolver_basic_11_20_50_py
from integrators.RK2 import rk2a_onestep as RK2
import ray
import psutil
import gc


def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


@ray.remote
def solveMPC_ray(all_parameters, xinit, x0): #Might be some issues with the shape of the vectors
    # call the NLP solver
    #aux1 = time.time()
    problem = {}
    problem['all_parameters'] = np.array(all_parameters)
    problem['xinit'] = np.array(xinit)
    problem['x0'] = np.array(x0)
    OUTPUT, EXITFLAG, INFO = solver(problem)
    #OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_basic_11_20_50_py.FORCESNLPsolver_basic_11_20_50_solve(problem)
    #print("Solving time drone:",time.time()-aux1)
    #auto_garbage_collect(35.0)
    return [OUTPUT, EXITFLAG, INFO]

@ray.remote
def setOnlineParameters_ray(Quad):
    # Set the real-time parameter vector
    # pAll include parameters for all N stage

    # prepare parameters
    envDim = Quad.cfg_["ws"]
    startPos = np.concatenate([Quad.pos_est_, np.array([Quad.euler_est_[2]])], 0)
    wayPoint = Quad.quad_goal_
    egoSize = Quad.size_
    weightStage = Quad.mpc_weights_[:,0]
    weightN = Quad.mpc_weights_[:,1]
    quadSize = Quad.cfg_["quad"]["size"]
    obsSize = Quad.cfg_["obs"]["size"]
    quadColl = Quad.mpc_coll_[0:2,0:1] #lambda, buffer (sigmoid function)
    obsColl = Quad.mpc_coll_[0:2,1:2] #lambda, buffer
    quadPath = Quad.quad_path_
    obsPath = Quad.obs_path_

    # all stage parameters
    pStage = np.zeros((Quad.npar_, 1))
    mpc_pAll_ = matlib.repmat(pStage, Quad.N_, 1)
    for iStage in range(0,Quad.N_):
        #general parameter
        pStage[Quad.index_["p"]["envDim"]] = envDim
        pStage[Quad.index_["p"]["startPos"]] = startPos
        pStage[Quad.index_["p"]["wayPoint"]] = wayPoint
        pStage[Quad.index_["p"]["size"]] = egoSize
        pStage[Quad.index_["p"]["weights"], 0] = weightStage
        # obstacle information, including other quadrotors
        # and moving obstacles, set other quad first
        idx = 0
        for iQuad in range(Quad.nQuad_):
            if iQuad == Quad.id_:
                continue
            else:
                pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["pos"], idx]] = quadPath[:, iStage,iQuad:iQuad+1]
                pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["size"], idx]] = quadSize
                pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["coll"], idx]] = quadColl
                idx = idx + 1

        for jObs in range(Quad.nDynObs_):
            pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["pos"],idx]] = obsPath[:, iStage, jObs:jObs+1]
            pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["size"], idx]] = obsSize
            pStage[Quad.index_["p"]["obsParam"][Quad.index_["p"]["obs"]["coll"], idx]] = obsColl
            idx = idx + 1

        # change the last stage cost term weights
        if iStage == Quad.N_-1:
            pStage[Quad.index_["p"]["weights"], 0] = weightN

        # insert into the all stage parameter
        mpc_pAll_[Quad.npar_ * iStage : Quad.npar_ * (iStage+1)] = pStage

    #auto_garbage_collect(35.0)
    return mpc_pAll_


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
        self.pos_est_ = np.zeros((3,1))
        self.vel_est_ = np.zeros((3,1))
        self.euler_est_ = np.zeros((3,1))
        self.pos_est_cov_ = np.eye(3)
        self.vel_est_cov_ = np.eye(3)
        self.euler_est_cov_ = np.eye(3)

        # running para
        self.quad_goal_ = np.zeros((4,1))
        self.mpc_coll_ = np.zeros((3,2))
        self.mpc_weights_ = np.zeros((4,2))

        # mpc plan
        self.mpc_pAll_ = None
        self.mpc_exitflag_ = None
        self.mpc_info_ = None
        self.mpc_Xk_ = None
        self.mpc_Zk_ = None
        self.mpc_Zk2_ = None
        self.mpc_ZPlan_ = None

        # control input
        self.u_mpc_ = np.zeros((4,1))
        self.u_body_ = np.zeros((4,1))




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
        self.mpc_Xk_ = x_start.copy()
        self.mpc_ZPlan_ = mpc_plan.copy()
        self.mpc_Path_[:,:] = mpc_plan[self.index_["z"]["pos"],:]

    #def initializeROS(self): Not needed for the moment
    # Initialize ROS publishers and subscribers for the quadrotor
    #subs to mocap raw data / bebop 2 est pos,vel,orient / predicted path of moving obstacles
    #pubs to mpc control input cmd_vel, Twist


    #def getObservedSystemState(self) #Not really used at the moment
    #Get measured real-time position and attitude of the drone

    def getEstimatedSystemState(self):
        #always simulated
        assert self.cfg_["modeSim"] == 1
        # set estimated state the same as real one
        self.pos_est_ = np.array(self.pos_real_)
        self.vel_est_ = np.array(self.vel_real_)
        self.euler_est_ = np.array(self.euler_real_)

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
        wayPoint = self.quad_goal_.copy()
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

    def solveMPC_original(self): #Might be some issues with the shape of the vectors
        # Calling the solver to solve the mpc for collision avoidance
        #set initial conditions
        self.mpc_Xk_ = np.concatenate([self.pos_est_, self.vel_est_, self.euler_est_], 0)

        problem = {}
        problem["all_parameters"] = self.mpc_pAll_.copy()
        problem["xinit"] = self.mpc_Xk_.copy()

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
        #aux1 = time.time()
        OUTPUT, EXITFLAG, INFO = solver(problem)
        #OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_basic_11_20_50_py.FORCESNLPsolver_basic_11_20_50_solve(problem)
        #print("Solving time drone:",time.time()-aux1)

        # store solving information
        self.mpc_exitflag_ = EXITFLAG
        self.mpc_info_ = INFO

        # store output
        for iStage in range(self.N_):
            self.mpc_ZPlan_[:,iStage] = OUTPUT["x{0:0=2d}".format(iStage+1)]
            self.mpc_Path_[:,iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"],iStage]
            self.mpc_traj_[0,iStage] = self.time_step_global_
            self.mpc_traj_[1:7, iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"]+self.index_["z"]["vel"], iStage]

        self.mpc_Zk_ = self.mpc_ZPlan_[:,0:1].copy()
        self.mpc_Zk2_ = self.mpc_ZPlan_[:,1:2].copy()


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
            self.u_mpc_[:,:] = self.mpc_Zk_[self.index_["z"]["inputs"]]
        else:
            # if infeasible
            self.u_mpc_ = -0.0 * self.u_mpc_

        # transform u, check the using dynamics model before doing this!
        yaw = self.euler_est_[2]
        self.u_body_ = self.u_mpc_.copy()
        self.u_body_[0] = self.u_mpc_[1]*np.sin(yaw) + self.u_mpc_[0]*np.cos(yaw) #TODO: clarify with Hai
        self.u_body_[1] = self.u_mpc_[1]*np.cos(yaw) + self.u_mpc_[0]*np.sin(yaw) #u_mpc global --> here transform to local
                                                                            # this is only useful if performing real experiments

    def solveMPC_pre(self): #Might be some issues with the shape of the vectors
        # Calling the solver to solve the mpc for collision avoidance
        #set initial conditions
        self.mpc_Xk_ = np.concatenate([self.pos_est_, self.vel_est_, self.euler_est_], 0)
        # prepare initial guess
        # self.mpc_exitflag_ = 0 # for debugging
        if self.mpc_exitflag_ == 1:  # last step mpc feasible
            x0_temp = np.reshape(
                np.concatenate([self.mpc_ZPlan_[:, 1:self.N_], self.mpc_ZPlan_[:, (self.N_ - 1):self.N_]], axis=1).T,
                (self.N_ * self.nvar_, 1))

        else:  # last step mpc infeasible
            x0_temp_stage = np.zeros((self.nvar_, 1))
            x0_temp_stage[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = self.mpc_Xk_
            x0_temp = matlib.repmat(x0_temp_stage, self.N_, 1)

        problem = {}
        problem["all_parameters"] = self.mpc_pAll_.copy()
        problem["xinit"] = self.mpc_Xk_.copy()
        problem["x0"] = x0_temp.copy()

        return problem.copy()


    def solveMPC(self, problem): #Might be some issues with the shape of the vectors
        # call the NLP solver
        #aux1 = time.time()
        OUTPUT, EXITFLAG, INFO = solver(problem)
        #OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_basic_11_20_50_py.FORCESNLPsolver_basic_11_20_50_solve(problem)
        #print("Solving time drone:",time.time()-aux1)

        return [OUTPUT.copy(), EXITFLAG, INFO]


    def solveMPC_pos(self, OUTPUT, EXITFLAG, INFO): #Might be some issues with the shape of the vectors

        # store solving information
        self.mpc_exitflag_ = EXITFLAG
        self.mpc_info_ = INFO

        # store output
        for iStage in range(self.N_):
            self.mpc_ZPlan_[:, iStage] = OUTPUT["x{0:0=2d}".format(iStage + 1)]
            self.mpc_Path_[:, iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"], iStage]
            self.mpc_traj_[0, iStage] = self.time_step_global_
            self.mpc_traj_[1:7, iStage] = self.mpc_ZPlan_[self.index_["z"]["pos"] + self.index_["z"]["vel"], iStage]

        self.mpc_Zk_ = self.mpc_ZPlan_[:, 0:1].copy()
        self.mpc_Zk2_ = self.mpc_ZPlan_[:, 1:2].copy()

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
            # if mpc solved successfully
            self.u_mpc_[:,:] = self.mpc_Zk_[self.index_["z"]["inputs"]]
        else:
            # if infeasible
            self.u_mpc_ = -0.0 * self.u_mpc_

        # transform u, check the using dynamics model before doing this!
        yaw = self.euler_est_[2]
        self.u_body_ = self.u_mpc_
        self.u_body_[0] = self.u_mpc_[1] * np.sin(yaw) + self.u_mpc_[0] * np.cos(yaw)  # TODO: clarify with Hai
        self.u_body_[1] = self.u_mpc_[1] * np.cos(yaw) + self.u_mpc_[0] * np.sin(yaw)  # u_mpc global --> here transform to local
        # this is only useful if performing real experiments

    def step(self):
        # send and execute the control command
        #TODO: we assume we are in simulation, build loop for real experiments

        # simulate one step in simple simulation mode
        # current state and control
        xNow = np.concatenate([self.pos_real_, self.vel_real_, self.euler_real_],0)
        u = self.u_mpc_.copy() # use u_mpc in simulation --> no need to transform to local

        # integrate one step
        xNext = RK2(xNow, u, [0,self.dt_])


        #update the implicit real state
        self.pos_real_[:,:] = xNext[self.index_["x"]["pos"]]
        self.vel_real_[:,:] = xNext[self.index_["x"]["vel"]]
        self.euler_real_[:,:] = xNext[self.index_["x"]["euler"]]

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
