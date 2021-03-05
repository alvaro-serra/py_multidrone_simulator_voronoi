
import numpy as np
import numpy.matlib as matlib

from modules.pyCDrone import pyCDrone, solveMPC_ray, setOnlineParameters_ray
from scenarios.scenarios import scn_circle_random, scn_circle, scn_random
from utils.utils import predictQuadPathFromCom, predictStateConstantV
import ray
import time

from solvers.solver.basic.Basic_Forces_11_20_50.FORCESNLPsolver_basic_11_20_50_py import FORCESNLPsolver_basic_11_20_50_solve as solver


"""
def solveMPC(all_parameters, xinit, x0): #Might be some issues with the shape of the vectors
    # call the NLP solver
    #aux1 = time.time()
    problem = {}
    problem['all_parameters'] = all_parameters
    problem['xinit'] = xinit
    problem['x0'] = x0
    OUTPUT, EXITFLAG, INFO = solver(problem)
    #OUTPUT, EXITFLAG, INFO = FORCESNLPsolver_basic_11_20_50_py.FORCESNLPsolver_basic_11_20_50_solve(problem)
    #print("Solving time drone:",time.time()-aux1)
    return [OUTPUT.copy(), EXITFLAG, INFO]
"""



class pyCSystem():
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
        #aux1 = time.time()
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



        #print("             exchange of information from central to drones:", time.time() - aux1)

        ##### parallelized part #####
        #aux1 = time.time()
        refs_setop = [setOnlineParameters_ray.remote(self.MultiQuad_[iQuad]) for iQuad in range(self.nQuad_)]
        multiquad_mpc_pAll_ = ray.get(refs_setop)
        for iQuad in range(self.nQuad_):
            # set online parameters for the MPC
            self.MultiQuad_[iQuad].mpc_pAll_ = multiquad_mpc_pAll_[iQuad]

        #print("             set online parameters:", time.time() - aux1)


        #aux1 = time.time()
        problems = [self.MultiQuad_[iQuad].solveMPC_pre() for iQuad in range(self.nQuad_)]
        #print("             pre mpc:", time.time() - aux1)

        #aux1 = time.time()
        #results = [solveMPC(problem['all_parameters'], problem["xinit"], problem['x0']) for problem in problems]
        refs = [solveMPC_ray.remote(problem['all_parameters'], problem["xinit"], problem['x0']) for problem in problems]
        results = ray.get(refs)

        #print("             solving mpc:", time.time() - aux1)

        #aux1 = time.time()
        for iQuad in range(self.nQuad_):
            # save values from the mpc problem
            self.MultiQuad_[iQuad].solveMPC_pos(results[iQuad][0], results[iQuad][1], results[iQuad][2])

        #print("             pos mpc:", time.time() - aux1)



        #aux1 = time.time()
        for iQuad in range(self.nQuad_):
            # send and execute the control command
            self.MultiQuad_[iQuad].step()
            self.MultiQuad_[iQuad].time_step_global_ += 1

            # communicate the planned mpc path only in centralized planning
            if self.MultiQuad_[iQuad].modeCoor_ == 0 or self.MultiQuad_[iQuad].modeCoor_==-1: # sequential or prioritized
                self.multi_quad_coor_path_[:,:,iQuad] = self.MultiQuad_[iQuad].mpc_Path_

        #print("             simulating and advancing the timestep:", time.time() - aux1)

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

        #aux1 = time.time()
        #retrive comm info
        comm_mtx = np.reshape(comm_vector, (self.nQuad_, self.nQuad_))
        self.multi_quad_comm_mtx_ = comm_mtx
        #print("retrieve comm info time:", time.time() - aux1)

        # determine if evaluation environment
        self.set_evaluation_ = self.multi_quad_comm_mtx_[1,1]

        #aux1 = time.time()
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
        #print("reset scenario:", time.time() - aux1)

        #aux1 = time.time()
        # communication (message passing inside the system)
        self.multiQuadComm()
        #print("communication on past computed trajs.:", time.time() - aux1)

        #aux1 = time.time()
        #planning & step
        self.multiQuadMpcSimStep()
        #print("planning & step time:", time.time() - aux1)

        #aux1 = time.time()
        #system states
        self.getSystemState()
        #print("get system states time:", time.time() - aux1)

        #return
        respData = np.concatenate([self.multi_quad_state_, self.multi_quad_goal_], axis = 0) # TODO: check that some are not concatenates
        flattened_state_goal = respData.T.flatten()

        # optional, comm to visualize
        # TODO visualization with gui

        return flattened_state_goal

    def resetScenario(self):
        # reset the scenario, including quad initial state and goal

        # reset initial state
        #rand_idx = np.random.permutation(self.nQuad_) # randomize initial positions
        rand_idx = np.arange(0,self.nQuad_) # FOR DEBUGGING
        for iQuad in range(self.nQuad_):
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

        for iQuad in range(self.nQuad_):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3,0] = quadStartPos[0:3, iQuad]
            self.MultiQuad_[iQuad].vel_real_[0:3,0] = quadStartVel[0:3, iQuad]
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
            z_start[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = x_start
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

        for iQuad in range(self.nQuad_):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3,0] = quadStartPos[0:3, iQuad]
            self.MultiQuad_[iQuad].vel_real_[0:3,0] = quadStartVel[0:3, iQuad]
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
            z_start[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = x_start
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
        num_pair = int(np.floor(self.nQuad_/2))  # number of pairs
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
        for iQuad in range(self.nQuad_):
            # initial state
            self.MultiQuad_[iQuad].pos_real_[0:3,0] = self.cfg_["quadStartPos"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].vel_real_[0:3,0] = self.cfg_["quadStartVel"][0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
            self.MultiQuad_[iQuad].euler_real_[2] = self.cfg_["quadStartPos"][3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].pos_est_ = self.MultiQuad_[iQuad].pos_real_
            self.MultiQuad_[iQuad].vel_est_ = self.MultiQuad_[iQuad].vel_real_
            self.MultiQuad_[iQuad].euler_est_ = self.MultiQuad_[iQuad].euler_real_

            # for mpc
            x_start = np.concatenate([self.MultiQuad_[iQuad].pos_real_, self.MultiQuad_[iQuad].vel_real_,
                                      self.MultiQuad_[iQuad].euler_real_], axis=0)
            z_start = np.zeros((self.model_["nvar"], 1))
            z_start[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = x_start
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

        for iQuad in range(self.nQuad_):
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
            self.MultiQuad_[iQuad].pos_real_[0:3,0] = quadStartPos[0:3, rand_idx[iQuad]]
            self.MultiQuad_[iQuad].vel_real_[0:3,0] = quadStartVel[0:3, rand_idx[iQuad]]
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
            z_start[self.index_["z"]["pos"] + self.index_["z"]["vel"] + self.index_["z"]["euler"]] = x_start
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

