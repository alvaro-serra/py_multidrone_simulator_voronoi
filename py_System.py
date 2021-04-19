import numpy as np
import numpy.matlib as matlib
import time

from modules.pyCSystem import pyCSystem
from scenarios.scenarios import scn_circle

import ray
import psutil
import gc


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

def createMultiDroneSystem(nQuad = 12, nDynObs = 0):
    ## Initialization
    nQuad = nQuad
    nDynObs = nDynObs

    # >>> initialize_func >>>>
    application = "basic"
    # getNewSolver = 0 # NOT NECESSARY IF GENERATED FROM MATLAB
    quadExpID = list(range(100))

    # Load problem setup
    # if application == "basic": # It is always basic for the moment
    pr, model, index = basic_setup(nQuad, nDynObs)

    quadStartPos, quadStartVel, quadEndPos = scn_circle(model["nQuad"], 4.0)

    cfg = {
        # running mode
        "quadStartPos": quadStartPos,
        "quadStartVel": quadStartVel,
        "quadEndPos": quadEndPos,
        "application": "basic",
        "modeSim": 1,
        "modeCoor": 2,

        # environment boundary, [xmax, ymax, zmax]
        "ws": np.array([[6.0, 6.0, 3.0]]).T,  # m

        "quad": {
            # goal
            "goal": quadEndPos,  # [quadEndPos],

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
        "wp": 10,
        "input": 0.0,
        "coll": 0.2,
        "slack": 1e4,
    }
    cfg["weightN"] = np.array([[wN["wp"]], [wN["input"]], [wN["coll"]], [wN["slack"]]])

    # moving obstacles
    cfg["obs"] = {
        "size": np.array([[0.5, 0.5, 0.9]]).T,  # [a, b, c]
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
    cfg["quad"]["coll"][2] = 0.03
    cfg["obs"]["coll"][2] = 0.03
    #cfg["quad"]["deltaAux"]
    #cfg["obs"]["deltaAux"]
    #cfg["quad"]["Mahalanobis"]
    # cfg["obs"]["Mahalanobis"]
    ## default added noise to the quad
    ## quad
    cfg["addQuadStateNoise"] = 0
    if cfg["application"] == "chance" or cfg["application"] == "chance_slack":
        cfg["addQuadStateNoise"] = 1

    cfg["quad"]["noise"] = {}
    cfg["quad"]["noise"]["pos"] = np.diag(np.array([0.06, 0.06, 0.06]) ** 2)
    cfg["quad"]["noise"]["vel"] = np.diag(np.array([0.01, 0.01, 0.01]) ** 2)
    cfg["quad"]["noise"]["euler"] = np.diag(np.deg2rad([0.5, 0.5, 0.0]) ** 2)

    ## obs
    cfg["addObsStateNoise"] = 1
    cfg["obs"]["noise"] = {}
    cfg["obs"]["noise"]["pos"] = np.diag(np.array([0.04, 0.04, 0.04]) ** 2)
    cfg["obs"]["noise"]["vel"] = np.diag(np.array([0.01, 0.01, 0.01]) ** 2)

    # for extra visualization
    cfg["quadPathCovShowNum"] = 5

    # <<<initialize_func<<<

    # Not necessary for the moment --> can be done through matlab
    # if getNewSolver:
    #    mpc_generator_basic

    # Create multi-robot system
    System = pyCSystem(nQuad, nDynObs, cfg, pr, model, index)

    # Initialization quad simulated initial state and mpc plan
    for iQuad in range(model["nQuad"]):
        # initialize ros --> NOT NEEDED
        # coordination mode
        System.MultiQuad_[iQuad].modeCoor_ = cfg["modeCoor"]
        # initial state
        System.MultiQuad_[iQuad].pos_real_[0:3] = quadStartPos[0:3, iQuad:iQuad + 1]
        System.MultiQuad_[iQuad].vel_real_[0:3] = quadStartVel[0:3, iQuad:iQuad + 1]
        System.MultiQuad_[iQuad].euler_real_[0:3] = np.zeros((3, 1))
        System.MultiQuad_[iQuad].euler_real_[2] = quadStartPos[3, iQuad:iQuad + 1]
        # for mpc
        x_start = np.concatenate([System.MultiQuad_[iQuad].pos_real_, System.MultiQuad_[iQuad].vel_real_,
                                  System.MultiQuad_[iQuad].euler_real_], axis=0)
        z_start = np.zeros((model["nvar"], 1))
        z_start[index["z"]["pos"] + index["z"]["vel"] + index["z"]["euler"]] = x_start
        mpc_plan = matlib.repmat(z_start, 1, model["N"])  #
        # initialize mpc
        System.MultiQuad_[iQuad].initializeMPC(x_start, mpc_plan)

    # to avoid whom in prioritized planning --> NOT NECESSARY, CHECK MATLAB IMPLEMENTATION IF WE WANT TO ADAPT
    #################################### Until here the fundamentally necessary ############################################
    #################################### ADDITIONAL CONTENT ###############################################################
    # Quad pathcov initialization --> UNNECESSARY (chance constraints)

    for iQuad in range(model["nQuad"]):
        System.multi_quad_state_[0:3,iQuad] = quadStartPos[0:3, iQuad]
        for iStage in range(model["N"]):
            System.multi_quad_mpc_path_[:, iStage, iQuad] = quadStartPos[0:3, iQuad]
            System.multi_quad_mpc_pathcov_[:, iStage, iQuad] = np.array([cfg["quad"]["noise"]["pos"][0,0], cfg["quad"]["noise"]["pos"][1,1],
                                                                          cfg["quad"]["noise"]["pos"][2,2], cfg["quad"]["noise"]["pos"][0,1],
                                                                          cfg["quad"]["noise"]["pos"][1,2], cfg["quad"]["noise"]["pos"][0,2]])
    System.multi_quad_prep_path_[:,:,:] = System.multi_quad_mpc_path_
    System.multi_quad_prep_pathcov_[:,:,:] = System.multi_quad_mpc_pathcov_
    System.multi_quad_coor_path_[:,:,:] = System.multi_quad_mpc_path_
    System.multi_quad_coor_pathcov_[:,:,:] = System.multi_quad_mpc_pathcov_


    # TODO Set moving obstacle objects in simulation mode --> used in information gathering settings.
    if cfg["modeSim"] == 1:
        for jObs in range(model["nDynObs"]):
            #System.MultiDynObs_[jObs].initializeROS()
            System.MultiDynObs_[jObs].initializeState(np.zeros((3,1)), np.zeros((3,1)),cfg['obs']['noise']['pos'],cfg['obs']['noise']['vel'])
            System.MultiDynObs_[jObs].randomState()



    # TODO:Initialization graphic communicator
    # initialize ROS
    # set default quad and obs size

    #n_action = np.zeros((nQuad, nQuad))
    # n_action = np.zeros((nQuad,nQuad))
    #n_action[0, 0] = -1
    #sent_action = n_action.flatten()
    #System.stepMultiAgent(sent_action)

    return System

def collision_check(quad_mat, quad1, quad2):
    pos_quad_1 = quad_mat[quad1][0:3]
    pos_quad_2 = quad_mat[quad2][0:3]
    dist = np.linalg.norm(pos_quad_2 - pos_quad_1)
    #print(dist)
    return True if dist <= 0.6 else False

def dist2goal(quad_mat, quad1):
    pos_quad_1 = quad_mat[quad1][0:3]
    pos_goal = quad_mat[quad1][9:12]
    return np.linalg.norm(pos_goal - pos_quad_1)

def norm_collision_check(quad_mat, quad1, quad2):
    pos_quad_1 = quad_mat[quad1][0:3]
    pos_quad_2 = quad_mat[quad2][0:3]
    dif = pos_quad_2 - pos_quad_1
    normdist = np.sqrt(dif[0] ** 2 / 0.6 ** 2 + dif[1] ** 2 / 0.6 ** 2 + dif[2] ** 2)
    return True if normdist <= 0.99 else False

def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return


if __name__ == '__main__':

    nQuad = 12
    nDynObs = 0
    n_episodes = 1
    ray.init(local_mode=False, log_to_driver=False)
    System = createMultiDroneSystem(nQuad=nQuad, nDynObs=nDynObs)
    # (i, j) --> robot i requests from robot j its traj. intention
    n_action = np.ones((nQuad,nQuad)) - np.eye(nQuad)
    #n_action = np.zeros((nQuad,nQuad))
    scenario = -1
    n_action[0,0] = scenario
    sent_action = n_action.flatten()
    n_timesteps = 100

    collisions = 0
    aux2 = time.time()
    episode = 1
    success_rate = 0
    collision_bool = 0
    for i in range(n_timesteps*n_episodes):
        print("episode:", episode)
        print("step:",i)
        aux1 = time.time()
        obs = System.stepMultiAgent(sent_action)
        n_action = np.ones((nQuad,nQuad)) - np.eye(nQuad)
        #n_action = np.zeros((nQuad, nQuad))
        n_action[1, 1] = -1

        if i%n_timesteps == n_timesteps-1:
            n_action[0, 0] = scenario
            success_rate += 1-collision_bool
            collision_bool = 0
            episode += 1

        sent_action = n_action.flatten()
        aux = np.array(obs).reshape(nQuad,13)
        mat_info_array = list(aux[:, 0:13])


        # collision check
        for iQuad1 in range(nQuad):
            for iQuad2 in range(nQuad):
                if iQuad1 == iQuad2:
                    continue
                if norm_collision_check(mat_info_array, iQuad1, iQuad2):
                    print('COLLISION!!')
                    collision_bool = 1
                    collisions += 0.5
            print("distance to goal:", dist2goal(mat_info_array, iQuad1))
        print("step time:", time.time() - aux1)
        print("memory being used:", psutil.virtual_memory().percent)
        #auto_garbage_collect(35.0)

    print("collisions:",collisions)
    print("success rate:", success_rate)
    print("time:", time.time() - aux2)
    print("everything's over")
    ray.shutdown()










