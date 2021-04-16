import time
import numpy as np
import numpy.matlib as matlib



class pyCDynObs():
    def __init__(self, id, cfg, model):
        self.model = model
        # id, configuration
        self.id_ = id
        self.cfg_ = cfg


        # shape, size
        self.size_ = np.zeros((3,1))

        # state
        self.pos_real_ = np.zeros((3,1))
        self.vel_real_ = np.zeros((3,1))

        self.pos_est_ = np.zeros((3,1))
        self.pos_est_cov_ = np.eye(3)
        self.vel_est_ = np.zeros((3,1))
        self.vel_est_cov_ = np.eye(3)

        # predicted state/path
        self.dt_ = model["dt"] # time step
        self.N_ = model["N"] # number of stages
        self.pred_path_ = np.zeros((3, self.N_)) # predicted path
        self.pred_pathcov_ = np.zeros((6, self.N_)) # corresponding covariance
        self.pred_state_ = np.zeros((6, self.N_))
        self.pred_statecov_ = np.zeros((12, self.N_))

        # ROS publisher
        #self.pred_state_pub_
        #self.pred_path_pub_
        #self.pred_pathcov_pub_


    #def initializeROS(self): #TODO publishers for visualization

    def initializeState(self, pos, vel, pos_cov, vel_cov):
        # Initialize obs state
        self.pos_real_[:,:] = pos
        self.vel_real_[:,:] = vel
        self.pos_est_cov_[:,:] = pos_cov
        self.vel_est_cov_[:,:] = vel_cov



    def randomState(self):
        # random generate obs state
        pos_x = -self.cfg_["ws"][0] + 2*self.cfg_["ws"][0]*np.random.rand(1.0)
        pos_y = -0.5*self.cfg_["ws"][1] + 1*self.cfg_["ws"][1]*np.random.rand(1.0)
        pos_z = 0.6 + 0.6*np.random.rand(0.6)

        speed = 0.8 + 0.4*np.random.rand(1.0)
        angle = -np.pi + 2*np.pi*np.random.rand(1.0)
        vel_x = 1.5*speed*np.cos(angle)
        vel_y = 0.5*speed*np.sin(angle)
        vel_z = -0.04 + 0.08*np.random.rand(1.0)

        self.pos_real_ = np.array([[pos_x, pos_y, pos_z]]).T
        self.vel_real_ = np.array([[vel_x, vel_y, vel_z]]).T

    def getEstimatedObsState(self):
        self.pos_est_[:,:] = self.pos_real_
        self.vel_est_[:,:] = self.vel_real_

        if self.cfg_["addObsStateNoise"] == 1:
            dpos = np.zeros((3,1))
            dvel = np.zeros((3,1))
            for i in range(3):
                dpos[i] = np.random.normal(0, np.sqrt(self.pos_est_cov_[i,i]))
                dvel[i] = np.random.normal(0, np.sqrt(self.vel_est_cov_[i,i]))
            self.pos_est_ = self.pos_est_ + dpos
            self.vel_est_ = self.vel_est_ + dvel


    def predictPathConstantV(self):
        # perform path prediction
        self.pred_path_[:,0] = self.pos_est_
        self.pred_pathcov_[:,0:1] = np.array([[self.pos_est_cov_[0,0],
                                               self.pos_est_cov_[1,1],
                                               self.pos_est_cov_[2,2],
                                               self.pos_est_cov_[0,1],
                                               self.pos_est_cov_[1,2],
                                               self.pos_est_cov_[0,2]]]).T
        xpred = np.concatenate([self.pos_est_, self.vel_est_], 0)
        aux0 = np.concatenate([self.pos_est_cov_, np.zeros((3, 3))], 1)
        aux1 = np.concatenate([np.zeros((3, 3)), self.vel_est_cov_], 1)
        Ppred = np.concatenate([aux0, aux1], 0)

        F = np.array([[1, 0, 0, self.dt_, 0, 0],
                      [0, 1, 0, 0, self.dt_, 0],
                      [0, 0, 1, 0, 0, self.dt_],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        for iStage in range(1, self.N_):
            xpred = np.matmul(F, xpred)
            Ppred = np.matmul(F, np.matmul(Ppred, F.T))
            self.pred_path_[:, iStage] = xpred[0:3, 0]
            self.pred_pathcov_[:, iStage] = np.array([Ppred[0, 0],
                                                      Ppred[1, 1],
                                                      Ppred[2, 2],
                                                      Ppred[0, 1],
                                                      Ppred[1, 2],
                                                      Ppred[0, 2]])


    #def sendPath(self):
        # publish the path #TODO necessary for visualization


    def step(self, dt):
        # simulate one step # Simple uniform linear dynamics
        self.pos_real_ = self.pos_real_ + dt*self.vel_real_
        self.vel_real_ = self.vel_real_

