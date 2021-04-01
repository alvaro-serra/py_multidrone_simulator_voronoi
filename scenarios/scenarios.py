import numpy as np

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
        quadEndPos[2,iQuad] = np.array(quadStartPos[2,iQuad])
        quadEndPos[3,iQuad] = np.array(quadStartPos[3,iQuad])

    return quadStartPos, quadStartVel, quadEndPos


def scn_random(nQuad, xDim, yDim, zDim):

    ## quad initial and end positions (yaw)
    quadStartPos = np.zeros((4,nQuad))
    quadStartVel = np.zeros((3,nQuad))
    quadEndPos = np.zeros((4,nQuad))

    ## generate random starting and end position
    resolution = 0.6
    # discrete the workspace
    xL = xDim[1] - xDim[0]
    yL = yDim[1] - yDim[0]
    zL = zDim[1] - zDim[0]

    # number of discrete points in each dimension
    xN = int(np.floor(xL/resolution))
    yN = int(np.floor(yL/resolution))
    zN = int(np.floor(zL/0.6))

    #non-repeating integer sequence
    xidx_rand = np.random.permutation(xN)+1
    yidx_rand = np.random.permutation(yN)+1

    # random starting and end pos
    for iQuad in range(nQuad):
        xN_i = xidx_rand[iQuad]
        yN_i = yidx_rand[iQuad]
        zN_i = np.random.randint(zN-1)+1
        quadStartPos[0, iQuad] = xDim[0] + resolution*xN_i
        quadStartPos[1, iQuad] = yDim[0] + resolution*yN_i
        quadStartPos[2, iQuad] = zDim[0] + resolution*zN_i
        #quadStartPos[2, iQuad] = 2.0

    xidx_rand = np.random.permutation(xN)+1
    yidx_rand = np.random.permutation(yN)+1
    #zidx_rand = np.random.permutation(zN)

    # random starting and end pos
    for iQuad in range(nQuad):
        xN_i = xidx_rand[iQuad]
        yN_i = yidx_rand[iQuad]
        zN_i = np.random.randint(zN-1)+1

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
        quadEndPos[0:2, iQuad] = -np.array(quadStartPos[0:2, iQuad])
        quadEndPos[2, iQuad] = np.array(quadStartPos[2, iQuad])
        quadEndPos[3, iQuad] = np.array(quadStartPos[3, iQuad])

    return quadStartPos, quadStartVel, quadEndPos


def scn_group_swap(nQuad, xDim, yDim, zDim):

    assert nQuad%2 == 0
    assert nQuad%3 == 0 #or nQuad%4==0

    ## quad initial and end positions (yaw)
    quadStartPos = np.zeros((4,nQuad))
    quadStartVel = np.zeros((3,nQuad))
    quadEndPos = np.zeros((4,nQuad))

    ## generate random starting and end position
    resolution = 0.6
    # discrete the workspace
    xL = xDim[1] - xDim[0]
    yL = yDim[1] - yDim[0]
    zL = zDim[1] - zDim[0]

    # number of discrete points in each dimension
    xN = int(np.floor(xL/resolution))
    yN = int(np.floor(yL/resolution))
    zN = int(np.floor(zL/0.6))

    #non-repeating integer sequence
    xidx_rand = np.random.permutation(xN)+1
    yidx_rand = np.random.permutation(yN)+1

    #define first set of initial positions
    pos_increment = [[2,0],[2,2],[2,-2]]
    xcenter = xN/2
    ycenter = yN / 2
    for iQuad in range(nQuad/2):
        # Drone group 1
        positionx = pos_increment[nQuad%3][0]*(1+nQuad/3)
        positiony = pos_increment[nQuad%3][1]
        quadStartPos[0, iQuad] = xDim[0] + resolution * (xcenter+positionx)
        quadStartPos[1, iQuad] = yDim[0] + resolution * (ycenter+positiony)
        quadStartPos[2, iQuad] = 1.6  # flying height
        quadEndPos[0, iQuad] = -quadStartPos[0, iQuad]
        quadEndPos[1, iQuad] = quadStartPos[1, iQuad]
        quadEndPos[2, iQuad] = quadStartPos[2, iQuad]

        # Drone group 2
        quadStartPos[0, nQuad/2 + iQuad] = quadEndPos[0, iQuad]
        quadStartPos[1, nQuad/2 + iQuad] = quadEndPos[1, iQuad]
        quadStartPos[2, nQuad/2 + iQuad] = quadEndPos[2, iQuad]
        quadEndPos[0, nQuad/2 + iQuad] = -quadStartPos[0, nQuad/2 + iQuad]
        quadEndPos[1, nQuad/2 + iQuad] = quadStartPos[1, nQuad/2 + iQuad]
        quadEndPos[2, nQuad/2 + iQuad] = quadStartPos[2, nQuad/2 + iQuad]


    return quadStartPos, quadStartVel, quadEndPos