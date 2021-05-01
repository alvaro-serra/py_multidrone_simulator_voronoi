function ineq = mpc_nonlinIneq_basic(z, p, nObs)

    % define nonlinear inequalities for mpc

    global index                            % global index information
    
    %% obtaining necessary information
    % environment dim
    env_dim     =   p(index.p.envDim);      % [xdim, ydim, zdim]
    % ego mav 
    ego_pos     =   z(index.z.pos);         % current stage position [x, y, z]
    ego_size    =   p(index.p.size);        % size
    ego_startPos =  p(index.p.startPos(1:3));    % initial stage position % CHANGED
    % slacks
    slack_env   =   z(index.z.slack(1));
    slack_coll  =   z(index.z.slack(2));

    %% environment boundary constraint
    cons_env    =   [ego_pos(1)/env_dim(1); ego_pos(2)/env_dim(2); ego_pos(3)/env_dim(3)] + slack_env;
    
    %% collision avoidance constraints
    cons_coll   =   [];
    cons_voronoi = [];
    for jObs = 1 : nObs
        % obtain obstacle information
        p_obs = p(index.p.obsParam(:, jObs));   % parameters of the obstacle
        obs_pos  = p_obs(index.p.obs.pos);      % position
        obs_size = p_obs(index.p.obs.size);     % size
        obs_comm = p_obs(index.p.obs.comm);% communication with obs % CHANGED
        obs_startPos = p_obs(index.p.obs.startPos);% position at the first stage % CHANGED
        % approximated minkovski sum (ellipsoid)
        a = ego_size(1) + obs_size(1);
        b = ego_size(2) + obs_size(2);
        c = ego_size(3) + obs_size(3);
        % collision avoidance constraint, d^2 - 1 + slack >= 0, (slack >= 0)
        d = ego_pos - obs_pos;                  % relative position
        cons_obs = sqrt(d(1)^2/a^2 + d(2)^2/b^2 + d(3)^2/c^2) - 1 + slack_coll;
        % voronoi cells collision avoidance constraint, n*d/rshift -1 + slack + 1000*comm >= 0, (slack >= 0) % CHANGED
        pmid_pos = (ego_startPos + obs_startPos)/2;
        daux = obs_startPos - ego_startPos;
        normal = daux / (daux(1)^2 +daux(2)^2 +daux(3)^2)^(1/2);
        dmid = pmid_pos - ego_pos;
        normdmid = (dmid(1)^2 +dmid(2)^2 +dmid(3)^2)^(1/2);
        %rshift = ego_size(1) + (ego_size(3)-ego_size(1))*sin(acos([0;0;1]'*dmid/normdmid));
        rshift = ego_size(3);
        cons_vor_obs = normal'*dmid/rshift -1 + slack_coll + 1000*obs_comm;
        % add for all obstacles
        cons_coll = [cons_coll; cons_obs];
        cons_voronoi = [cons_voronoi; cons_vor_obs];
    end

    %% combine inequality constraints
    ineq = [cons_env; cons_coll; cons_voronoi];
end