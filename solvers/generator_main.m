% Script for simulating multi-quad collision avoidance using mpc

%% Clean workspace
clear
close all
rosshutdown
clearvars
clearvars -global
clc


%% Initialization
nQuad   = 4;            % number of quadrotors
nDynObs = 0;            % number of moving obstacles
srv_idx = 2;            % idx of the service
initialize_func;


%% Generate solver
mpc_generator_basicVoronoi;