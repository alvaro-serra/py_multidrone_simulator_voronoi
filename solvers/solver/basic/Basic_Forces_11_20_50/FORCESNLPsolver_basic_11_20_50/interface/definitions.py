import numpy
import ctypes

name = "FORCESNLPsolver_basic_11_20_50"
requires_callback = True
lib = "lib/libFORCESNLPsolver_basic_11_20_50.so"
lib_static = "lib/libFORCESNLPsolver_basic_11_20_50.a"
c_header = "include/FORCESNLPsolver_basic_11_20_50.h"

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  9,   1),    9),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (300,   1),  300),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, (3000,   1), 3000)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x02"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x03"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x04"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x05"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x06"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x07"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x08"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x09"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x10"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x11"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x12"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x13"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x14"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x15"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x16"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x17"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x18"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x19"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15),
 ("x20"                 , ""      , ""               , ctypes.c_double, numpy.float64,     ( 15,),   15)]

# Info Struct Fields
info = \
[('it', ctypes.c_int),
('it2opt', ctypes.c_int),
('res_eq', ctypes.c_double),
('res_ineq', ctypes.c_double),
('rsnorm', ctypes.c_double),
('rcompnorm', ctypes.c_double),
('pobj', ctypes.c_double),
('dobj', ctypes.c_double),
('dgap', ctypes.c_double),
('rdgap', ctypes.c_double),
('mu', ctypes.c_double),
('mu_aff', ctypes.c_double),
('sigma', ctypes.c_double),
('lsit_aff', ctypes.c_int),
('lsit_cc', ctypes.c_int),
('step_aff', ctypes.c_double),
('step_cc', ctypes.c_double),
('solvetime', ctypes.c_double),
('fevalstime', ctypes.c_double)
]