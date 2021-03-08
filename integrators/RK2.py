import numpy
from dynamics.bebop_dynamics import bebop_dynamics as dynamics

def rk2a_onestep(x0, u, t):
    """Second-order Runge-Kutta method to solve x' = f(x,t) with x(t[0]) = x0.

    USAGE:
        x = rk2a(f, x0, t)

    INPUT:
        f     - function of x and t equal to dx/dt.  x may be multivalued,
                in which case it should a list or a NumPy array.  In this
                case f must return a NumPy array with the same dimension
                as x.
        x0    - the initial condition(s).  Specifies the value of x when
                t = t[0].  Can be either a scalar or a list or NumPy array
                if a system of equations is being solved.
        t     - list or NumPy array of t values to compute solution at.
                t[0] is the the initial condition point, and the difference
                h=t[i+1]-t[i] determines the step size h.

    OUTPUT:
        x     - NumPy array containing solution values corresponding to each
                entry in t array.  If a system is being solved, x will be
                an array of arrays.

    NOTES:
        This version is based on the algorithm presented in "Numerical
        Analysis", 6th Edition, by Burden and Faires, Brooks-Cole, 1997.
    """

    n = len( t )
    #x = numpy.array( [ x0 ] * n )
    x = numpy.array([x0.copy() for i in range(n)])
    for i in range( n - 1 ):
        h = t[i+1] - t[i]
        k1 = h * dynamics( x[i], u ) / 2.0
        x[i+1] = x[i] + h * dynamics(x[i] + k1, u)

    return x[n-1]