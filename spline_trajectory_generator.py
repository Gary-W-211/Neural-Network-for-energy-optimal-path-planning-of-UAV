"""
Trajectory Generator - Using Spline Method

This module provides a spline-based trajectory generator for creating smooth drone trajectories.
Unlike the original polynomial method, this implementation uses splines to generate more flexible trajectories.

Author: Based on original code by Mark W. Mueller
"""

import numpy as np
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.optimize import minimize

class SplineTrajectory:
    """Single-axis spline trajectory
    
    This class implements a spline-based single-axis trajectory generator. It can be used to generate
    smooth trajectories that satisfy position, velocity, and acceleration constraints.
    """
    
    def __init__(self, pos0, vel0, acc0):
        """Initialize trajectory with starting state"""
        self._p0 = pos0
        self._v0 = vel0
        self._a0 = acc0
        self._pf = 0
        self._vf = 0
        self._af = 0
        self._spline = None
        self._cost = float("inf")
        self.reset()
        
    def reset(self):
        """Reset trajectory parameters"""
        self._cost = float("inf")
        self._accGoalDefined = False
        self._velGoalDefined = False
        self._posGoalDefined = False
        self._spline = None
        self._time_points = None
        self._control_points = None
    
    def set_goal_position(self, posf):
        """Set the goal position"""
        self._posGoalDefined = True
        self._pf = posf
        
    def set_goal_velocity(self, velf):
        """Set the goal velocity"""
        self._velGoalDefined = True
        self._vf = velf
        
    def set_goal_acceleration(self, accf):
        """Set the goal acceleration"""
        self._accGoalDefined = True
        self._af = accf
        
    def generate(self, Tf, num_points=10):
        """Generate a trajectory with duration Tf
        
        Args:
            Tf: Trajectory duration
            num_points: Number of control points for generating the spline
        """
        # Define time points
        self._time_points = np.linspace(0, Tf, num_points)
        
        # Initialize constraint list
        constraints = []
        
        # If endpoint position is defined
        if self._posGoalDefined and self._velGoalDefined and self._accGoalDefined:
            # Use fully constrained spline interpolation method
            self._generate_fully_constrained_trajectory(Tf, num_points)
        elif self._posGoalDefined:
            # Position-only constrained optimization problem
            self._generate_position_constrained_trajectory(Tf, num_points)
        elif self._velGoalDefined:
            # Velocity-only constrained optimization problem
            self._generate_velocity_constrained_trajectory(Tf, num_points)
        elif self._accGoalDefined:
            # Acceleration-only constrained optimization problem
            self._generate_acceleration_constrained_trajectory(Tf, num_points)
        else:
            # No constraints, maintain initial trajectory
            self._control_points = np.linspace(self._p0, self._p0, num_points)
            self._spline = CubicSpline(self._time_points, self._control_points, 
                                      bc_type=((1, self._v0), (1, 0)))
        
        # Calculate trajectory cost (integral of squared jerk)
        self._calculate_cost(Tf)
    
    def _generate_fully_constrained_trajectory(self, Tf, num_points):
        """Generate a fully constrained trajectory (position, velocity, acceleration all constrained at endpoints)"""
        # Use spline method to generate trajectory satisfying all boundary conditions
        # Here we use 5 key points: start, end, and 3 intermediate points
        t = np.array([0, Tf/4, Tf/2, 3*Tf/4, Tf])
        
        # Initial and final states
        x0 = self._p0
        v0 = self._v0
        a0 = self._a0
        xf = self._pf
        vf = self._vf
        af = self._af
        
        # Initial guess for middle points (linear interpolation)
        middle_points = np.linspace(x0, xf, 5)[1:-1]
        
        # Define optimization objective: minimize the integral of squared jerk
        def objective(middle):
            # Construct control points
            control_points = np.array([x0, middle[0], middle[1], middle[2], xf])
            # Create spline
            spline = CubicSpline(t, control_points, 
                                bc_type=((2, a0), (2, af)))
            
            # Check velocity boundary conditions
            v_start = spline(0, 1)
            v_end = spline(Tf, 1)
            
            # Penalize solutions that don't satisfy velocity constraints
            vel_penalty = 100 * ((v_start - v0)**2 + (v_end - vf)**2)
            
            # Calculate integral of squared jerk
            t_eval = np.linspace(0, Tf, 100)
            jerk = spline(t_eval, 3)
            jerk_squared_integral = np.trapz(jerk**2, t_eval)
            
            return jerk_squared_integral + vel_penalty
        
        # Constraints: ensure velocities at endpoints match requirements
        def constraint_v_start(middle):
            control_points = np.array([x0, middle[0], middle[1], middle[2], xf])
            spline = CubicSpline(t, control_points, 
                               bc_type=((2, a0), (2, af)))
            return spline(0, 1) - v0
        
        def constraint_v_end(middle):
            control_points = np.array([x0, middle[0], middle[1], middle[2], xf])
            spline = CubicSpline(t, control_points, 
                               bc_type=((2, a0), (2, af)))
            return spline(Tf, 1) - vf
        
        constraints = [
            {'type': 'eq', 'fun': constraint_v_start},
            {'type': 'eq', 'fun': constraint_v_end}
        ]
        
        # Run optimization
        result = minimize(objective, middle_points, method='SLSQP', 
                         constraints=constraints, options={'maxiter': 100})
        
        # Construct final spline using optimization result
        optimal_middle = result.x
        self._control_points = np.array([x0, optimal_middle[0], optimal_middle[1], 
                                       optimal_middle[2], xf])
        self._time_points = t
        self._spline = CubicSpline(t, self._control_points, 
                                 bc_type=((2, a0), (2, af)))
    
    def _generate_position_constrained_trajectory(self, Tf, num_points):
        """Generate a trajectory with only position constraints"""
        # Simply use start and end positions, and start velocity and acceleration
        t = np.array([0, Tf])
        control_points = np.array([self._p0, self._pf])
        
        # Use natural spline, but specify start velocity and acceleration
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((1, self._v0), (2, 0)))
        self._time_points = t
        self._control_points = control_points
    
    def _generate_velocity_constrained_trajectory(self, Tf, num_points):
        """Generate a trajectory with only velocity constraints"""
        # Use start point and one control point
        t = np.array([0, Tf])
        
        # Calculate end position that satisfies velocity constraints
        end_pos = self._p0 + self._v0 * Tf + 0.5 * (self._vf - self._v0) * Tf
        control_points = np.array([self._p0, end_pos])
        
        # Create spline, specifying velocities at both ends
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((1, self._v0), (1, self._vf)))
        self._time_points = t
        self._control_points = control_points
    
    def _generate_acceleration_constrained_trajectory(self, Tf, num_points):
        """Generate a trajectory with only acceleration constraints"""
        # Use start point and one control point
        t = np.array([0, Tf])
        
        # Calculate end position that satisfies acceleration constraints
        end_pos = self._p0 + self._v0 * Tf + 0.5 * self._a0 * Tf**2 + (1/6) * (self._af - self._a0) * Tf**2
        control_points = np.array([self._p0, end_pos])
        
        # Create spline, specifying accelerations at both ends
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((2, self._a0), (2, self._af)))
        self._time_points = t
        self._control_points = control_points
    
    def _calculate_cost(self, Tf):
        """Calculate trajectory cost (integral of squared jerk)"""
        if self._spline is None:
            self._cost = float("inf")
            return
        
        # Evaluate jerk over entire time range
        t_eval = np.linspace(0, Tf, 100)
        jerk = self._spline(t_eval, 3)  # third derivative is jerk
        
        # Use trapezoidal rule to calculate integral
        self._cost = np.trapz(jerk**2, t_eval)
    
    def get_position(self, t):
        """Return position at time t"""
        if self._spline is None:
            return self._p0
        return float(self._spline(t))
    
    def get_velocity(self, t):
        """Return velocity at time t"""
        if self._spline is None:
            return self._v0
        return float(self._spline(t, 1))
    
    def get_acceleration(self, t):
        """Return acceleration at time t"""
        if self._spline is None:
            return self._a0
        return float(self._spline(t, 2))
    
    def get_jerk(self, t):
        """Return jerk at time t"""
        if self._spline is None:
            return 0.0
        return float(self._spline(t, 3))
    
    def get_cost(self):
        """Return the total cost of the trajectory"""
        return self._cost
    
    def get_min_max_acc(self, t1, t2):
        """Return minimum and maximum accelerations between t1 and t2"""
        if self._spline is None:
            return (self._a0, self._a0)
        
        # Sample enough points between t1 and t2 to find extrema
        t_samples = np.linspace(t1, t2, 50)
        acc_samples = self._spline(t_samples, 2)
        
        return (float(np.min(acc_samples)), float(np.max(acc_samples)))
    
    def get_max_jerk_squared(self, t1, t2):
        """Return maximum squared jerk between t1 and t2"""
        if self._spline is None:
            return 0.0
        
        # Sample enough points between t1 and t2 to find extrema
        t_samples = np.linspace(t1, t2, 50)
        jerk_samples = self._spline(t_samples, 3)
        
        return float(np.max(jerk_samples**2))


class SplineTrajectoryGenerator:
    """Spline Trajectory Generator
    
    Uses spline methods to generate 3D trajectories that satisfy start and end constraints.
    Can test feasibility with respect to input constraints (thrust/body rates) and 
    state constraints (position).
    """
    
    def __init__(self, pos0, vel0, acc0, gravity):
        """Initialize trajectory generator
        
        Args:
            pos0: Initial position [x, y, z]
            vel0: Initial velocity [vx, vy, vz]
            acc0: Initial acceleration [ax, ay, az]
            gravity: Gravity acceleration [gx, gy, gz]
        """
        self._axis = [SplineTrajectory(pos0[i], vel0[i], acc0[i]) for i in range(3)]
        self._grav = gravity
        self._tf = None
        self.reset()
    
    def reset(self):
        """Reset trajectory generator"""
        for i in range(3):
            self._axis[i].reset()
    
    def set_goal_position(self, pos):
        """Set goal position
        
        Args:
            pos: Goal position [x, y, z], can contain None for unconstrained axes
        """
        for i in range(3):
            if pos[i] is not None:
                self.set_goal_position_in_axis(i, pos[i])
    
    def set_goal_velocity(self, vel):
        """Set goal velocity
        
        Args:
            vel: Goal velocity [vx, vy, vz], can contain None for unconstrained axes
        """
        for i in range(3):
            if vel[i] is not None:
                self.set_goal_velocity_in_axis(i, vel[i])
    
    def set_goal_acceleration(self, acc):
        """Set goal acceleration
        
        Args:
            acc: Goal acceleration [ax, ay, az], can contain None for unconstrained axes
        """
        for i in range(3):
            if acc[i] is not None:
                self.set_goal_acceleration_in_axis(i, acc[i])
    
    def set_goal_position_in_axis(self, axNum, pos):
        """Set goal position in specified axis"""
        self._axis[axNum].set_goal_position(pos)
    
    def set_goal_velocity_in_axis(self, axNum, vel):
        """Set goal velocity in specified axis"""
        self._axis[axNum].set_goal_velocity(vel)
    
    def set_goal_acceleration_in_axis(self, axNum, acc):
        """Set goal acceleration in specified axis"""
        self._axis[axNum].set_goal_acceleration(acc)
    
    def generate(self, timeToGo, num_points=10):
        """Generate a trajectory with duration timeToGo
        
        Args:
            timeToGo: Trajectory duration
            num_points: Number of control points for generating splines
        """
        self._tf = timeToGo
        for i in range(3):
            self._axis[i].generate(self._tf, num_points)
    
    def check_input_feasibility(self, fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection):
        """Check input feasibility
        
        Tests if trajectory satisfies thrust and body rate constraints.
        
        Args:
            fminAllowed: Minimum allowed thrust [m/s^2]
            fmaxAllowed: Maximum allowed thrust [m/s^2]
            wmaxAllowed: Maximum allowed body rate [rad/s]
            minTimeSection: Minimum time interval for recursive checking [s]
            
        Returns:
            InputFeasibilityResult: Feasibility result enumeration
        """
        return self._check_input_feasibility_section(fminAllowed, fmaxAllowed,
                                                   wmaxAllowed, minTimeSection, 0, self._tf)
    
    def _check_input_feasibility_section(self, fminAllowed, fmaxAllowed, 
                                       wmaxAllowed, minTimeSection, t1, t2):
        """Internal method to recursively check input feasibility"""
        if (t2-t1) < minTimeSection:
            return InputFeasibilityResult.Indeterminable
        
        # Check thrust at boundary points
        if max(self.get_thrust(t1), self.get_thrust(t2)) > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh
        if min(self.get_thrust(t1), self.get_thrust(t2)) < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow
        
        fminSqr = 0
        fmaxSqr = 0
        jmaxSqr = 0
        
        # Check the limits of the box around the trajectory segment
        for i in range(3):
            amin, amax = self._axis[i].get_min_max_acc(t1, t2)
            
            # Distance from zero thrust point in this axis
            v1 = amin - self._grav[i]  # left bound
            v2 = amax - self._grav[i]  # right bound
            
            # Definitely infeasible:
            if (max(v1**2, v2**2) > fmaxAllowed**2):
                return InputFeasibilityResult.InfeasibleThrustHigh
            
            if (v1*v2 < 0):
                # Sign of acceleration changes, so we've gone through zero
                fminSqr += 0
            else:
                fminSqr += min(np.fabs(v1), np.fabs(v2))**2
            
            fmaxSqr += max(np.fabs(v1), np.fabs(v2))**2
            
            jmaxSqr += self._axis[i].get_max_jerk_squared(t1, t2)
        
        fmin = np.sqrt(fminSqr)
        fmax = np.sqrt(fmaxSqr)
        
        # Avoid division by zero
        if fminSqr > 1e-6:
            wBound = np.sqrt(jmaxSqr / fminSqr)
        else:
            wBound = float("inf")
        
        # Definitely infeasible:
        if fmax < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow
        if fmin > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh
        
        # Possibly infeasible:
        if (fmin < fminAllowed) or (fmax > fmaxAllowed) or (wBound > wmaxAllowed):
            # Indeterminate: need to check more closely:
            tHalf = (t1 + t2) / 2.0
            r1 = self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, t1, tHalf)
            
            if r1 == InputFeasibilityResult.Feasible:
                # Check the other half
                return self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, tHalf, t2)
            else:
                # Infeasible, or indeterminable
                return r1
        
        # Definitely feasible:
        return InputFeasibilityResult.Feasible
    
    def check_position_feasibility(self, boundaryPoint, boundaryNormal):
        """Check position feasibility
        
        Tests if trajectory stays on allowed side of a plane defined by a point and normal vector.
        
        Args:
            boundaryPoint: A point on the boundary plane
            boundaryNormal: Normal vector of the plane, pointing toward allowed region
            
        Returns:
            StateFeasibilityResult: Feasibility result enumeration
        """
        boundaryNormal = np.array(boundaryNormal)
        boundaryPoint = np.array(boundaryPoint)
        
        # Ensure it's a unit vector
        boundaryNormal = boundaryNormal / np.linalg.norm(boundaryNormal)
        
        # Calculate critical times
        # For spline trajectories, we find critical times by finding zeroes of velocity projection on normal
        
        # Sample enough time points to ensure we don't miss critical points
        t_samples = np.linspace(0, self._tf, 100)
        
        # Calculate distance from point to plane at each time
        distances = []
        for t in t_samples:
            pos = self.get_position(t)
            dist = np.dot(pos - boundaryPoint, boundaryNormal)
            distances.append(dist)
        
        # Check if all sampled points are on allowed side
        if min(distances) < 0:
            return StateFeasibilityResult.Infeasible
        
        return StateFeasibilityResult.Feasible
    
    def get_jerk(self, t):
        """Return 3D jerk at time t"""
        return np.array([self._axis[i].get_jerk(t) for i in range(3)])
    
    def get_acceleration(self, t):
        """Return 3D acceleration at time t"""
        return np.array([self._axis[i].get_acceleration(t) for i in range(3)])
    
    def get_velocity(self, t):
        """Return 3D velocity at time t"""
        return np.array([self._axis[i].get_velocity(t) for i in range(3)])
    
    def get_position(self, t):
        """Return 3D position at time t"""
        return np.array([self._axis[i].get_position(t) for i in range(3)])
    
    def get_normal_vector(self, t):
        """Return body normal vector at time t
        
        The drone's normal vector is the direction in which thrust points.
        It can be calculated from acceleration minus gravity.
        """
        v = (self.get_acceleration(t) - self._grav)
        return v / np.linalg.norm(v)
    
    def get_thrust(self, t):
        """Return thrust magnitude at time t"""
        return np.linalg.norm(self.get_acceleration(t) - self._grav)
    
    def get_body_rates(self, t, dt=1e-3):
        """Return body rates at time t
        
        Args:
            t: Time
            dt: Time differential for angular velocity calculation
            
        Returns:
            Angular velocity vector in inertial frame
        """
        n0 = self.get_normal_vector(t)
        n1 = self.get_normal_vector(t + dt)
        
        crossProd = np.cross(n0, n1)  # Direction of angular velocity in inertial frame
        
        if np.linalg.norm(crossProd) > 1e-6:
            return np.arccos(np.dot(n0, n1)) / dt * (crossProd / np.linalg.norm(crossProd))
        else:
            return np.array([0, 0, 0])
    
    def get_cost(self):
        """Return total trajectory cost"""
        return self._axis[0].get_cost() + self._axis[1].get_cost() + self._axis[2].get_cost()


# Input feasibility result enumeration
class InputFeasibilityResult:
    """Enumeration of possible outcomes for input feasibility test
    
    If the test result is not "feasible", it returns the result of the first failing segment:
        0: Feasible -- trajectory is feasible with respect to input constraints
        1: Indeterminable -- a segment's feasibility could not be determined
        2: InfeasibleThrustHigh -- a segment failed due to max thrust constraint
        3: InfeasibleThrustLow -- a segment failed due to min thrust constraint
    """
    Feasible, Indeterminable, InfeasibleThrustHigh, InfeasibleThrustLow = range(4)
    
    @classmethod
    def to_string(cls, ifr):
        """Return the name of the result"""
        if ifr == InputFeasibilityResult.Feasible:
            return "Feasible"
        elif ifr == InputFeasibilityResult.Indeterminable:
            return "Indeterminable"
        elif ifr == InputFeasibilityResult.InfeasibleThrustHigh:
            return "InfeasibleThrustHigh"
        elif ifr == InputFeasibilityResult.InfeasibleThrustLow:
            return "InfeasibleThrustLow"
        return "Unknown"


# State feasibility result enumeration
class StateFeasibilityResult:
    """Enumeration of possible outcomes for state feasibility test
    
    The result is either feasible (0) or infeasible (1).
    """
    Feasible, Infeasible = range(2)
    
    @classmethod
    def to_string(cls, ifr):
        """Return the name of the result"""
        if ifr == StateFeasibilityResult.Feasible:
            return "Feasible"
        elif ifr == StateFeasibilityResult.Infeasible:
            return "Infeasible"
        return "Unknown"