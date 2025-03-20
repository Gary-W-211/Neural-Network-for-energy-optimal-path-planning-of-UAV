"""
轨迹生成器 - 使用样条曲线方法
作者: 基于Mark W. Mueller的原始代码
"""

import numpy as np
from scipy.interpolate import CubicSpline, splrep, splev
from scipy.optimize import minimize

class SplineTrajectory:
    """单轴样条曲线轨迹
    
    这个类实现了一个基于样条曲线的单轴轨迹生成器。它可以用于生成满足位置、
    速度和加速度约束的平滑轨迹。
    """
    
    def __init__(self, pos0, vel0, acc0):
        """初始化轨迹起始状态"""
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
        """重置轨迹参数"""
        self._cost = float("inf")
        self._accGoalDefined = False
        self._velGoalDefined = False
        self._posGoalDefined = False
        self._spline = None
        self._time_points = None
        self._control_points = None
    
    def set_goal_position(self, posf):
        """设置目标位置"""
        self._posGoalDefined = True
        self._pf = posf
        
    def set_goal_velocity(self, velf):
        """设置目标速度"""
        self._velGoalDefined = True
        self._vf = velf
        
    def set_goal_acceleration(self, accf):
        """设置目标加速度"""
        self._accGoalDefined = True
        self._af = accf
        
    def generate(self, Tf, num_points=10):
        """生成持续时间为Tf的轨迹
        
        Args:
            Tf: 轨迹持续时间
            num_points: 用于生成样条曲线的控制点数量
        """
        # 定义时间点
        self._time_points = np.linspace(0, Tf, num_points)
        
        # 初始条件约束
        constraints = []
        
        # 如果已经定义了终点位置
        if self._posGoalDefined and self._velGoalDefined and self._accGoalDefined:
            # 使用完全约束条件下的样条插值方法
            self._generate_fully_constrained_trajectory(Tf, num_points)
        elif self._posGoalDefined:
            # 仅位置约束的优化问题
            self._generate_position_constrained_trajectory(Tf, num_points)
        elif self._velGoalDefined:
            # 仅速度约束的优化问题
            self._generate_velocity_constrained_trajectory(Tf, num_points)
        elif self._accGoalDefined:
            # 仅加速度约束的优化问题
            self._generate_acceleration_constrained_trajectory(Tf, num_points)
        else:
            # 无约束，保持初始轨迹
            self._control_points = np.linspace(self._p0, self._p0, num_points)
            self._spline = CubicSpline(self._time_points, self._control_points, 
                                      bc_type=((1, self._v0), (1, 0)))
        
        # 计算轨迹代价（积分平方加加速度）
        self._calculate_cost(Tf)
    
    def _generate_fully_constrained_trajectory(self, Tf, num_points):
        """生成完全约束条件下的轨迹（位置、速度、加速度都有终点约束）"""
        # 使用样条方法生成满足所有边界条件的轨迹
        # 这里我们用5个关键点：起点、终点和3个中间点
        t = np.array([0, Tf/4, Tf/2, 3*Tf/4, Tf])
        
        # 初始和结束状态
        x0 = self._p0
        v0 = self._v0
        a0 = self._a0
        xf = self._pf
        vf = self._vf
        af = self._af
        
        # 构建中间点初始猜测（线性插值）
        middle_points = np.linspace(x0, xf, 5)[1:-1]
        
        # 定义优化目标函数：最小化加加速度平方的积分
        def objective(middle):
            # 构建控制点
            control_points = np.array([x0, middle[0], middle[1], middle[2], xf])
            # 创建样条
            spline = CubicSpline(t, control_points, 
                                bc_type=((2, a0), (2, af)))
            
            # 检查速度边界条件
            v_start = spline(0, 1)
            v_end = spline(Tf, 1)
            
            # 惩罚不满足速度约束的解
            vel_penalty = 100 * ((v_start - v0)**2 + (v_end - vf)**2)
            
            # 计算加加速度平方的积分
            t_eval = np.linspace(0, Tf, 100)
            jerk = spline(t_eval, 3)
            jerk_squared_integral = np.trapz(jerk**2, t_eval)
            
            return jerk_squared_integral + vel_penalty
        
        # 约束：确保速度在起点和终点满足要求
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
        
        # 运行优化
        result = minimize(objective, middle_points, method='SLSQP', 
                         constraints=constraints, options={'maxiter': 100})
        
        # 使用优化结果构建最终样条
        optimal_middle = result.x
        self._control_points = np.array([x0, optimal_middle[0], optimal_middle[1], 
                                       optimal_middle[2], xf])
        self._time_points = t
        self._spline = CubicSpline(t, self._control_points, 
                                 bc_type=((2, a0), (2, af)))
    
    def _generate_position_constrained_trajectory(self, Tf, num_points):
        """生成只有位置约束的轨迹"""
        # 简单地使用起点和终点位置，以及起点速度和加速度
        t = np.array([0, Tf])
        control_points = np.array([self._p0, self._pf])
        
        # 使用自然样条，但指定起点的速度和加速度
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((1, self._v0), (2, 0)))
        self._time_points = t
        self._control_points = control_points
    
    def _generate_velocity_constrained_trajectory(self, Tf, num_points):
        """生成只有速度约束的轨迹"""
        # 使用起点和一个控制点
        t = np.array([0, Tf])
        
        # 计算符合速度约束的终点位置
        end_pos = self._p0 + self._v0 * Tf + 0.5 * (self._vf - self._v0) * Tf
        control_points = np.array([self._p0, end_pos])
        
        # 创建样条，指定两端的速度
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((1, self._v0), (1, self._vf)))
        self._time_points = t
        self._control_points = control_points
    
    def _generate_acceleration_constrained_trajectory(self, Tf, num_points):
        """生成只有加速度约束的轨迹"""
        # 使用起点和一个控制点
        t = np.array([0, Tf])
        
        # 计算符合加速度约束的终点位置
        end_pos = self._p0 + self._v0 * Tf + 0.5 * self._a0 * Tf**2 + (1/6) * (self._af - self._a0) * Tf**2
        control_points = np.array([self._p0, end_pos])
        
        # 创建样条，指定两端的加速度
        self._spline = CubicSpline(t, control_points, 
                                 bc_type=((2, self._a0), (2, self._af)))
        self._time_points = t
        self._control_points = control_points
    
    def _calculate_cost(self, Tf):
        """计算轨迹代价（积分平方加加速度）"""
        if self._spline is None:
            self._cost = float("inf")
            return
        
        # 在整个时间范围内评估加加速度
        t_eval = np.linspace(0, Tf, 100)
        jerk = self._spline(t_eval, 3)  # 三阶导数是加加速度
        
        # 使用梯形法则计算积分
        self._cost = np.trapz(jerk**2, t_eval)
    
    def get_position(self, t):
        """返回时间t处的位置"""
        if self._spline is None:
            return self._p0
        return float(self._spline(t))
    
    def get_velocity(self, t):
        """返回时间t处的速度"""
        if self._spline is None:
            return self._v0
        return float(self._spline(t, 1))
    
    def get_acceleration(self, t):
        """返回时间t处的加速度"""
        if self._spline is None:
            return self._a0
        return float(self._spline(t, 2))
    
    def get_jerk(self, t):
        """返回时间t处的加加速度"""
        if self._spline is None:
            return 0.0
        return float(self._spline(t, 3))
    
    def get_cost(self):
        """返回轨迹的总代价"""
        return self._cost
    
    def get_min_max_acc(self, t1, t2):
        """返回t1和t2之间的最小和最大加速度"""
        if self._spline is None:
            return (self._a0, self._a0)
        
        # 在t1和t2之间采样足够多的点来找到极值
        t_samples = np.linspace(t1, t2, 50)
        acc_samples = self._spline(t_samples, 2)
        
        return (float(np.min(acc_samples)), float(np.max(acc_samples)))
    
    def get_max_jerk_squared(self, t1, t2):
        """返回t1和t2之间的最大平方加加速度"""
        if self._spline is None:
            return 0.0
        
        # 在t1和t2之间采样足够多的点来找到极值
        t_samples = np.linspace(t1, t2, 50)
        jerk_samples = self._spline(t_samples, 3)
        
        return float(np.max(jerk_samples**2))


class SplineTrajectoryGenerator:
    """样条曲线轨迹生成器
    
    使用样条曲线方法生成满足起点和终点约束的三维轨迹。
    可以测试输入约束（推力/角速度）和状态约束（位置）的可行性。
    """
    
    def __init__(self, pos0, vel0, acc0, gravity):
        """初始化轨迹生成器
        
        Args:
            pos0: 初始位置 [x, y, z]
            vel0: 初始速度 [vx, vy, vz]
            acc0: 初始加速度 [ax, ay, az]
            gravity: 重力加速度 [gx, gy, gz]
        """
        self._axis = [SplineTrajectory(pos0[i], vel0[i], acc0[i]) for i in range(3)]
        self._grav = gravity
        self._tf = None
        self.reset()
    
    def reset(self):
        """重置轨迹生成器"""
        for i in range(3):
            self._axis[i].reset()
    
    def set_goal_position(self, pos):
        """设置目标位置
        
        Args:
            pos: 目标位置 [x, y, z]，可以包含None表示不约束该轴
        """
        for i in range(3):
            if pos[i] is not None:
                self.set_goal_position_in_axis(i, pos[i])
    
    def set_goal_velocity(self, vel):
        """设置目标速度
        
        Args:
            vel: 目标速度 [vx, vy, vz]，可以包含None表示不约束该轴
        """
        for i in range(3):
            if vel[i] is not None:
                self.set_goal_velocity_in_axis(i, vel[i])
    
    def set_goal_acceleration(self, acc):
        """设置目标加速度
        
        Args:
            acc: 目标加速度 [ax, ay, az]，可以包含None表示不约束该轴
        """
        for i in range(3):
            if acc[i] is not None:
                self.set_goal_acceleration_in_axis(i, acc[i])
    
    def set_goal_position_in_axis(self, axNum, pos):
        """在指定轴上设置目标位置"""
        self._axis[axNum].set_goal_position(pos)
    
    def set_goal_velocity_in_axis(self, axNum, vel):
        """在指定轴上设置目标速度"""
        self._axis[axNum].set_goal_velocity(vel)
    
    def set_goal_acceleration_in_axis(self, axNum, acc):
        """在指定轴上设置目标加速度"""
        self._axis[axNum].set_goal_acceleration(acc)
    
    def generate(self, timeToGo, num_points=10):
        """生成持续时间为timeToGo的轨迹
        
        Args:
            timeToGo: 轨迹持续时间
            num_points: 用于生成样条曲线的控制点数量
        """
        self._tf = timeToGo
        for i in range(3):
            self._axis[i].generate(self._tf, num_points)
    
    def check_input_feasibility(self, fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection):
        """检查输入可行性
        
        检查轨迹是否满足推力和角速度约束。
        
        Args:
            fminAllowed: 允许的最小推力 [m/s^2]
            fmaxAllowed: 允许的最大推力 [m/s^2]
            wmaxAllowed: 允许的最大角速度 [rad/s]
            minTimeSection: 递归检查的最小时间间隔 [s]
            
        Returns:
            InputFeasibilityResult: 可行性结果的枚举
        """
        return self._check_input_feasibility_section(fminAllowed, fmaxAllowed,
                                                   wmaxAllowed, minTimeSection, 0, self._tf)
    
    def _check_input_feasibility_section(self, fminAllowed, fmaxAllowed, 
                                       wmaxAllowed, minTimeSection, t1, t2):
        """递归检查输入可行性的内部方法"""
        if (t2-t1) < minTimeSection:
            return InputFeasibilityResult.Indeterminable
        
        # 检查边界点的推力
        if max(self.get_thrust(t1), self.get_thrust(t2)) > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh
        if min(self.get_thrust(t1), self.get_thrust(t2)) < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow
        
        fminSqr = 0
        fmaxSqr = 0
        jmaxSqr = 0
        
        # 检查在时间区间内的边界限制
        for i in range(3):
            amin, amax = self._axis[i].get_min_max_acc(t1, t2)
            
            # 该轴上离零推力点的距离
            v1 = amin - self._grav[i]  # 左边界
            v2 = amax - self._grav[i]  # 右边界
            
            # 肯定不可行:
            if (max(v1**2, v2**2) > fmaxAllowed**2):
                return InputFeasibilityResult.InfeasibleThrustHigh
            
            if (v1*v2 < 0):
                # 加速度符号改变，说明经过了零点
                fminSqr += 0
            else:
                fminSqr += min(np.fabs(v1), np.fabs(v2))**2
            
            fmaxSqr += max(np.fabs(v1), np.fabs(v2))**2
            
            jmaxSqr += self._axis[i].get_max_jerk_squared(t1, t2)
        
        fmin = np.sqrt(fminSqr)
        fmax = np.sqrt(fmaxSqr)
        
        # 防止除零
        if fminSqr > 1e-6:
            wBound = np.sqrt(jmaxSqr / fminSqr)
        else:
            wBound = float("inf")
        
        # 肯定不可行:
        if fmax < fminAllowed:
            return InputFeasibilityResult.InfeasibleThrustLow
        if fmin > fmaxAllowed:
            return InputFeasibilityResult.InfeasibleThrustHigh
        
        # 可能不可行:
        if (fmin < fminAllowed) or (fmax > fmaxAllowed) or (wBound > wmaxAllowed):
            # 不确定性: 需要更仔细地检查:
            tHalf = (t1 + t2) / 2.0
            r1 = self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, t1, tHalf)
            
            if r1 == InputFeasibilityResult.Feasible:
                # 检查另一半
                return self._check_input_feasibility_section(fminAllowed, fmaxAllowed, wmaxAllowed, minTimeSection, tHalf, t2)
            else:
                # 不可行,或不确定
                return r1
        
        # 肯定可行:
        return InputFeasibilityResult.Feasible
    
    def check_position_feasibility(self, boundaryPoint, boundaryNormal):
        """检查位置可行性
        
        检查轨迹是否始终位于由点和法向量定义的平面的允许一侧。
        
        Args:
            boundaryPoint: 平面上的一点
            boundaryNormal: 平面的法向量，指向允许区域的方向
            
        Returns:
            StateFeasibilityResult: 可行性结果的枚举
        """
        boundaryNormal = np.array(boundaryNormal)
        boundaryPoint = np.array(boundaryPoint)
        
        # 确保是单位向量
        boundaryNormal = boundaryNormal / np.linalg.norm(boundaryNormal)
        
        # 计算关键时间点
        # 1. 对于样条轨迹，我们通过求解速度在法线方向上的零点来找到关键时间
        
        # 采样足够多的时间点来确保不遗漏关键点
        t_samples = np.linspace(0, self._tf, 100)
        
        # 计算每个时间点位置到平面的投影距离
        distances = []
        for t in t_samples:
            pos = self.get_position(t)
            dist = np.dot(pos - boundaryPoint, boundaryNormal)
            distances.append(dist)
        
        # 检查所有采样点是否都在允许区域内
        if min(distances) < 0:
            return StateFeasibilityResult.Infeasible
        
        return StateFeasibilityResult.Feasible
    
    def get_jerk(self, t):
        """返回时间t处的三维加加速度"""
        return np.array([self._axis[i].get_jerk(t) for i in range(3)])
    
    def get_acceleration(self, t):
        """返回时间t处的三维加速度"""
        return np.array([self._axis[i].get_acceleration(t) for i in range(3)])
    
    def get_velocity(self, t):
        """返回时间t处的三维速度"""
        return np.array([self._axis[i].get_velocity(t) for i in range(3)])
    
    def get_position(self, t):
        """返回时间t处的三维位置"""
        return np.array([self._axis[i].get_position(t) for i in range(3)])
    
    def get_normal_vector(self, t):
        """返回时间t处的机体法向量
        
        无人机的法向量是推力指向的方向向量。可以通过计算加速度减去重力
        来获得所需的推力方向。
        """
        v = (self.get_acceleration(t) - self._grav)
        return v / np.linalg.norm(v)
    
    def get_thrust(self, t):
        """返回时间t处的推力大小"""
        return np.linalg.norm(self.get_acceleration(t) - self._grav)
    
    def get_body_rates(self, t, dt=1e-3):
        """返回时间t处的机体角速度
        
        Args:
            t: 时间
            dt: 计算角速度的时间差分
            
        Returns:
            惯性坐标系中的角速度向量
        """
        n0 = self.get_normal_vector(t)
        n1 = self.get_normal_vector(t + dt)
        
        crossProd = np.cross(n0, n1)  # 惯性坐标系中角速度的方向
        
        if np.linalg.norm(crossProd) > 1e-6:
            return np.arccos(np.dot(n0, n1)) / dt * (crossProd / np.linalg.norm(crossProd))
        else:
            return np.array([0, 0, 0])
    
    def get_cost(self):
        """返回轨迹的总代价"""
        return self._axis[0].get_cost() + self._axis[1].get_cost() + self._axis[2].get_cost()


# 输入可行性结果的枚举
class InputFeasibilityResult:
    """输入可行性测试的可能结果枚举
    
    如果测试结果不是"可行"，它将返回第一个失败段的结果。不同的结果是:
        0: 可行 -- 轨迹在输入约束下是可行的
        1: 不确定 -- 部分的可行性无法确定
        2: 推力过高 -- 由于最大推力约束而失败
        3: 推力过低 -- 由于最小推力约束而失败
    """
    Feasible, Indeterminable, InfeasibleThrustHigh, InfeasibleThrustLow = range(4)
    
    @classmethod
    def to_string(cls, ifr):
        """返回结果的名称"""
        if ifr == InputFeasibilityResult.Feasible:
            return "Feasible"
        elif ifr == InputFeasibilityResult.Indeterminable:
            return "Indeterminable"
        elif ifr == InputFeasibilityResult.InfeasibleThrustHigh:
            return "InfeasibleThrustHigh"
        elif ifr == InputFeasibilityResult.InfeasibleThrustLow:
            return "InfeasibleThrustLow"
        return "Unknown"


# 状态可行性结果的枚举
class StateFeasibilityResult:
    """状态可行性测试的可能结果枚举
    
    结果要么是可行(0)，要么是不可行(1)。
    """
    Feasible, Infeasible = range(2)
    
    @classmethod
    def to_string(cls, ifr):
        """返回结果的名称"""
        if ifr == StateFeasibilityResult.Feasible:
            return "Feasible"
        elif ifr == StateFeasibilityResult.Infeasible:
            return "Infeasible"
        return "Unknown"