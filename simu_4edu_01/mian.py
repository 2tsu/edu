import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import quaternion as qtn
import pandas as pd

#質量初期と終端
m_init =5.573 #[kg]
m_end = 4.8030#[kg]
#機軸方向の慣性モーメント
I_init = 1.554
I_end = 1.435
#燃焼時間
burn_time = 4.6
#重力加速度
g = 9.8
gravity = np.array([0,0,-1*g])
#ロケットの底面から測った重心距離　初期と終端
l_cg_init =0.83494 #[m]
l_cg_end = 0.85972 #[m]
#ロケット底面から測った圧力中心距離　初期と終端
l_cp = 0.5299 #[m]
#法線力傾斜係数
C_na = 10.74
#軸力係数
C_a = 0.3948
#直径
R = 0.102 #[m]
#断面積
S = 1/2 * R**2 /4 *np.pi
#空気密度
rho = 1.293 #[kg/m3]
#ランチャーの迎角　地面に沿う方向を０°
launcher_angle = 70 #[deg]
#風ベクトル
v_wind = np.array([1, 0, 0])
#ランチャーの長さ
launcher_length = 5. #[m]
#推力データ
thrust_excel = pd.read_excel('./thrust_K240_20230604.xlsx').values
thrust_data = thrust_excel[:, 1]
thrust_time = thrust_excel[:,0]

#以降の力は機体軸座標系で定義するものとします。
def thrust_vecter(t):
    x=thrust_time
    y=thrust_data
    if t <= x[-1]:
        f=interpolate.CubicSpline(x, y)
        return np.array([f(t), 0, 0])
    else:
        return np.array([0,0,0])


def axial_force_vecter(t, v_airspeed):
    return np.array([-1 * 1/2 * rho * np.dot(v_airspeed, v_airspeed) * S * C_a, 0, 0])


def normal_force_vecter(t, v_airspeed, alpha):
    return np.array([0, 0, 1/2 * rho * np.dot(v_airspeed, v_airspeed) * S * C_na * alpha])



def mass_scaler(t):
    return m_init
def alpha_scaler(t, v_airspeed, q):
    R = qtn.as_rotation_matrix(q)
    normal_direction = np.array([1,0,0]) @ R
    normal_airspeed = v_airspeed / np.linalg.norm(v_airspeed)

    alpha = np.arccos(np.dot(normal_airspeed, normal_direction))
    if (normal_airspeed - normal_direction)[2] > 0:
        return alpha
    else:
        return -1 * alpha#rad

def quaternion_update_matrix(t, w, q):
  return 1/2 *np.array([[0, -1*w[0],-1*w[1], -1*w[2]],
                        [w[0], 0, w[2], -1*w[1]],
                        [w[1], -1*w[2], 0, w[0]],
                        [w[2], w[1], -1*w[0], 0]])
def event_launchclear(t, A):
    r = A[:3]
    q = A[9:]
    return launcher_length - np.linalg.norm(r)
event_launchclear.teminal = True
event_launchclear.direction = -1

def event_ground(t, A):
    return A[2]

event_ground.direction = -1
event_ground.terminal = True
def ode_launcher(t, A):
    """
    A = (r, v, w, q )
    ランチャーレール上の運動方程式の数値計算を定義する。
    自由度が１になっていることに注意
    
    運動方程式
    dvdt = (軸力ベクトル+法線力ベクトル+推力ベクトル) / 質量 @ 回転行列 + 重力加速度ベクトル
    一項目の回転行列より前半は機軸系で定義されている力たちであるため、
    回転行列を作用させてnet座標系に変換した後、net座標系で定義されている重力加速度ベクトルを足す
    """
    #要素を取り出す。
    r =A[:3]
    v = A[3:6]
    w = A[6:9]
    q = A[9:]
    #np.quaternion.quaternionにしておく。
    q = np.quaternion(*q)
    R = qtn.as_rotation_matrix(q)
    #対気速度(相対速度)
    v_airspeed = v - v_wind
    #自由度を下げる。
    dwdt = np.zeros(3)
    #速度積分
    drdt = v
    #機軸方向の重力成分
    gravity_force = np.array([np.dot(R @ np.array([1,0,0]), gravity), 0, 0])
    #並進運動方程式
    dvdt = R @ ( axial_force_vecter(t, v_airspeed) + thrust_vecter(t) + gravity_force ) / mass_scaler(t) 
    #ランチャー上では姿勢は更新しない。
    dqdt = np.zeros(4)
    return np.array([*drdt, *dvdt, *dwdt, *dqdt])

def quaternion_generater(axe=list, degree=float):
    angle = np.deg2rad(degree)#convert deg to rad
    q = np.quaternion(np.cos(angle/2),  *np.array(axe)*np.sin(angle/2))
    return q

np.rad2deg(alpha_scaler(0, np.array([1,0,0]), quaternion_generater([0,-1,0], launcher_angle)))
def ode_flight(t, A):
    """
    A = (r, v, w, q )
    let r = (r_x, r_y, r_z), v = (v_x, v_y, v_z), w = (w_x, w_y, w_z), q = qtn

    """
    #要素を取り出す。
    r =A[:3]
    v = A[3:6]
    w = A[6:9]
    q = A[9:]
    #np.quaternion.quaternionにしておく。
    q = np.quaternion(*q)
    R = qtn.as_rotation_matrix(q)
    #対気速度(相対速度)
    v_airspeed = v - v_wind
    alpha = alpha_scaler(t, v_airspeed, q)
    
    #速度積分
    drdt = v
    #並進運動方程式
    dvdt = R @ ( axial_force_vecter(t, v_airspeed) + normal_force_vecter(t, v_airspeed, alpha) + thrust_vecter(t) ) / mass_scaler(t) + gravity
    #回転運動方程式　今回はy軸周りのみ
    dwdt = np.array([0,(1*np.abs(l_cp - l_cg_init) * normal_force_vecter(t, v_airspeed, alpha)[2]) / I_init, 0])
    #クオータニオン
    dqdt =quaternion_update_matrix(t, w,q)@qtn.as_float_array(q)
    return np.array([*drdt, *dvdt, *dwdt, *dqdt])

t_span = [0, 1]
t_eval = np.linspace(*t_span, 1000)
r0= [0,0,0]
v0 =[0.1, 0,0]
w0 = [0,0,0]
q0 = qtn.as_float_array(quaternion_generater([0,-1,0], launcher_angle))
A0 = np.array([*r0, *v0, *w0,*q0])
sol_launcher = integrate.solve_ivp(ode_launcher,t_span,A0, t_eval = t_eval,dense_output = True, events =event_launchclear)

print(sol_launcher.y_events)
x = sol_launcher.y[0]
z = sol_launcher.y[2]

plt.plot(x,z)
plt.show()

A_launcher = np.ravel(sol_launcher.y_events)
print(A_launcher)
t_launcher = sol_launcher.t_events[0][0]
print(t_launcher)

t_span = [t_launcher, 100]
t_eval = np.linspace(*t_span, 100000)
A0 = np.array(A_launcher)
sol_flight = integrate.solve_ivp(ode_flight,t_span,A0, t_eval = t_eval,dense_output = True, events =event_ground)

x1 = sol_flight.y[0]
z1 = sol_flight.y[2]
plt.plot(x1,z1)
plt.show()

axe2= plt.figure(figsize=(5,5)).add_subplot()
axe2.plot(sol_flight.t, sol_flight.y[0], color = "blue", label = "x")
axe2.plot(sol_flight.t, sol_flight.y[2], color ="orange", label = "z")
axe2.plot(sol_flight.t, sol_flight.y[3], color = "red", label = "vx")
axe2.plot(sol_flight.t, sol_flight.y[5], color ="pink", label = "vz")
axe2.plot(sol_flight.t, sol_flight.y[7], color ="black", label = "wy")
axe2.legend()
plt.show()