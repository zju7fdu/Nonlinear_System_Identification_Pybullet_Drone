





# Place at the top of models_1.py alongside other imports
import math
import torch
import torch.nn as nn
import xml.etree.ElementTree as etxml
import pkg_resources
import numpy as np
import os

# If you already have enums.DroneModel, optional: from enums import DroneModel
# Here we directly use the string "cf2x" to reduce dependencies
_DEFAULT_URDF_NAME = "cf2x"


class DronePlant(nn.Module):
    """
    CF2X quadrotor plant (batched, RPM inputs) with optional drag & ground-effect.
    Dimensions remain unchanged:
      state x = [p(3), v(3), q(4), w(3)] (13)
      input u = [rpm0, rpm1, rpm2, rpm3] (4, units: RPM)
      output "full" -> 13; "pos_yaw" -> 4
    """
    def __init__(
        self,
        dt: float = 1/240,
        # -- New: URDF auto-loading switch and URDF name (default is cf2x) --
        use_urdf: bool = True,
        urdf_name: str = _DEFAULT_URDF_NAME,

        # If use_urdf=False, use these default values (previously hardcoded approximate parameters)
        mass: float = 0.027,
        J: torch.Tensor | None = None,
        g: float = 9.81,
        arm_length: float = 0.0397,
        KF: float = 3.16e-10,
        KM: float = 7.94e-12,
        drag_coeff_xy: float = 9.1785e-7,
        drag_coeff_z: float = 10.311e-7,
        gnd_eff_coeff: float = 1.0,
        prop_radius: float = 0.0204,
        max_speed_kmh: float = 30.0,
        thrust2weight_ratio: float = 1.9,

        output_mode: str = "full",
        enable_drag: bool = True,
        enable_ground_effect: bool = True,
        enable_downwash: bool = False,
    ):
        super().__init__()
        self.state_dim  = 13
        self.input_dim  = 4
        self.output_mode = output_mode
        self.output_dim = 13 if output_mode == "full" else 4
        self.dt = float(dt)

        # ========== URDF Auto-Loading ==========
        if use_urdf:
            params = self._load_urdf_params(urdf_name)  # Parse from gym_pybullet_drones/assets package
            self.m  = float(params["M"])
            J_np    = params["J"]                        # numpy 3x3
            self.J  = torch.tensor(J_np, dtype=torch.float32)
            self.L  = float(params["L"])
            self.KF = float(params["KF"])
            self.KM = float(params["KM"])
            self.max_speed_kmh = float(params["MAX_SPEED_KMH"])
            self.gnd_eff_coeff = float(params["GND_EFF_COEFF"])
            self.prop_radius   = float(params["PROP_RADIUS"])
            DRAG = params["DRAG_COEFF"]                 # numpy [xy, xy, z]
            self.register_buffer("Cd", torch.tensor(DRAG, dtype=torch.float32).view(1,1,3))
            # Thrust-to-weight ratio is in URDF (thrust2weight)
            self.thrust2weight_ratio = float(params["THRUST2WEIGHT_RATIO"])
        else:
            # Fallback: use fixed default values
            self.m  = float(mass)
            if J is None:
                self.J = torch.diag(torch.tensor([1.4e-5, 1.4e-5, 2.17e-5], dtype=torch.float32))
            else:
                self.J = J.clone().detach().to(torch.float32)
            self.L  = float(arm_length)
            self.KF = float(KF)
            self.KM = float(KM)
            self.max_speed_kmh = float(max_speed_kmh)
            self.gnd_eff_coeff = float(gnd_eff_coeff)
            self.prop_radius   = float(prop_radius)
            Cd = torch.tensor([float(drag_coeff_xy), float(drag_coeff_xy), float(drag_coeff_z)], dtype=torch.float32)
            self.register_buffer("Cd", Cd.view(1,1,3))
            self.thrust2weight_ratio = float(thrust2weight_ratio)

        # Other switches & constants
        self.enable_drag          = bool(enable_drag)
        self.enable_ground_effect = bool(enable_ground_effect)
        self.enable_downwash      = bool(enable_downwash)

        self.register_buffer("g_vec", torch.tensor([0.0, 0.0, -float(g)], dtype=torch.float32).view(1,1,3))

        # -- Derived quantities (consistent with BaseAviary derivation) --
        self.GRAVITY    = float(g) * self.m
        self.HOVER_RPM  = (self.GRAVITY / (4.0 * self.KF))**0.5
        self.MAX_RPM    = ((self.thrust2weight_ratio*self.GRAVITY) / (4.0*self.KF))**0.5
        self.MAX_THRUST = 4.0 * self.KF * (self.MAX_RPM**2)
        # Ground effect height clipping
        self.GND_EFF_H_CLIP = 0.25 * self.prop_radius * (
            (15.0 * (self.MAX_RPM**2) * self.KF * self.gnd_eff_coeff) / max(self.MAX_THRUST, 1e-9)
        )**0.5

    # ===== URDF Parsing (directly borrowed from BaseAviary's method logic) =====
    @staticmethod
    def _load_urdf_params(urdf_name: str) -> dict:
        """
        Parse assets/{urdf_name}.urdf and return key parameter dictionary.
        """
        urdf_path = os.path.join(os.path.dirname(__file__), "assets", f"{urdf_name}.urdf")

        tree = etxml.parse(urdf_path).getroot()

        # Reference field reading from BaseAviary._parseURDFParameters()
        M  = float(tree[1][0][1].attrib['value'])
        L  = float(tree[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(tree[0].attrib['thrust2weight'])
        IXX = float(tree[1][0][2].attrib['ixx'])
        IYY = float(tree[1][0][2].attrib['iyy'])
        IZZ = float(tree[1][0][2].attrib['izz'])
        J   = np.diag([IXX, IYY, IZZ])
        KF  = float(tree[0].attrib['kf'])
        KM  = float(tree[0].attrib['km'])
        MAX_SPEED_KMH = float(tree[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(tree[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS   = float(tree[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(tree[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z  = float(tree[0].attrib['drag_coeff_z'])
        DRAG_COEFF    = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])

        return dict(
            M=M, L=L, THRUST2WEIGHT_RATIO=THRUST2WEIGHT_RATIO, J=J,
            KF=KF, KM=KM, MAX_SPEED_KMH=MAX_SPEED_KMH,
            GND_EFF_COEFF=GND_EFF_COEFF, PROP_RADIUS=PROP_RADIUS,
            DRAG_COEFF=DRAG_COEFF
        )

    # ===== Rest of dynamics and observation: identical to previous version (omitted) =====
    @staticmethod
    def _quat_normalize(q):  # q: (...,4) as [w,x,y,z]
        return q / (q.norm(dim=-1, keepdim=True) + 1e-9)

    @staticmethod
    def _quat_to_rotmat(q):  # (B,1,4)->(B,1,3,3)
        qw, qx, qy, qz = q.unbind(-1)
        tx, ty, tz = 2*qx, 2*qy, 2*qz
        twx, twy, twz = tx*qw, ty*qw, tz*qw
        txx, txy, txz = tx*qx, ty*qx, tz*qx
        tyy, tyz, tzz = ty*qy, tz*qy, tz*qz
        r00 = 1 - (tyy + tzz); r01 = txy - twz; r02 = txz + twy
        r10 = txy + twz; r11 = 1 - (txx + tzz); r12 = tyz - twx
        r20 = txz - twy; r21 = tyz + twx; r22 = 1 - (txx + tyy)
        R = torch.stack([
            torch.stack([r00,r01,r02], -1),
            torch.stack([r10,r11,r12], -1),
            torch.stack([r20,r21,r22], -1)
        ], -2)
        return R

    @staticmethod
    def _yaw_from_quat(q):  # (B,1,4)->(B,1,1)
        w,x,y,z = q.unbind(-1)
        num = 2*(w*z + x*y)
        den = 1 - 2*(y*y + z*z)
        return torch.atan2(num, den).unsqueeze(-1)

    def _h(self, x):
        if self.output_mode == "full":
            return x
        p = x[..., 0:3]
        q = x[..., 6:10]
        psi = self._yaw_from_quat(q)
        return torch.cat([p, psi], dim=-1)

    def _forces_and_torques_from_rpm(self, rpm):
        forces   = (rpm**2) * self.KF
        torquesz = (rpm**2) * self.KM
        spin = torch.tensor([-1.0, -1.0, 1.0, 1.0], device=rpm.device).view(1,1,4)
        z_torque = torch.sum(spin * torquesz, dim=-1, keepdim=True)
        Ls2 = self.L / math.sqrt(2.0)
        f0,f1,f2,f3 = [forces[...,i:i+1] for i in range(4)]
        x_torque = - (f0 + f1 - f2 - f3) * Ls2
        y_torque =   (-f0 + f1 + f2 - f3) * Ls2
        # x_torque = -1 * x_torque  # x 轴反向
        # y_torque = -1 * y_torque  # y 轴反向
        tau_body  = torch.cat([x_torque, y_torque, z_torque], dim=-1)
        thrust_body = torch.zeros_like(tau_body); thrust_body[...,2:3] = torch.sum(forces, dim=-1, keepdim=True)
        return thrust_body, tau_body

    def _drag_world(self, v_world, rpm):
        if not self.enable_drag:
            return torch.zeros_like(v_world)
        omega_sum = torch.sum(2*math.pi*rpm/60.0, dim=-1, keepdim=True)
        return - self.Cd.to(v_world.device) * omega_sum * v_world

    def _ground_effect_world(self, p_world, rpm, R):
        if not self.enable_ground_effect:
            return torch.zeros_like(p_world)
        h = torch.clamp(p_world[...,2:3], min=self.GND_EFF_H_CLIP)
        gnd_per_prop = (rpm**2) * self.KF * self.gnd_eff_coeff * (self.prop_radius / (4.0*h))**2
        F_gnd_body = torch.zeros_like(p_world)
        F_gnd_body[...,2:3] = torch.sum(gnd_per_prop, dim=-1, keepdim=True)
        F_gnd_world = torch.matmul(R, F_gnd_body.unsqueeze(-1)).squeeze(-1)
        return F_gnd_world

    def forward(self, x, u_rpm):
        B = x.shape[0]; dt = self.dt
        J = self.J.to(x.device).view(1,1,3,3)
        p = x[..., 0:3]; v = x[..., 3:6]; q = x[..., 6:10]; w = x[...,10:13]
        q = self._quat_normalize(q)
        R = self._quat_to_rotmat(q)

        u_rpm = torch.clamp(u_rpm, min=0.0, max=self.MAX_RPM)
        thrust_body, tau_body = self._forces_and_torques_from_rpm(u_rpm)

        F_thrust_world = torch.matmul(R, thrust_body.unsqueeze(-1)).squeeze(-1)
        F_drag_world   = self._drag_world(v, u_rpm)
        F_gnd_world    = self._ground_effect_world(p, u_rpm, R)
        F_world = F_thrust_world + self.g_vec.to(x.device) * self.m + F_drag_world + F_gnd_world

        a = F_world / self.m
        v_next = v + dt * a
        p_next = p + dt * v_next

        Jw = torch.matmul(J, w.unsqueeze(-1)).squeeze(-1)
        w_dot = torch.matmul(J.inverse(), (tau_body - torch.cross(w, Jw, dim=-1)).unsqueeze(-1)).squeeze(-1)
        w_next = w + dt * w_dot

        qw,qx,qy,qz = q.unbind(-1)
        zero = torch.zeros_like(qw)
        Omega = torch.stack([
            torch.stack([zero, -w[...,0], -w[...,1], -w[...,2]], -1),
            torch.stack([w[...,0], zero,  w[...,2], -w[...,1]], -1),
            torch.stack([w[...,1], -w[...,2], zero,  w[...,0]], -1),
            torch.stack([w[...,2],  w[...,1], -w[...,0], zero], -1)
        ], -2)
        q_dot  = 0.5 * torch.matmul(Omega, q.unsqueeze(-1)).squeeze(-1)
        q_next = self._quat_normalize(q + dt * q_dot)

        x_next = torch.cat([p_next, v_next, q_next, w_next], dim=-1)
        y_next = self._h(x_next)
        return x_next, y_next

    def noisy_forward(self, x, u_rpm, output_noise_std: float = 0.0):
        x_next, y_next = self.forward(x, u_rpm)
        if output_noise_std and output_noise_std > 0:
            y_next = y_next + torch.randn_like(y_next) * output_noise_std
        return x_next, y_next

    # def run(self, x0, u_ext, output_noise_std):
    #     B, T = u_ext.shape[0], u_ext.shape[1]
    #     if x0.ndim == 1: x = x0.view(1,1,-1).expand(B,1,-1)
    #     elif x0.shape == (1,1,self.state_dim): x = x0.expand(B,1,-1)
    #     elif x0.shape == (B,1,self.state_dim): x = x0.clone()
    #     elif x0.shape == (1,self.state_dim): x = x0.unsqueeze(1).expand(B,1,-1)
    #     else: raise ValueError("x0 shape not supported")
    #     y_traj = torch.zeros(B,T,self.output_dim, device=u_ext.device, dtype=x.dtype)
    #     for t in range(T):
    #         u = u_ext[:,t:t+1,:]
    #         x, y = self.noisy_forward(x, u, output_noise_std)
    #         y_traj[:,t:t+1,:] = y
    #     return y_traj

    # def __call__(self, x0, u_ext, output_noise_std):
    #     return self.run(x0, u_ext, output_noise_std)
    






class DroneControllerK(nn.Module):
    """
    DSLPID-style controller that outputs motor RPMs (B,1,4).
    Default use_urdf=True, will use the same m/KF/thrust2weight_ratio as Plant to derive MAX_RPM,
    so that the controller and plant have consistent saturation boundaries.
    """
    def __init__(
        self,
        output_mode: str = "full",
        g: float = 9.81,

        # -- New: URDF auto-loading -- 
        use_urdf: bool = True,
        urdf_name: str = _DEFAULT_URDF_NAME,

        # Fallback parameters (consistent with DSLPIDControl)
        KF: float = 3.16e-10,
        m: float = 0.027,
        thrust2weight_ratio: float = 1.9,

        PWM2RPM_SCALE: float = 0.2685,
        PWM2RPM_CONST: float = 4070.3,
        MIN_PWM: float = 20000.0,
        MAX_PWM: float = 65535.0,

        # mixer (CF2X)
        mixer_matrix: torch.Tensor | None = None,

        # Position loop PID
        P_FOR=(0.4, 0.4, 1.25),
        I_FOR=(0.05, 0.05, 0.05),
        D_FOR=(0.2, 0.2, 0.5),
        # Attitude loop PID
        P_TOR=(70000., 70000., 60000.),
        I_TOR=(0, 0, 500.0),
        D_TOR=(20000., 20000., 12000.),

        p_d=(0, 0.0, 1.0),
        psi_d: float = 0.0,
        control_timestep: float = 1/240
    ):
        super().__init__()
        self.output_mode = output_mode
        self.g   = float(g)
        # -- URDF / Fallback -- 
        if use_urdf:
            params = DronePlant._load_urdf_params(urdf_name)
            _m  = float(params["M"])
            _KF = float(params["KF"])
            _t2w= float(params["THRUST2WEIGHT_RATIO"])
        else:
            _m, _KF, _t2w = float(m), float(KF), float(thrust2weight_ratio)

        self.m   = _m  # Save mass for thrust calculation
        self.KF  = _KF
        self.PWM2RPM_SCALE = float(PWM2RPM_SCALE)
        self.PWM2RPM_CONST = float(PWM2RPM_CONST)
        self.MIN_PWM = float(MIN_PWM); self.MAX_PWM = float(MAX_PWM)
        # MAX_RPM consistent with Plant
        self.MAX_RPM = (( _t2w * self.g * _m ) / (4.0 * self.KF))**0.5

        # PID parameters
        self.register_buffer("P_FOR", torch.tensor(P_FOR).view(1,1,3))
        self.register_buffer("I_FOR", torch.tensor(I_FOR).view(1,1,3))
        self.register_buffer("D_FOR", torch.tensor(D_FOR).view(1,1,3))
        self.register_buffer("P_TOR", torch.tensor(P_TOR).view(1,1,3))
        self.register_buffer("I_TOR", torch.tensor(I_TOR).view(1,1,3))
        self.register_buffer("D_TOR", torch.tensor(D_TOR).view(1,1,3))

        # CF2X mixer (same as DSLPIDControl)
        if mixer_matrix is None:
            M = torch.tensor([[-.5, -.5, -1.0],
                              [-.5,  .5,  1.0],
                              [ .5,  .5, -1.0],
                              [ .5, -.5,  1.0]], dtype=torch.float32)
            # M[:,0] = -1 * M[:,0]  # x 轴反向
            # M[:,1] = -1 * M[:,1]  # y 轴反向
            # M[:, 1] = -1*M[:, 0]  # 交换 x,y 轴
            # M[:, 0] = -1*M[:, 1]  # x 轴反向
            self.register_buffer("MIXER", M.view(1,4,3))
        else:
            self.register_buffer("MIXER", mixer_matrix.view(1,4,3))

        self.register_buffer("p_d", torch.tensor(p_d).view(1,1,3))
        self.psi_d = float(psi_d)
        self.control_timestep = float(control_timestep)

        self._int_pos = None
        self._int_rot = None
        self._last_rpy = None

    def reset(self, batch_size: int, device):
        self._int_pos = torch.zeros(batch_size,1,3, device=device)
        self._int_rot = torch.zeros(batch_size,1,3, device=device)
        self._last_rpy = torch.zeros(batch_size,1,3, device=device)

    def _yaw_to_frame(self, psi, B, dev):
        c, s = torch.cos(psi), torch.sin(psi)
        R = torch.zeros((B,1,3,3), device=dev)
        R[...,0,0], R[...,0,1], R[...,0,2] =  c.squeeze(-1), -s.squeeze(-1), 0.0
        R[...,1,0], R[...,1,1], R[...,1,2] =  s.squeeze(-1),  c.squeeze(-1), 0.0
        R[...,2,0], R[...,2,1], R[...,2,2] =  0.0, 0.0, 1.0
        return R

    def forward(self, y):
        """
        Input y:
          - "full" -> (B,1,13): [p,v,q,w]
          - "pos_yaw" -> (B,1,4): [p, psi]
        Output RPM: (B,1,4)
        """
        B = y.shape[0]; dev = y.device; dt = self.control_timestep

        if self.output_mode == "full":
            p = y[...,0:3]; v = y[...,3:6]; q = y[...,6:10]; w = y[...,10:13]
            qw,qx,qy,qz = q.unbind(-1)
            tx,ty,tz = 2*qx,2*qy,2*qz
            twx,twy,twz = tx*qw,ty*qw,tz*qw
            txx,txy,txz = tx*qx,ty*qx,tz*qx
            tyy,tyz,tzz = ty*qy,tz*qy,tz*qz
            r00 = 1-(tyy+tzz); r01=txy-twz; r02=txz+twy
            r10 = txy+twz;    r11=1-(txx+tzz); r12=tyz-twx
            r20 = txz-twy;    r21=tyz+twx;    r22=1-(txx+tyy)
            R = torch.stack([
                torch.stack([r00,r01,r02], -1),
                torch.stack([r10,r11,r12], -1),
                torch.stack([r20,r21,r22], -1)
            ], -2)
            psi = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)).unsqueeze(-1)
        else:
            p = y[...,0:3]; v = torch.zeros_like(p)
            psi = y[...,3:4]
            R = self._yaw_to_frame(psi, B, dev)
            w = torch.zeros_like(p)

        # Position loop PID
        e_pos = self.p_d.to(dev) - p
        e_vel = -v
        if self._int_pos is None or self._int_pos.shape != e_pos.shape:
            self._int_pos = torch.zeros_like(e_pos)
        self._int_pos = torch.clamp(self._int_pos + e_pos*dt, min=-2.0, max=2.0)
        self._int_pos[...,2] = torch.clamp(self._int_pos[...,2], min=-0.15, max=0.15)

        # target_thrust_vec is the target acceleration vector (m/s²)
        target_thrust_vec = self.P_FOR*e_pos + self.I_FOR*self._int_pos + self.D_FOR*e_vel + torch.tensor([0,0, self.g], device=dev).view(1,1,3)
        
        # Project to body Z-axis to get scalar thrust acceleration, then multiply by mass to convert to force (N)
        scalar_thrust = torch.clamp((target_thrust_vec * R[...,2]).sum(dim=-1, keepdim=True) * self.m , min=0.0)

        pwm_base = torch.clamp(
            (torch.sqrt(scalar_thrust / (4.0*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE,
            min=self.MIN_PWM, max=self.MAX_PWM
        )

        # Construct target orientation (z-axis aligned with thrust; heading=psi_d)
        z_d = target_thrust_vec / (target_thrust_vec.norm(dim=-1, keepdim=True) + 1e-9)
        x_c = torch.tensor([math.cos(self.psi_d), math.sin(self.psi_d), 0.0], device=dev).view(1,1,3).expand_as(z_d)  # psi_d=0
        y_d = torch.cross(z_d, x_c, dim=-1); y_d = y_d/(y_d.norm(dim=-1, keepdim=True)+1e-9)
        x_d = torch.cross(y_d, z_d, dim=-1)
        R_d = torch.stack([x_d, y_d, z_d], dim=-1)

        e_R = 0.5*(torch.matmul(R_d.transpose(-1,-2), R) - torch.matmul(R.transpose(-1,-2), R_d))
        rot_e = torch.stack([e_R[...,2,1], e_R[...,0,2], e_R[...,1,0]], dim=-1)
        if self.output_mode == "full":
            rpy_rates_e = -w
            if self._last_rpy is None: self._last_rpy = torch.zeros_like(rot_e)
        else:
            cur_rpy = torch.zeros_like(rot_e); cur_rpy[...,2] = psi.squeeze(-1)
            if self._last_rpy is None: self._last_rpy = cur_rpy.clone()
            rpy_rates_e = -(cur_rpy - self._last_rpy) / max(dt, 1e-6)
            self._last_rpy = cur_rpy.detach()

        if self._int_rot is None or self._int_rot.shape != rot_e.shape:
            self._int_rot = torch.zeros_like(rot_e)
        self._int_rot = torch.clamp(self._int_rot - rot_e*dt, min=-1500.0, max=1500.0)
        self._int_rot[...,0:2] = torch.clamp(self._int_rot[...,0:2], min=-1.0, max=1.0)

        target_torques = - self.P_TOR*rot_e + self.D_TOR*rpy_rates_e + self.I_TOR*self._int_rot
        target_torques = torch.clamp(target_torques, min=-3200.0, max=3200.0)

        mix = torch.matmul(self.MIXER.to(dev), target_torques.transpose(-1,-2)).transpose(-1,-2)  # (B,1,4)

        pwm = torch.clamp(pwm_base + mix, min=self.MIN_PWM, max=self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
        rpm = torch.clamp(rpm, min=0.0, max=self.MAX_RPM).to(dev)
        return rpm
    






class ClosedLoopSystem(nn.Module):
    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, system_model, controller, negative: bool = False):
        super().__init__()
        self.system_model = system_model
        self.controller = controller
        self.negative = negative

        if hasattr(self.system_model, "noisy_forward"):
            self.system_model_tipe = "real_sys"
        else:
            self.system_model_tipe = "REN"

    def forward(self, y, u_ext):
        """
        Compute the next state and output of the system.

        Args:
            u_ext (torch.Tensor): external input at t. shape = (batch_size, 1, input_dim)
            y (torch.Tensor): plant's output at t. shape = (batch_size, 1, output_dim)

        Returns:
            torch.Tensor, torch.Tensor: Input of plant and next output at t+1. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        control_u = self.controller.forward(y)  # Compute control input
        u = control_u + u_ext
        x = y
        if self.system_model_tipe == "real_sys":
            x, y = self.system_model.forward(x, u)
        elif self.system_model_tipe == "REN":
            y = self.system_model.forward(u)
        return u, y

    def noisy_forward(self, y, u_ext, output_noise_std):
        u, y = self.forward(y, u_ext)
        noise = torch.randn_like(y) * output_noise_std
        y_noisy = y + noise
        return u, y_noisy


    def run(self, x0, u_ext, u_ext_index=None, output_noise_std=0.0):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. 
                                 If u_ext_index is None: Shape = (batch_size, horizon, input_dim)
                                 If u_ext_index is specified: Shape = (batch_size, horizon, 1)
            u_ext_index (int, optional): If specified, apply u_ext only to this input channel
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        
        batch_size = u_ext.shape[0]
        horizon = u_ext.shape[1]
        # self.controller.reset(batch_size, u_ext.device)

        if self.system_model_tipe == "real_sys":
            output_dim = self.system_model.output_dim
            input_dim = self.system_model.input_dim
            state_dim = self.system_model.state_dim

            # Handle different x0 formats more comprehensively
            if x0.shape == (batch_size, 1, state_dim):
                x = x0.clone()
            elif x0.shape == (1, 1, state_dim):
                x = x0.expand(batch_size, 1, state_dim).clone()
            elif x0.shape == torch.Size([state_dim]):
                x = x0.view(1, 1, -1).expand(batch_size, 1, state_dim).clone()
            elif x0.shape == torch.Size([1]):
                x = x0.view(1,1,-1).expand(batch_size, 1, state_dim).clone()
            elif x0.shape == (1, state_dim):
                x = x0.unsqueeze(1).expand(batch_size, 1, state_dim).clone()
            else:
                raise ValueError(f'Wrong shape of initial conditions: {x0.shape}. Expected: (B,1,{state_dim}), (1,1,{state_dim}), ({state_dim},), (1,{state_dim}), or (1,)')
                x = None
            
            # Reset controller if it has a reset method
            if hasattr(self.controller, 'reset'):
                self.controller.reset(batch_size=batch_size, device=x.device)

            # Initial output (measurement)
            y = self.system_model._h(x)
            if output_noise_std and output_noise_std > 0:
                y = y + torch.randn_like(y) * output_noise_std

            y_traj = []
            u_traj = []

            for t in range(horizon):
                control_u = self.controller.forward(y)
                # if getattr(self, "negative", False):
                control_u = control_u
                
                # Handle external disturbance
                if u_ext_index is not None:
                    # Add disturbance only to specified channel
                    u = control_u.clone()
                    u_ext_current = u_ext[:, t:t + 1, :].squeeze(-1)  # (batch_size, 1)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_squared = torch.sign(u_ext_current) * (u_ext_current ** 2)
                    u[:, :, u_ext_index] = u[:, :, u_ext_index] + u_ext_squared
                else:
                    # Add disturbance to all channels (original behavior)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_t = u_ext[:, t:t + 1, :]
                    u_ext_squared = torch.sign(u_ext_t) * (u_ext_t ** 2)
                    u = control_u + u_ext_squared

                # [PATCH B1] Clamp RPM after superimposing external disturbance and control to [0, MAX_RPM]
                if hasattr(self.system_model, "MAX_RPM"):
                    u = torch.clamp(u, min=0.0, max=self.system_model.MAX_RPM)

                x, y = self.system_model.noisy_forward(x, u, output_noise_std)
                y_traj.append(y)
                u_traj.append(u)

            y_traj = torch.cat(y_traj, dim=1)  # (B,T,output_dim)
            u_traj = torch.cat(u_traj, dim=1)  # (B,T,input_dim)

        elif self.system_model_tipe == "REN":
            output_dim = self.system_model.dim_out
            input_dim = self.system_model.dim_in

            self.system_model.reset()
            y = self.system_model.y_init.detach().clone().repeat(batch_size, 1, 1)

            # Storage for trajectories
            y_traj = []
            u_traj = []

            for t in range(horizon):
                control_u = self.controller.forward(y)  # Compute control input

                # if self.negative:
                control_u = control_u
                
                # Handle external disturbance
                if u_ext_index is not None:
                    # Add disturbance only to specified channel
                    u = control_u.clone()
                    u_ext_current = u_ext[:, t:t + 1, :].squeeze(-1)  # (batch_size, 1)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_squared = torch.sign(u_ext_current) * (u_ext_current ** 2)
                    u[:, :, u_ext_index] = u[:, :, u_ext_index] + u_ext_squared
                else:
                    # Add disturbance to all channels (original behavior)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_t = u_ext[:, t:t + 1, :]
                    u_ext_squared = torch.sign(u_ext_t) * (u_ext_t ** 2)
                    u = control_u + u_ext_squared

                y = y + torch.randn_like(y) * output_noise_std
                y_traj.append(y)  # Store output
                u_traj.append(u)  # Store input
                y = self.system_model.forward(u)

            y_traj = torch.cat(y_traj, dim=1)
            u_traj = torch.cat(u_traj, dim=1)
        else:
            output_dim = self.system_model.sys.dim_out
            input_dim = self.system_model.sys.dim_in

            self.system_model.sys.reset()
            y = self.system_model.sys.y_init.detach().clone().repeat(batch_size, 1, 1)

            y_traj = []
            u_traj = []

            for t in range(horizon):
                control_u = self.controller.forward(y)
                
                # Handle external disturbance
                if u_ext_index is not None:
                    # Add disturbance only to specified channel
                    u = control_u.clone()
                    u_ext_current = u_ext[:, t:t + 1, :].squeeze(-1)  # (batch_size, 1)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_squared = torch.sign(u_ext_current) * (u_ext_current ** 2)
                    u[:, :, u_ext_index] = u[:, :, u_ext_index] + u_ext_squared
                else:
                    # Add disturbance to all channels (original behavior)
                    # Square the disturbance while preserving sign: sign(x) * x^2
                    u_ext_t = u_ext[:, t:t + 1, :]
                    u_ext_squared = torch.sign(u_ext_t) * (u_ext_t ** 2)
                    u = control_u + u_ext_squared

                y_traj.append(y)
                u_traj.append(u)
                y = self.system_model(u)

            y_traj = torch.cat(y_traj, dim=1)
            u_traj = torch.cat(u_traj, dim=1)

        return u_traj, y_traj

    def __call__(self, x0, u_ext, u_ext_index, output_noise_std):
        return self.run(x0, u_ext, u_ext_index, output_noise_std)

