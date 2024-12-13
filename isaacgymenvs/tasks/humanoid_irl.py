# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.base.vec_task import VecTask

from isaacgymenvs.tasks.RND import RND
from isaacgymenvs.tasks.icm import ICM

import wandb


class HumanoidIRL(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg


        
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 108+3+3
        self.cfg["env"]["numActions"] = 21

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.num_envs <= 64:
            wandb.init(project="irl_project", name='r23', mode='disabled')
        else:
            wandb.init(project="irl_project", name='r23')
            # wandb.init(project="irl_project", name='test', mode='disabled')



        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.ball_targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))

        # rnd
        self.use_rnd = False
        if self.use_rnd:
            # self.rnd = RND([self.cfg["env"]["numObservations"], 128, 128, self.cfg["env"]["numObservations"]], self.device, optimzer_lr=1e-5)
            self.rnd = RND([2, 64, 2], self.device, optimzer_lr=1e-4)
            # self.rnd = RND([self.cfg["env"]["numObservations"], 64, self.cfg["env"]["numObservations"]], self.device, optimzer_lr=1e-4)

        self.use_icm = True
        self.icm = None
        self.icm_optimizer = None
        self.last_obs_for_icm = None
        self.icm_loss_reset_coeff = torch.zeros(self.num_envs, device=self.device)
        assert not self.use_rnd or not self.use_icm, "Only one of RND or ICM can be used at a time"
        if self.use_icm:
            obs_dim = self.cfg["env"]["numObservations"]
            action_dim = self.cfg["env"]["numActions"]

            self.icm = ICM(
                input_dim=obs_dim,
                action_dim=action_dim,
                device=self.device
            )
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=3e-4)

        self.i = 0

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_humanoid.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # ball asset
        # ball_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        # ball_asset_file = "mjcf/ball.xml"
        # ball_asset_path = os.path.join(ball_asset_root, ball_asset_file)
        # ball_asset_root = os.path.dirname(ball_asset_path)
        # ball_asset_file = os.path.basename(ball_asset_path)
        # ball_asset_options = gymapi.AssetOptions()
        # ball_asset_options.angular_damping = 0.01
        # ball_asset_options.max_angular_velocity = 100.0
        # ball_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # ball_asset = self.gym.load_asset(self.sim, ball_asset_root, ball_asset_file, ball_asset_options)

        asset_options = gymapi.AssetOptions()
        asset_options.density = 10.0
        ball_asset = self.gym.create_sphere(self.sim, 0.1, asset_options)

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(0.5, 0.0, 0.1)
        ball_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.ball_handles = []
        self.humanoid_actor_idxs = []
        self.ball_actor_idxs = []


        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, 0)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)
            self.humanoid_actor_idxs.append(self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM))

            # load ball
            ball_handle = self.gym.create_actor(env_ptr, ball_asset, ball_start_pose, "ball", i, 0, 0)
            self.ball_handles.append(ball_handle)
            self.ball_actor_idxs.append(self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM))

        self.humanoid_actor_idxs = torch.Tensor(self.humanoid_actor_idxs).to(device=self.device,dtype=torch.int32)
        self.ball_actor_idxs = torch.Tensor(self.ball_actor_idxs).to(device=self.device,dtype=torch.int32)


        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf = self.compute_humanoid_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.max_motor_effort,
            self.motor_efforts,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.root_states[self.humanoid_actor_idxs, :],
            self.root_states[self.ball_actor_idxs, :],
            self.ball_targets
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_force_tensor(self.sim)

        # self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
        #     self.obs_buf, self.root_states, self.targets, self.potentials,
        #     self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
        #     self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
        #     self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
        #     self.basis_vec0, self.basis_vec1)

        self.targets = self.ball_targets.clone()
        self.targets[:,2] = 1.34

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = self.compute_humanoid_observations(
            self.obs_buf, self.root_states[self.humanoid_actor_idxs, :], self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1, 
            self.root_states[self.ball_actor_idxs, :], self.ball_targets)
        

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        actor_ids = []
        for i in range(env_ids.shape[0]):
            actor_ids.append(2*env_ids[i]) # bot
            actor_ids.append(2*env_ids[i]+1) # ball
        actor_ids = torch.tensor(actor_ids)
        actor_ids_int32 = actor_ids.to(dtype=torch.int32).to(self.device)

        humanoid_ids = (env_ids*2).clone()
        humanoid_ids_int32 = humanoid_ids.to(dtype=torch.int32).to(self.device)
        
        ball_ids = (env_ids*2 + 1).clone()
        ball_ids_int32 = ball_ids.to(dtype=torch.int32).to(self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32).to(self.device)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states), gymtorch.unwrap_tensor(actor_ids_int32), len(actor_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(humanoid_ids_int32), len(humanoid_ids_int32))

        # to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target = self.targets[env_ids] - self.initial_root_states[humanoid_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        self.ball_target_range = 2
        # self.ball_targets[env_ids] = self.initial_root_states[ball_ids_int32, 0:3] + torch.randn_like(self.initial_root_states[ball_ids_int32, 0:3]).to(self.device)*self.ball_target_range
        self.ball_targets[env_ids] = self.initial_root_states[ball_ids_int32, 0:3] + torch.ones_like(self.initial_root_states[ball_ids_int32, 0:3]).to(self.device)*self.ball_target_range
        self.ball_targets[:,2] = 0.1

        self.icm_loss_reset_coeff[env_ids] = 0.0



    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
        # if self.viewer and True:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

        # if True:
        # if False:
        if self.num_envs<=64:
            self.gym.clear_lines(self.viewer)
            sphere_geom = gymutil.WireframeSphereGeometry(0.5, 12, 12, None, color=(1, 0, 0))    
            for i in range(self.num_envs):
                sphere_pose = gymapi.Vec3(float(self.ball_targets[i,0]), float(self.ball_targets[i,1]), float(self.ball_targets[i,2]))
                sphere_pose = gymapi.Transform(sphere_pose, r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
    def compute_humanoid_reward(
        self,
        obs_buf,
        reset_buf,
        progress_buf,
        actions,
        up_weight,
        heading_weight,
        potentials,
        prev_potentials,
        actions_cost_scale,
        energy_cost_scale,
        joints_at_limit_cost_scale,
        max_motor_effort,
        motor_efforts,
        termination_height,
        death_cost,
        max_episode_length, 
        humanoid_root_states,
        ball_root_states,
        ball_targets
    ):
        # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]


        # reward from the direction headed
        heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
        heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

        # reward for being upright
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

        actions_cost = torch.sum(actions ** 2, dim=-1)

        # energy cost reward
        motor_effort_ratio = motor_efforts / max_motor_effort
        scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
        dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

        # reward for duration of being alive
        alive_reward = torch.ones_like(potentials) * 2.0
        progress_reward = potentials - prev_potentials

        total_reward = progress_reward + alive_reward + up_reward + heading_reward - actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

        # total_reward = alive_reward + up_reward + heading_reward - actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost
        # total_reward = progress_reward + alive_reward + up_reward + actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost
        # total_reward = alive_reward + up_reward + actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

        DEBUG_PRINT = False
        if DEBUG_PRINT:
            print('=====')
            print(f'{total_reward.mean()=}')
            print(f'{progress_reward.mean()=}')
            print(f'{alive_reward.mean()=}')
            print(f'{up_reward.mean()=}')
            print(f'{heading_reward.mean()=}')
            print(f'{(-actions_cost_scale*actions_cost).mean()=}')
            print(f'{(-energy_cost_scale * electricity_cost).mean()=}')
            print(f'{-dof_at_limit_cost.mean()=}')


        
        # adjust reward for fallen agents
        total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

        # ball_target =  obs_buf[:,111:114]
        ball_pos_local =  obs_buf[:,108:111]
        ball_target =  ball_targets
        ball_pos =  ball_root_states[:,0:3]
        ball_vel = ball_root_states[:,3:6]

        humanoid_pos = humanoid_root_states[:,0:3]
        humanoid_vel = humanoid_root_states[:,3:6]

        ball_target_distance = torch.norm(ball_target[:,0:2] - ball_pos[:,0:2], dim=1)
        humanoid_ball_distance = torch.norm(humanoid_pos[:,0:2]-ball_pos[:,0:2], dim=1)
        humanoid_ball_veocity = torch.norm(humanoid_vel[:,0:2]-ball_vel[:,0:2], dim=1)

        ball_target_distance_reward =  -ball_target_distance/5*2
        humanoid_ball_distance_reward =  -humanoid_ball_distance/5*2
        humanoid_ball_veocity_reward =  -humanoid_ball_veocity/2 

        if DEBUG_PRINT:
            print('--')
            print(f'{ball_target_distance_reward.mean()=}')
            print(f'{humanoid_ball_distance_reward.mean()=}')
            print(f'{humanoid_ball_veocity_reward.mean()=}')

        total_reward += ball_target_distance_reward
        # total_reward += humanoid_ball_distance_reward
        # total_reward += humanoid_ball_veocity_reward

        if self.use_rnd:
            rnd_reward = self.rnd.compute_reward(ball_pos_local[:,:2])/1
            # rnd_reward = self.rnd.compute_reward(ball_target[:,0:2] - ball_pos[:,0:2])/1
            # rnd_reward = self.rnd.compute_reward(obs_buf)/10
            for _ in range(10):
                # pass
                self.rnd.update(ball_pos_local[:,:2])
                # self.rnd.update(ball_target[:,0:2] - ball_pos[:,0:2])
                # self.rnd.update(obs_buf)
            if self.use_rnd:
                total_reward += rnd_reward

        if self.use_icm:
            assert self.last_obs_for_icm is not None
            with torch.enable_grad():
                intrinsic_reward, loss = self.icm.compute_intrinsic_reward(
                    self.last_obs_for_icm, obs_buf, actions
                )

                intrinsic_reward *= self.icm_loss_reset_coeff
                loss *= self.icm_loss_reset_coeff

                # Accumulate gradients and update periodically
                self.icm_optimizer.zero_grad()
                loss.mean().backward()
                self.icm_optimizer.step()

            self.last_obs_for_icm = obs_buf.detach().clone()

            if DEBUG_PRINT:
                print('--')
                print(f'{intrinsic_reward.mean()}')

            intrinsic_reward_scale = 10 # TODO - we don't seem to use this in lab.
            total_reward += intrinsic_reward_scale * intrinsic_reward.detach()


        if DEBUG_PRINT:
            print('--')
            print(f'{rnd_reward.mean()=}')

        # Sparse massive reward
        # total_reward = torch.where(ball_target_distance <= 0.5, torch.ones_like(total_reward)*100, total_reward)
        sparse_massive_reward = torch.where(ball_target_distance <= 0.5, torch.ones_like(total_reward)*1000, torch.zeros_like(total_reward))
        total_reward += sparse_massive_reward


        self.i += 1
        if self.i%32==0:
            log_dict = {
                "total_reward.mean()=": total_reward.mean(),
                "progress_reward.mean()=": progress_reward.mean(),
                "alive_reward.mean()=": alive_reward.mean(), 
                "up_reward.mean()=": up_reward.mean(), 
                "heading_reward.mean()=": heading_reward.mean(), 
                "(-actions_cost_scale*actions_cost).mean()=": (-actions_cost_scale*actions_cost).mean(), 
                "(-energy_cost_scale * electricity_cost).mean()=": (-energy_cost_scale * electricity_cost).mean(), 
                "-dof_at_limit_cost.mean()=": -dof_at_limit_cost.mean(), 

                "ball_target_distance_reward.mean()=": ball_target_distance_reward.mean(),
                "humanoid_ball_distance_reward.mean()=": humanoid_ball_distance_reward.mean(),
                "humanoid_ball_veocity_reward.mean()=": humanoid_ball_veocity_reward.mean(),

                "sparse_massive_reward.mean()=": sparse_massive_reward.mean(),
                }
            if self.use_rnd:
                log_dict["rnd_reward.mean()="] = rnd_reward.mean()
            if self.use_icm:
                log_dict["icm_reward.mean()="] = intrinsic_reward.mean()
            wandb.log(data=log_dict, step=int(self.i/32))


        # reset agents
        reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
        reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
        reset = torch.where(ball_target_distance <= 0.5, torch.ones_like(reset_buf), reset)

        # ICM should be calculable after this for all envs - in reset_idx() we can turn off any that are
        # being reset.
        self.icm_loss_reset_coeff[:] = 1.0

        return total_reward, reset


# @torch.jit.script
    def compute_humanoid_observations(self, obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                    dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                    sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                    basis_vec0, basis_vec1, ball_states, ball_target):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

        torso_position = root_states[:, 0:3]
        torso_rotation = root_states[:, 3:7]
        velocity = root_states[:, 7:10]
        ang_velocity = root_states[:, 10:13]

        to_target = targets - torso_position
        to_target[:, 2] = 0

        prev_potentials_new = potentials.clone()
        potentials = -torch.norm(to_target, p=2, dim=-1) / dt

        torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
            torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

        vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
            torso_quat, velocity, ang_velocity, targets, torso_position)

        roll = normalize_angle(roll).unsqueeze(-1)
        yaw = normalize_angle(yaw).unsqueeze(-1)
        angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
        dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

        ball_position = ball_states[:, 0:3]
        ball_rotation = ball_states[:, 3:7]
        ball_velocity = ball_states[:, 7:10]
        ball_ang_velocity = ball_states[:, 10:13]

        ball_position_local = ball_position - torso_position
        ball_target_local = ball_target - torso_position

        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21), 3 ball_pos
        obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                        yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                        dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                        sensor_force_torques.view(-1, 12) * contact_force_scale, actions, ball_position_local, ball_target_local), dim=-1)

        if self.last_obs_for_icm is None:
            self.last_obs_for_icm = obs.detach().clone()

        return obs, potentials, prev_potentials_new, up_vec, heading_vec
