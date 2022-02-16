import logging
from igibson.tasks.task_base import BaseTask
import pybullet as p
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.termination_conditions.max_collision import MaxCollision
from igibson.termination_conditions.timeout import Timeout
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.point_goal import PointGoal
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward


from igibson.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from igibson.objects.visual_marker import VisualMarker
from igibson.objects.ycb_object import YCBObject

import numpy as np


class GoToObjectTask(BaseTask):
    """
    Point Nav Fixed Task
    The goal is to navigate to a fixed goal position
    """

    def __init__(self, env):
        print("GoToObjectTask: init")
        super(GoToObjectTask, self).__init__(env)
        self.reward_type = self.config.get("reward_type", "l2")
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]
        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]
        self.initial_pos = np.array(self.config.get("initial_pos", [0, 0, 0]))
        self.initial_orn = np.array(self.config.get("initial_orn", [0, 0, 0]))
        self.target_pos = np.array([99, 99, -99])
        self.goal_format = self.config.get("goal_format", "polar")
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.goal_object = self.load_goal_object(
            env, "003_cracker_box", [[-1, -1], [1, 1]]
        )  # TODO: Make yaml config property for area and object args

        self.visible_target = self.config.get("visible_target", False)
        self.visible_path = self.config.get("visible_path", False)
        self.floor_num = 0

        print("[GoToObject init] configs:", f"{self.config.get('visible_target')=}")

        self.load_visualization(env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        print("Env mode=", env.mode)
        if env.mode != "gui":
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )
        self.target_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        #if self.target_visual_object_visible_to_agent:
        env.simulator.import_object(self.initial_pos_vis_obj)
        env.simulator.import_object(self.target_pos_vis_obj)
        #else:
        #    self.initial_pos_vis_obj.load(env.simulator)
        #    self.target_pos_vis_obj.load(env.simulator)

        # The visual object indicating the initial location is always hidden
        for instance in self.initial_pos_vis_obj.renderer_instances:
            instance.hidden = True

                # The visual object indicating the target location may be visible
        for instance in self.target_pos_vis_obj.renderer_instances:
            instance.hidden = not self.visible_target

        if env.scene.build_graph:
            self.num_waypoints_vis = 250
            self.waypoints_vis = [
                VisualMarker(
                    visual_shape=p.GEOM_CYLINDER,
                    rgba_color=[0, 1, 0, 0.3],
                    radius=0.1,
                    length=cyl_length,
                    initial_offset=[0, 0, cyl_length / 2.0],
                )
                for _ in range(self.num_waypoints_vis)
            ]
            for waypoint in self.waypoints_vis:
                env.simulator.import_object(waypoint)
                # The path to the target may be visible
                for instance in waypoint.renderer_instances:
                    instance.hidden = not self.visible_path

    def get_geodesic_potential(self, env):
        """
        Get potential based on geodesic distance

        :param env: environment instance
        :return: geodesic distance to the target position
        """
        _, geodesic_dist = self.get_shortest_path(env)
        return geodesic_dist

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2], self.target_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == "l2":
            return self.get_l2_potential(env)
        elif self.reward_type == "geodesic":
            return self.get_geodesic_potential(env)

    def load_goal_object(self, env, obj_path, area=None):
        """
        Load the in the desired object at a semi-random position

            area: Array of [[x0, y0],[x1, y1]] representing the 2d area where the object is allowed to spawn
        """
        obj = YCBObject(obj_path)
        env.simulator.import_object(obj)
        self.reset_goal(env, obj, area)

        return obj

    def reset_goal(self, env, obj, area=None):
        """
        Attemps to place the goal object in a random position within the area.
        """
        pos = []
        reset_success = False
        orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        max_trials = 100
        state_id = -1

        if area is None:

            pos = [1, 1, 0.1]
            reset_success = env.test_valid_position(obj, pos, orn)
        else:
            state_id = p.saveState()
            for _ in range(max_trials):
                x = np.random.uniform(low=area[0][0], high=area[1][0])
                y = np.random.uniform(low=area[0][1], high=area[1][1])
                pos = [x, y, 0.1]
                # print("DEBUG: The tried position is=", pos)
                reset_success = env.test_valid_position(obj, pos, orn)
                # print("DEBUG: Random pos: reset_success=", reset_success)
                p.restoreState(state_id)
                if reset_success:
                    break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")
        p.removeState(state_id)
        env.land(obj, pos, orn)
        self.target_pos = np.array(pos)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

        self.reset_goal(env, self.goal_object, [[-1, -1], [1, 1]])

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """
        print("Resetting agent")
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
        self.path_length = 0.0
        self.robot_pos = self.initial_pos[:2]
        self.geodesic_dist = self.get_geodesic_potential(env)
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggregate termination conditions and fill info
        """
        done, info = super(GoToObjectTask, self).get_termination(env, collision_links, action, info)

        info["path_length"] = self.path_length
        if done:
            info["spl"] = float(info["success"]) * min(1.0, self.geodesic_dist / self.path_length)
        else:
            info["spl"] = 0.0

        return done, info

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(), *env.robots[0].get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == "polar":
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))
        
        

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(
            env.robots[0].get_linear_velocity(), *env.robots[0].get_rpy()
        )[0]
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(
            env.robots[0].get_angular_velocity(), *env.robots[0].get_rpy()
        )[2]
        task_obs = np.append(task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def get_shortest_path(self, env, from_initial_pos=False, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        if from_initial_pos:
            source = self.initial_pos[:2]
        else:
            source = env.robots[0].get_position()[:2]
        target = self.target_pos[:2]
        return env.scene.get_shortest_path(self.floor_num, source, target, entire_path=entire_path)

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != "gui":
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)

        if env.scene.build_graph:
            shortest_path, _ = self.get_shortest_path(env, entire_path=True)
            floor_height = env.scene.get_floor_height(self.floor_num)
            num_nodes = min(self.num_waypoints_vis, shortest_path.shape[0])
            for i in range(num_nodes):
                self.waypoints_vis[i].set_position(
                    pos=np.array([shortest_path[i][0], shortest_path[i][1], floor_height])
                )
            for i in range(num_nodes, self.num_waypoints_vis):
                self.waypoints_vis[i].set_position(pos=np.array([0.0, 0.0, 100.0]))

    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
