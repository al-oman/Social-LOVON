"""
Live-visualization variant of crowd_test.py.

Renders the simulation in real time using plt.ion() so the plot updates
each timestep while policy.predict() runs, rather than recording all
states first and replaying afterward.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import logging
import argparse
import configparser
import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import patches

from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA

from models.lovon_crowd_policy import LOVONCrowdPolicy

policy_factory['lovon'] = LOVONCrowdPolicy


def draw_frame(ax, env, robot, action, step_num, safety_heatmap, heatmap_img):
    """
    Redraw a single simulation frame on the given axes.
    Updates artists in-place where possible to avoid full redraws.
    """
    ax.cla()
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('x(m)', fontsize=12)
    ax.set_ylabel('y(m)', fontsize=12)

    cmap = plt.cm.get_cmap('hsv', 10)
    arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

    # safety heatmap via social_nav (same method deploy.py uses)
    if safety_heatmap:
        social_nav = getattr(robot.policy, 'social_nav', None)
        if social_nav is not None:
            safety_grid, extent = social_nav.get_safety_heatmap(
                xlim=(-6, 6), ylim=(-6, 6))
            if safety_grid is not None:
                ax.imshow(safety_grid, extent=extent, origin='lower',
                          cmap='RdYlGn', vmin=0, vmax=1, alpha=0.5, zorder=0)

    # goal marker
    goal = mlines.Line2D([robot.gx], [robot.gy], color='red', marker='*',
                         linestyle='None', markersize=15, label='Goal')
    ax.add_artist(goal)

    # robot
    robot_circle = plt.Circle(robot.get_position(), robot.radius,
                              fill=True, color='yellow', zorder=3)
    ax.add_artist(robot_circle)

    # robot heading arrow
    r = robot.radius
    theta = robot.theta
    arrow = patches.FancyArrowPatch(
        (robot.px, robot.py),
        (robot.px + r * np.cos(theta), robot.py + r * np.sin(theta)),
        color='red', arrowstyle=arrow_style, zorder=4)
    ax.add_artist(arrow)

    # humans
    for i, human in enumerate(env.humans):
        hc = plt.Circle(human.get_position(), human.radius,
                        fill=False, color=cmap(i), zorder=2)
        ax.add_artist(hc)
        ax.text(human.px - 0.11, human.py - 0.11, str(i),
                color='black', fontsize=12, zorder=5)

    # time + status
    ax.text(-1, 5.5, f'Time: {env.global_time:.2f}', fontsize=14)
    if action is not None:
        ax.text(2, 5.5, f'Step: {step_num}', fontsize=12)

    ax.legend([robot_circle, goal], ['Robot', 'Goal'], fontsize=12, loc='upper left')

    return heatmap_img


def main():
    parser = argparse.ArgumentParser('Live-visualization crowd test')
    parser.add_argument('--env_config', type=str, default='configs/env_lovon.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy_lovon.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=None)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--safety_heatmap', default=False, action='store_true')
    parser.add_argument('--goal', type=str, default='run to bag at 1.0 m/s')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.policy_config

    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)

    if args.policy == 'lovon':
        policy.load_lovon(
            model_path="models/model_language2motion_n1000000_d128_h8_l4_f512_msl64_hold_success",
            tokenizer_path="models/tokenizer_language2motion_n1000000",
            social_nav_enabled=True,
        )
        policy.set_mission(args.goal, args.goal, "handbag")

    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)

    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)

    policy.set_phase(args.phase)
    policy.set_device(device)
    if isinstance(robot.policy, ORCA):
        robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)
    policy.set_env(env)
    robot.print_info()

    # --- live visualization loop ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    ob = env.reset(args.phase, args.test_case)
    done = False
    step_num = 0
    heatmap_img = None

    # draw initial state before first action
    heatmap_img = draw_frame(ax, env, robot, None, step_num,
                             args.safety_heatmap, heatmap_img)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.5)

    while not done:
        # compute action (this is where L2MM + social nav runs)
        action = robot.act(ob)
        print(action)
        ob, _, done, info = env.step(action)
        step_num += 1

        # redraw with updated positions
        heatmap_img = draw_frame(ax, env, robot, action, step_num,
                                 args.safety_heatmap, heatmap_img)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    logging.info('It takes %.2f seconds to finish. Final status is %s',
                 env.global_time, info)
    if robot.visible and info == 'reach goal':
        human_times = env.get_human_times()
        logging.info('Average time for humans to reach goal: %.2f',
                     sum(human_times) / len(human_times))

    # keep window open until closed
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
