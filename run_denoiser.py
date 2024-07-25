import os
import argparse
import gym
import glob
import h5py
import copy
import shutil

import numpy as np

import d3rlpy
from d3rlpy.online.explorers import NormalNoise
from d3rlpy.preprocessing.scalers import StandardScaler
from d3rlpy.adversarial_training.utility import set_name_wandb_project
from d3rlpy.adversarial_training.utility import get_stats_from_ckpt
from d3rlpy.iterators.round_iterator import RoundIterator
from d3rlpy.adversarial_training.attackers import random_attack, actor_state_attack, critic_normal_attack
from d3rlpy.models.torch.vector_quantization import AEDenoiser

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='TD3', choices=['TD3', 'SAC'])
parser.add_argument('--dataset', type=str, default='walker2d-medium-v2')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--exp', type=str, default='')
parser.add_argument('--project', type=str, default='WALKER')
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--logdir', type=str, default='d3rlpy_logs')
parser.add_argument('--n_steps', type=int, default=2000000)
parser.add_argument('--n_steps_collect_data', type=int, default=10000000)
parser.add_argument('--n_steps_per_epoch', type=int, default=5000)
parser.add_argument('--eval_interval', type=int, default=10)
parser.add_argument('--save_interval', type=int, default=50)
parser.add_argument('--n_eval_episodes', type=int, default=10)

parser.add_argument('--no_replacement', action='store_true', default=False)
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--stats_update_interval', type=int, default=1000)

parser.add_argument('--loss_type', type=str, default="normal", choices=["normal", "mad_loss"])
parser.add_argument('--attack_type', type=str, default="actor_state_linf")
parser.add_argument('--epsilon', type=float, default=0.05)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--actor_reg', type=float, default=0.5)

# Scope for Vector Quantization representation
SUPPORTED_INPUT_TRANSFORMS = ["vq_ema", "vq_sgd", "vanilla_sgd", "bdr"]
parser.add_argument('--use_input_transformation', action='store_true', default=False)
parser.add_argument('--transformation_type', type=str, default="ema", choices=SUPPORTED_INPUT_TRANSFORMS)

parser.add_argument('--n_embeddings', type=int, default=16)
parser.add_argument('--embedding_dim', type=int, default=1)

parser.add_argument('--vq_loss_weight', type=float, default=1.0)
parser.add_argument('--vq_no_reduction', action='store_true')
parser.add_argument('--autoscale_vq_loss', action='store_true', default=False)
parser.add_argument('--scale_factor', type=float, default=60.0)

parser.add_argument('--n_steps_start_at', type=int, default=0)

parser.add_argument('--bdr_step', type=float, default=0.1)

parser.add_argument('--vq_decay', type=float, default=0.99)
parser.add_argument('--vq_decay_scheduler', action='store_true', default=False)
parser.add_argument('--vq_decay_start_val', type=float, default=0.5)
parser.add_argument('--vq_decay_end_val', type=float, default=0.99)
parser.add_argument('--vq_decay_start_step', type=int, default=0)
parser.add_argument('--vq_decay_end_step', type=int, default=1000000)

parser.add_argument('--finetune', action='store_true')
parser.add_argument('--ckpt', type=str, default='none')
parser.add_argument('--ckpt_steps', type=str, default='model_500000.pt')
parser.add_argument('--load_buffer', action='store_true', default=False)
parser.add_argument('--backup_file', action='store_true')

parser.add_argument('--ckpt_id', type=int, default=0)

args = parser.parse_args()

N_SEEDS = 5
def process_checkpoint(args):
    if os.path.isfile(args.ckpt):
        print(f"[INFO] Preparing to load checkpoint: {args.ckpt}")
        return args.ckpt

    elif os.path.isdir(args.ckpt):
        entries = os.listdir(args.ckpt)
        entries.sort()

        print("\tFound %d experiments." % (len(entries)))
        ckpt_list = []
        for entry in entries:
            ckpt_file = os.path.join(args.ckpt, entry, args.ckpt_steps)
            if not os.path.isfile(ckpt_file):
                print("\tCannot find checkpoint {} in {}".format(args.ckpt_steps, ckpt_file))
            else:
                ckpt_list.append(ckpt_file)
        assert 1 <= args.seed <= N_SEEDS and len(ckpt_list) == N_SEEDS
        print(f"[INFO] Preparing to load checkpoint: {ckpt_list[args.ckpt_id]}")
        return ckpt_list

    else:
        print("[INFO] Training from scratch.")
        return None

def load_buffer_from_checkpoint(args):
    from d3rlpy.dataset import MDPDataset
    assert os.path.isfile(args.ckpt)
    ckpt_dir = args.ckpt[:args.ckpt.rfind('/')]
    ckpt_step = int(args.ckpt_steps.split('.')[0].split('_')[-1])
    entries = glob.glob(os.path.join(ckpt_dir, "*.h5"))
    entries.sort()
    buffer_path = entries[-1]
    buffer_at_step = int(buffer_path.split('/')[-1].split('_')[-1][:-8])
    total_of_samples = buffer_at_step

    with h5py.File(buffer_path, 'r') as f:
        observations = f['observations'][()]
        actions = f['actions'][()]
        rewards = f['rewards'][()]
        terminals = f['terminals'][()]
        discrete_action = f['discrete_action'][()]

        # for backward compatibility
        if 'episode_terminals' in f:
            episode_terminals = f['episode_terminals'][()]
        else:
            episode_terminals = None

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_terminals=episode_terminals,
        discrete_action=discrete_action,
    )
    print(f"[INFO] Loaded previous buffer (n_episodes={dataset.size()}): {buffer_path}")

    stats_filename = copy.copy(args.ckpt)
    stats_filename = stats_filename.replace('model_', 'stats_')
    stats_filename = stats_filename.replace('.pt', '.npz')
    if os.path.isfile(stats_filename):
        data = np.load(stats_filename)
        mean, std = data['mean'], data['std']
    else:
        raise ValueError("Cannot find statistics of buffer.")

    buffer_state = dict(
        total_samples=total_of_samples,
        obs_mean=mean, obs_std=std,
        obs_sum=mean * total_of_samples,
        obs_sum_sq=(std ** 2 + mean ** 2) * total_of_samples
    )
    return dataset.episodes, buffer_state



def main():
    ckpt_list = process_checkpoint(args)
    exp_name = ckpt_list[args.ckpt_id].split('/')[-2]
    args.ckpt = ckpt_list[args.ckpt_id]
    env = gym.make(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    if args.standardization:
        scaler = StandardScaler(mean=np.zeros(env.observation_space.shape), std=np.ones(env.observation_space.shape))
    else:
        scaler = None


    if args.transformation_type == "vanilla_sgd":
        args.embedding_dim = env.observation_space.shape[0]

    sac = d3rlpy.algos.SAC(
        use_gpu=args.gpu,
        scaler=scaler,
        env_name=args.dataset,
        use_input_transformation=False,
    )
    sac.build_with_env(env)  # Create policy/critic for env, must be performed after fitting scaler

    previous_buffer, buffer_state = None, None
    if args.load_buffer:
        previous_buffer, buffer_state = load_buffer_from_checkpoint(args)

    args.logdir = os.path.join(args.logdir, exp_name)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    shutil.copy("run_denoiser.py", args.logdir)

    # TODO: Training parameters
    batch_size = 512
    n_types_of_noise = 2
    n_epoch_per_config = 20
    # epsilons_attack = [0.05]
    # epsilons_attack = [0.1]   # Hopper
    # epsilons_attack = [0.3]  # Invert Pendulum
    # epsilons_attack = [0.25]  # Reacher
    # epsilons_attack = [0.05]  # Walker
    epsilons_attack = [0.15]    # Ant
    n_epochs = len(epsilons_attack) * n_types_of_noise * n_epoch_per_config # 1 x 4 x 10 (#epsilons x #attacks x #epochs/combination)
    num_steps = 10
    # step_size = epsilon_attack / num_steps

    # TODO: Make data loader
    transitions = []
    for episode in previous_buffer:
        transitions += episode.transitions

    iterator = RoundIterator(
        transitions,
        batch_size=batch_size,
        shuffle=True,
    )

    checkpoint = args.ckpt
    if not os.path.isfile(checkpoint):
        print("Checkpoint is not found: %s" % (checkpoint))
        raise ValueError

    sac._impl.load_model(checkpoint)
    print("Load model successfully.")
    mean_ckpt, std_ckpt = get_stats_from_ckpt(checkpoint)
    if mean_ckpt is not None and std_ckpt is not None and sac._impl.scaler is not None:
        sac._impl.scaler._mean = mean_ckpt
        sac._impl.scaler._std = std_ckpt
        print("Updated statistical from checkpoint.")

    # TODO: Construct model
    denoiser = AEDenoiser(input_dim=env.observation_space.shape[0])
    denoiser = denoiser.cuda()
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=3e-4)


    # Start training
    for epoch in range(1, n_epochs + 1):
        range_gen = tqdm(range(len(iterator)), disable=False, desc=f"Epoch {int(epoch)}/{n_epochs}")
        iterator.reset()
        loss_total = []
        for itr in range_gen:
            batch = next(iterator)
            x = batch.observations
            x = torch.tensor(data=x, dtype=torch.float32, device=sac._impl.device, )
            x = sac._impl.scaler.transform(x)

            epsilon = np.random.choice(epsilons_attack)

            step_size = epsilon / num_steps

            attack_type = np.random.randint(n_types_of_noise)
            if attack_type == 0:
                input_x = x
            elif attack_type == 1:
                input_x = actor_state_attack(x, sac._impl._policy, None,
                                             epsilon, num_steps, step_size,
                                             None, None, clip=False, use_assert=True)
            elif attack_type == 2:
                input_x = random_attack(x, epsilon, None, None, clip=False, use_assert=True)
            elif attack_type == 3:
                input_x = critic_normal_attack(x, sac._impl._policy, sac._impl._q_func, None,
                                               epsilon, num_steps, step_size,
                                               None, None, clip=False, use_assert=True)
            else:
                raise NotImplementedError

            optimizer.zero_grad()
            z = denoiser.encoder(input_x)
            x_rec = denoiser.decoder(z)
            loss = F.mse_loss(x_rec, x)
            loss.backward()
            optimizer.step()

            loss_total.append(loss.item())

        state_dict = {
            'model': denoiser.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state_dict, f"{args.logdir}/denoiser_last.pt")
        print(f"MSE loss: {np.mean(loss_total):.4f}")

if __name__ == '__main__':
    main()
