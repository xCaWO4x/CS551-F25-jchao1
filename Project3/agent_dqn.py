#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.args = args
        self.num_actions = env.action_space.n
        self.resume_from_model = getattr(args, 'resume_from_model', False)
        self.resume_steps = max(0, getattr(args, 'resume_steps', 0))
        
        # Device setup (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize Q-network and target network
        self.q_network = DQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_network = DQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)
        
        # Learning rate schedule (step-count based)
        # 0   - 2.5M steps: args.learning_rate (default 2.5e-4)
        # 2.5M-5.0M steps: 1.75e-4
        # 5.0M-10.0M steps: 1.0e-4
        self.lr_schedule = [
            (0, args.learning_rate),
            (2_500_000, 1.75e-4),
            (5_000_000, 1.0e-4),
        ]
        self.current_lr = None
        
        # Experience Replay buffer
        buffer_size = args.replay_buffer_size
        self.buffer_size = buffer_size
        self.buffer_ptr = 0
        self.buffer_count = 0
        
        # Pre-allocate arrays (much more memory efficient than deque)
        self.states = np.zeros((buffer_size, 84, 84, 4), dtype=np.uint8)
        self.next_states = np.zeros((buffer_size, 84, 84, 4), dtype=np.uint8)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        
        # Training parameters
        # Step-based epsilon decay (not episode-based)
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon = self.epsilon_start  # Will be updated based on steps
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq
        self.train_start = args.train_start
        
        # Training statistics
        self.step_count = self.resume_steps
        self.episode_count = 0
        self.train_step_count = 0  # Track number of training steps
        self.loss_history = deque(maxlen=100)  # Track recent losses
        if self.resume_steps > 0:
            print(f"[RESUME] Initializing step counter at {self.step_count:,}.")
            sys.stdout.flush()
        
        if args.test_dqn or self.resume_from_model:
            mode = "testing" if args.test_dqn else "resume"
            print(f'Loading trained model for {mode}...')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            if os.path.exists(args.model_path):
                # Get file modification time before loading
                from datetime import datetime
                mod_time = os.path.getmtime(args.model_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                checkpoint = torch.load(args.model_path, map_location=self.device)
                self.q_network.load_state_dict(checkpoint)
                self.target_network.load_state_dict(checkpoint)
                print(f"Model loaded from {args.model_path}")
                print(f"  Checkpoint last modified: {mod_time_str} (timestamp: {mod_time})")
            else:
                print(f"Warning: Model file {args.model_path} not found!")
                if self.resume_from_model:
                    print("Resume requested but checkpoint missing; starting from scratch.")
            
        # Ensure optimizer starts with correct LR
        self._apply_lr_for_step(self.step_count)

    def _apply_lr_for_step(self, step):
        """Adjust optimizer learning rate based on the step-count schedule."""
        for threshold, lr in reversed(self.lr_schedule):
            if step >= threshold:
                if self.current_lr != lr:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                    self.current_lr = lr
                    print(f"[LR SCHEDULE] Step {step:,} -> setting learning rate to {lr:.6f}", flush=True)
                    sys.stdout.flush()
                break

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Epsilon-greedy action selection
        if test:
            # During testing, use greedy policy (no exploration)
            epsilon = 0.0
        else:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            # Random action (exploration)
            action = random.randrange(self.num_actions)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                # Convert observation to tensor and normalize: (84, 84, 4) -> (4, 84, 84)
                # CRITICAL: Must normalize same as training! Divide by 255.0 to match sample_batch
                obs_tensor = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()
        
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # Efficient storage: store directly in pre-allocated arrays
        # Convert to uint8 [0, 255] properly handling both uint8 and float32 inputs
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        
        # Handle float32 [0, 1] inputs: multiply by 255 before converting to uint8
        # Direct casting float32 [0,1] -> uint8 would destroy information (0.5 -> 0)
        if state.dtype == np.float32 or state.dtype == np.float64:
            if state.max() <= 1.0 and state.min() >= 0.0:
                # Input is normalized [0, 1], scale to [0, 255]
                state = (state * 255.0).astype(np.uint8)
                next_state = (next_state * 255.0).astype(np.uint8)
            else:
                # Already in [0, 255] range but float, just cast
                state = state.astype(np.uint8)
                next_state = next_state.astype(np.uint8)
        else:
            # Already uint8 or int, ensure uint8
            state = state.astype(np.uint8)
            next_state = next_state.astype(np.uint8)
        
        self.states[self.buffer_ptr] = state
        self.next_states[self.buffer_ptr] = next_state
        self.actions[self.buffer_ptr] = action
        self.rewards[self.buffer_ptr] = reward
        self.dones[self.buffer_ptr] = done
        
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        self.buffer_count = min(self.buffer_count + 1, self.buffer_size)
        
        ###########################
        
        
    def sample_batch(self):
        """ Sample batch using uniform experience replay.
        Returns batch data.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if self.buffer_count < self.batch_size:
            return None
        
        # Randomly select indices
        indices = np.random.choice(self.buffer_count, size=self.batch_size, replace=False)
        
        # Get batch from pre-allocated arrays
        batch_states = self.states[indices]  # (batch, 84, 84, 4) uint8
        batch_next_states = self.next_states[indices]  # (batch, 84, 84, 4) uint8
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_dones = self.dones[indices]
        
        # Normalize uint8 to float32 [0, 1] only when sampling (memory efficient!)
        states = torch.FloatTensor(batch_states).permute(0, 3, 1, 2).to(self.device) / 255.0  # (batch, 4, 84, 84)
        next_states = torch.FloatTensor(batch_next_states).permute(0, 3, 1, 2).to(self.device) / 255.0  # (batch, 4, 84, 84)
        
        actions = torch.LongTensor(batch_actions).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        dones = torch.FloatTensor(batch_dones).to(self.device)
        
        return states, actions, rewards, next_states, dones
        
        ###########################
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        print("Starting DQN training...")
        print(f"Device: {self.device}")
        print(f"Number of actions: {self.num_actions}")
        
        episode_rewards = []
        episode_lengths = []
        recent_rewards = deque(maxlen=30)  # For tracking average reward
        
        # Training episodes - train until step budget reached
        # Will train until step budget reached (10,000,000 steps)
        num_episodes = 200000  # Large upper bound, but step budget will stop before this
        max_training_steps = 10_000_000  # Step budget (10M steps)
        
        # Early stopping setup (only for target reward, not plateau)
        early_stop_enabled = self.args.early_stop
        early_stop_window = deque(maxlen=self.args.early_stop_window) if early_stop_enabled else None
        early_stopped = False
        
        # Create progress bar
        pbar = tqdm(range(num_episodes), desc="Training", unit="episode")
        
        # Sanity check flag (only check once after a few episodes)
        sanity_check_done = False
        
        for episode in pbar:
            state = self.env.reset()
            # Debug: Check state immediately after reset
            if episode % 100 == 0 or episode < 5:
                s = np.asarray(state)
                print(f"\n[Episode {episode}] RESET state: type={type(state)}, shape={s.shape}, dtype={s.dtype}, min={s.min()}, max={s.max()}", flush=True)
                sys.stdout.flush()
                if isinstance(state, tuple):
                    print(f"  WARNING: state is a tuple! {state}", flush=True)
                    sys.stdout.flush()
            
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Debug: Check state before make_action (first step only)
                if (episode % 100 == 0 or episode < 5) and episode_length == 0:
                    s = np.asarray(state)
                    print(f"[Episode {episode}, Step 0] Before make_action: shape={s.shape}, dtype={s.dtype}, min={s.min()}, max={s.max()}", flush=True)
                    sys.stdout.flush()
                
                # Select action
                action = self.make_action(state, test=False)
                
                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Debug: Check next_state immediately after step (first step only)
                if (episode % 100 == 0 or episode < 5) and episode_length == 0:
                    ns = np.asarray(next_state)
                    print(f"[Episode {episode}, Step 0] After step - next_state: type={type(next_state)}, shape={ns.shape}, dtype={ns.dtype}, min={ns.min()}, max={ns.max()}", flush=True)
                    sys.stdout.flush()
                    if isinstance(next_state, tuple):
                        print(f"  WARNING: next_state is a tuple! {next_state}", flush=True)
                        sys.stdout.flush()
                    # (b) Check reward values
                    print(f"[Episode {episode}, Step 0] Reward: {reward} (type={type(reward)}, value={reward})", flush=True)
                    sys.stdout.flush()
                    if abs(reward) > 10:
                        print(f"  WARNING: Unexpectedly large reward value: {reward}", flush=True)
                        sys.stdout.flush()
                    if reward == 0 and episode_length == 0:
                        print(f"  NOTE: First step reward is 0 (may be normal)", flush=True)
                        sys.stdout.flush()
                
                # Debug: Check state and next_state before push (first step only)
                if (episode % 100 == 0 or episode < 5) and episode_length == 0:
                    s = np.asarray(state)
                    ns = np.asarray(next_state)
                    print(f"[Episode {episode}, Step 0] Before push - state: shape={s.shape}, dtype={s.dtype}, min={s.min()}, max={s.max()}", flush=True)
                    sys.stdout.flush()
                    print(f"[Episode {episode}, Step 0] Before push - next_state: shape={ns.shape}, dtype={ns.dtype}, min={ns.min()}, max={ns.max()}", flush=True)
                    sys.stdout.flush()
                
                # Store transition in replay buffer
                self.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.step_count += 1
                self._apply_lr_for_step(self.step_count)
                
                # Step-based epsilon decay (not episode-based)
                # Linear decay from epsilon_start to epsilon_end over epsilon_decay_steps
                frac = min(1.0, self.step_count / self.epsilon_decay_steps)
                self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)
                
                # Train if we have enough samples
                if self.step_count >= self.train_start and self.buffer_count >= self.batch_size:
                    # Sanity check: verify replay buffer contents (once after a few episodes)
                    if not sanity_check_done and self.buffer_count >= 100:
                        print("\n" + "="*60)
                        print("REPLAY BUFFER SANITY CHECK")
                        print("="*60)
                        
                        # Check raw buffer state
                        sample = self.states[0]
                        print(f"Buffer state dtype: {sample.dtype}")
                        print(f"Buffer state min: {sample.min()}, max: {sample.max()}")
                        print(f"Buffer state shape: {sample.shape}")
                        print(f"Buffer state sample values (first 5 pixels): {sample[0, 0, :5]}")
                        
                        # Check after sampling and normalization
                        batch = self.sample_batch()
                        if batch is not None:
                            states, actions, rewards, next_states, dones = batch
                            print(f"\nAfter sampling (normalized):")
                            print(f"  Batch states shape: {states.shape}")
                            print(f"  Batch states dtype: {states.dtype}")
                            print(f"  Batch states min: {states.min().item():.6f}, max: {states.max().item():.6f}")
                            print(f"  Batch states mean: {states.mean().item():.6f}")
                            print(f"  Batch states sample (first pixel, first channel): {states[0, 0, 0, 0].item():.6f}")
                            
                            # Check actions, rewards, dones
                            print(f"\nBatch other components:")
                            print(f"  Actions shape: {actions.shape}, dtype: {actions.dtype}")
                            print(f"  Actions sample: {actions[:5].cpu().numpy()}")
                            print(f"  Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
                            print(f"  Rewards min: {rewards.min().item():.2f}, max: {rewards.max().item():.2f}, mean: {rewards.mean().item():.2f}")
                            print(f"  Dones shape: {dones.shape}, dtype: {dones.dtype}")
                            print(f"  Dones sum (terminal states): {dones.sum().item()}")
                            
                            print("\nExpected values:")
                            print("  Buffer: uint8, min ~0, max ~255")
                            print("  Batch states: float32, min ~0.0, max ~1.0 (after /255.0)")
                            print("="*60 + "\n")
                            
                            sanity_check_done = True
                    
                    # Train once per environment step
                    batch = self.sample_batch()
                    if batch is not None:
                        states, actions, rewards, next_states, dones = batch
                        
                        # DDQN: Use main network to select action, target network to evaluate
                        # Compute Q-values for current states
                        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                        
                        # DDQN: Select best action using main network
                        with torch.no_grad():
                            next_q_values_main = self.q_network(next_states)  # Main network
                            next_actions = next_q_values_main.argmax(1)  # Best action from main network
                            
                            # Evaluate selected action using target network
                            next_q_values_target = self.target_network(next_states)
                            next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                            
                            q_targets = rewards + (1 - dones) * self.gamma * next_q_values
                        
                        # MSE loss
                        loss = F.mse_loss(q_values, q_targets)
                        
                        # Track loss
                        self.loss_history.append(loss.item())
                        self.train_step_count += 1
                        
                        # (1) Q-value evolution check: Log Q-value stats every 5k steps
                        if self.step_count % 5000 == 0:
                            with torch.no_grad():
                                # Sample a fresh batch for Q-value inspection
                                q_batch = self.sample_batch()
                                if q_batch is not None:
                                    q_states, _, _, _, _ = q_batch
                                    q_values_all = self.q_network(q_states)  # All Q-values for all actions
                                    q_mean = q_values_all.mean().item()
                                    q_std = q_values_all.std().item()
                                    q_min = q_values_all.min().item()
                                    q_max = q_values_all.max().item()
                                    print(f"\n[DEBUG Q-VALUES] step={self.step_count} | "
                                          f"Q: mean={q_mean:.4f}, std={q_std:.4f}, min={q_min:.4f}, max={q_max:.4f}", flush=True)
                                    sys.stdout.flush()
                        
                        # Optimize
                        self.optimizer.zero_grad()
                        loss.backward()
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
                        self.optimizer.step()
                    
                    # Update target network (only once per environment step)
                    if self.step_count % self.target_update_freq == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
            
            # Track statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            recent_rewards.append(episode_reward)
            
            # (c) Check episode termination - log suspiciously short episodes
            if episode % 100 == 0 or episode < 10:
                avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
                if episode_length < 10:
                    print(f"\n[Episode {episode}] WARNING: Very short episode! Length={episode_length}, Reward={episode_reward:.2f}", flush=True)
                    sys.stdout.flush()
                    print(f"  Average episode length (last 100): {avg_length:.1f}", flush=True)
                    sys.stdout.flush()
                elif episode % 100 == 0:
                    print(f"[Episode {episode}] Episode length: {episode_length}, Avg (last 100): {avg_length:.1f}", flush=True)
                    sys.stdout.flush()
            
            # (b) Check reward statistics periodically
            if episode % 100 == 0 and episode > 0:
                recent_reward_array = np.array(episode_rewards[-100:])
                print(f"[Episode {episode}] Reward stats (last 100): min={recent_reward_array.min():.2f}, max={recent_reward_array.max():.2f}, mean={recent_reward_array.mean():.2f}, std={recent_reward_array.std():.2f}", flush=True)
                sys.stdout.flush()
                non_zero = np.sum(recent_reward_array != 0)
                print(f"  Episodes with non-zero reward: {non_zero}/100", flush=True)
                sys.stdout.flush()
            self.episode_count += 1
            
            # Periodically save episode rewards to disk (for plot generation if interrupted)
            if (episode + 1) % 100 == 0:  # Save every 100 episodes
                rewards_file = self.args.model_path.replace('.pth', '_rewards.npy')
                np.save(rewards_file, np.array(episode_rewards))
            
            # Update progress bar
            avg_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0
            
            # Early stopping logic (only target reward, no plateau detection)
            stop_reason = None
            
            # Check step budget
            if self.step_count >= max_training_steps:
                stop_reason = f"Step budget reached ({self.step_count} >= {max_training_steps})"
                early_stopped = True
            
            # Check target reward (if early stopping enabled)
            elif early_stop_enabled and early_stop_window is not None:
                early_stop_window.append(episode_reward)
                
                # Only check after we have enough episodes in the window
                if len(early_stop_window) >= self.args.early_stop_window:
                    window_avg = np.mean(early_stop_window)
                    
                    # Check if we reached the target
                    if window_avg >= self.args.early_stop_target:
                        stop_reason = f"Target reward reached ({window_avg:.2f} >= {self.args.early_stop_target})"
                        early_stopped = True
            
            # Update progress bar
            postfix_dict = {
                'Reward': f'{episode_reward:.2f}',
                'Avg30': f'{avg_reward:.2f}',
                'Epsilon': f'{self.epsilon:.3f}',
                'Steps': self.step_count,
                'Budget': f'{self.step_count}/{max_training_steps}'
            }
            if early_stop_enabled and early_stop_window and len(early_stop_window) >= self.args.early_stop_window:
                window_avg = np.mean(early_stop_window)
                postfix_dict['AvgWin'] = f'{window_avg:.2f}'
                postfix_dict['Target'] = f'{self.args.early_stop_target}'
            pbar.set_postfix(postfix_dict)
            
            # Print detailed progress every 50 episodes
            if (episode + 1) % 50 == 0 or episode == 0:
                avg_loss = np.mean(self.loss_history) if len(self.loss_history) > 0 else 0.0
                min_loss = np.min(self.loss_history) if len(self.loss_history) > 0 else 0.0
                max_loss = np.max(self.loss_history) if len(self.loss_history) > 0 else 0.0
                std_loss = np.std(self.loss_history) if len(self.loss_history) > 0 else 0.0
                print(f"\nEpisode {episode + 1}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Avg Reward (last 30): {avg_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Steps: {self.step_count} | "
                      f"Train Steps: {self.train_step_count} | "
                      f"Loss: avg={avg_loss:.6f}, min={min_loss:.6f}, max={max_loss:.6f}, std={std_loss:.6f}", flush=True)
                sys.stdout.flush()
                
                # (2) Loss collapse warning: Check if loss has collapsed to near-zero
                if avg_loss < 0.001 and len(self.loss_history) >= 50:
                    print(f"  [WARNING] Loss has collapsed to near-zero (avg={avg_loss:.6f}). "
                          f"This may indicate TD-errors are ~0 (targets ≈ predictions).", flush=True)
                    sys.stdout.flush()
                elif avg_loss > 10.0:
                    print(f"  [WARNING] Loss is very high (avg={avg_loss:.6f}). "
                          f"Network may be unstable.", flush=True)
                    sys.stdout.flush()
            
            # Save model periodically
            if (episode + 1) % self.args.save_freq == 0:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.args.model_path) if os.path.dirname(self.args.model_path) else '.', exist_ok=True)
                torch.save(self.q_network.state_dict(), self.args.model_path)
                
                # Get file modification time after saving
                from datetime import datetime
                mod_time = os.path.getmtime(self.args.model_path)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"Model saved at episode {episode + 1} to {self.args.model_path}")
                print(f"  Checkpoint last modified: {mod_time_str} (timestamp: {mod_time})")
            
            # Early stopping check
            if early_stopped:
                print(f"\nEarly stopping triggered: {stop_reason}")
                break
        
        # Final save
        pbar.close()
        os.makedirs(os.path.dirname(self.args.model_path) if os.path.dirname(self.args.model_path) else '.', exist_ok=True)
        torch.save(self.q_network.state_dict(), self.args.model_path)
        
        # Get file modification time after saving
        from datetime import datetime
        mod_time = os.path.getmtime(self.args.model_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        final_msg = "Early stopped" if early_stopped else "Training completed"
        print(f"\n{final_msg}! Final model saved to {self.args.model_path}")
        print(f"  Checkpoint last modified: {mod_time_str} (timestamp: {mod_time})")
        print(f"Final average reward (last 30 episodes): {np.mean(recent_rewards):.2f}")
        if early_stop_enabled and early_stop_window and len(early_stop_window) >= self.args.early_stop_window:
            print(f"Final average reward (early stop window): {np.mean(early_stop_window):.2f}")
        
        # Save final episode rewards to disk
        rewards_file = self.args.model_path.replace('.pth', '_rewards.npy')
        np.save(rewards_file, np.array(episode_rewards))
        
        # Generate learning curve plot
        if len(episode_rewards) > 0:
            # Compute rolling 30-episode average
            rolling_avg = []
            for i in range(len(episode_rewards)):
                start_idx = max(0, i - 29)  # 30 episodes: current + 29 previous
                rolling_avg.append(np.mean(episode_rewards[start_idx:i+1]))
            
            # Create plot
            plt.figure(figsize=(10, 6))
            episodes = np.arange(1, len(episode_rewards) + 1)
            plt.plot(episodes, rolling_avg, linewidth=1.5, label='Average reward (last 30 episodes)')
            plt.xlabel('Number of Episodes', fontsize=12)
            plt.ylabel('Average Reward (last 30 episodes)', fontsize=12)
            plt.title('DQN Learning Curve', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = self.args.model_path.replace('.pth', '_learning_curve.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Learning curve saved to {plot_path}")
            print(f"Episode rewards saved to {rewards_file} (can be used to regenerate plot)")
        
        ###########################
