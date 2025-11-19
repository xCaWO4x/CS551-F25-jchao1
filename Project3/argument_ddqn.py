def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    # DDQN Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='learning rate for training')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='starting epsilon for epsilon-greedy')
    parser.add_argument('--epsilon_end', type=float, default=0.1, help='ending epsilon for epsilon-greedy (step-based decay)')
    parser.add_argument('--epsilon_decay_steps', type=int, default=1000000, help='number of steps to decay epsilon from start to end')
    parser.add_argument('--replay_buffer_size', type=int, default=1000000, help='size of replay buffer')
    parser.add_argument('--target_update_freq', type=int, default=10000, help='frequency to update target network')
    parser.add_argument('--train_start', type=int, default=50000, help='number of steps before training starts')
    parser.add_argument('--save_freq', type=int, default=100, help='frequency to save model (episodes)')
    parser.add_argument('--model_path', type=str, default='./dqn_model.pth', help='path to save/load model')
    parser.add_argument('--resume_from_model', action='store_true', help='continue training from existing model checkpoint if available')
    parser.add_argument('--resume_steps', type=int, default=0, help='number of environment steps already completed (for LR schedule/budget)')
    
    # Early stopping parameters (disabled by default)
    parser.add_argument('--early_stop', action='store_true', help='enable early stopping (disabled by default)')
    parser.add_argument('--early_stop_patience', type=int, default=200, help='number of episodes without improvement before stopping')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.1, help='minimum improvement in average reward to reset patience')
    parser.add_argument('--early_stop_window', type=int, default=100, help='window size for computing average reward for early stopping')
    parser.add_argument('--early_stop_target', type=float, default=50.0, help='target average reward - if reached, can stop early')
    
    return parser

