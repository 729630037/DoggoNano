import matplotlib.pyplot as plt
import os,sys,inspect
import reverb
import time
import envs.tools.bullet_client

# sys.path.append("/home/sq/minitaur-nano")

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
tempdir = currentdir


import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import suite_pybullet
from tf_agents.experimental.train import actor
from tf_agents.experimental.train import learner
from tf_agents.experimental.train import triggers
from tf_agents.experimental.train.utils import spec_utils
from tf_agents.experimental.train.utils import strategy_utils
from tf_agents.experimental.train.utils import train_utils
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import common

env_name = "MinitaurTrottingEnv-v1" # @param {type:"string"}
# env_name = "MinitaurReactiveEnv-v1" # @param {type:"string"}

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 1000000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256   # @param {type:"integer"}

critic_learning_rate = 1e-6 # @param {type:"number"}
actor_learning_rate = 1e-6 # @param {type:"number"}
alpha_learning_rate = 3e-6 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"} Factor for soft update of the target networks
target_update_period = 1 # @param {type:"number"}  Period for soft update of the target networks.
gamma = 0.9 # @param {type:"number"}  A discount factor for future rewards.
reward_scale_factor = 1.0 # @param {type:"number"}  Multiplicative scale for the reward.

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 5 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}

# 加载Minituar环境
env = suite_pybullet.load(env_name)
env.reset()
print('Observation Spec:')
print(env.time_step_spec().observation)
print('Action Spec:')
print(env.action_spec())

# 我们创建两种环境：一种用于在训练期间收集数据，另一种用于评估
collect_env = suite_pybullet.load(env_name,max_episode_steps=2500)
eval_env = suite_pybullet.load(env_name,max_episode_steps=2500)


# 启用GPU
use_gpu = True #@param {type:"boolean"}
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)


# SAC 是actor-critic agent, 需要两个网络。critic为我们提供Q(s,a)的值估计，也就是说，它将收到输入和观察到的动作，并且会给我们估计该动作对于给定状态的效果。
observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(collect_env))

with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),   #  input_tensor_spec: A tuple of (observation, action) each a nest of `tensor_spec.TensorSpec` representing the inputs.
        observation_fc_layer_params=None,  #  Optional list of fully connected parameters for observations, where each item is the number of units in the layer.
        action_fc_layer_params=None,       #  Optional list of fully connected parameters for actions, where each item is the number of units in the layer.
        joint_fc_layer_params=critic_joint_fc_layer_params,     # Optional list of fully connected parameters after merging observations and actions
        kernel_initializer='glorot_uniform',    #  kernel initializer for all layers except for the value regression layer. If None, a VarianceScaling initializer will be used.
        last_kernel_initializer='glorot_uniform')  # kernel initializer for the value regression layer. If None, a RandomUniform initializer will be used.

# 我们将使用critic来训练一个actor网络，这将使我们能够在观察到的情况下产生动作。
# ActorNetwork会预测经过tanh压缩的MultivariateNormalDiag分布的参数。每当我们需要采取行动时，便会根据当前观察结果对这种分布进行采样。
with strategy.scope():
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,        #     input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the input.
      action_spec,             #     output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing the output. 
      fc_layer_params=actor_fc_layer_params,         #   Optional list of fully_connected parameters, where each item is the number of units in the layer.
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))  # Callable that generates a continuous projectionnetwork
                                                                        # to be called with some hidden state and the outer_rank of the state.


# 有了这些网络，我们现在可以实例化agent。
with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = sac_agent.SacAgent(
        time_step_spec,           # A `TimeStep` spec of the expected time_steps.
        action_spec,              # A nest of BoundedTensorSpec representing the actions.
        actor_network=actor_net,  # A function actor_network(observation, action_spec) that returns action distribution.
        critic_network=critic_net, # A function critic_network((observations, actions)) that returns the q_values for each observation and action.
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),     # The optimizer to use for the actor network.
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),    # The default optimizer to use for the critic network.
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),   # The default optimizer to use for the alpha variable.
        target_update_tau=target_update_tau,      # Factor for soft update of the target networks.
        target_update_period=target_update_period,  #  Period for soft update of the target networks.
        td_errors_loss_fn=tf.math.squared_difference,   # A function for computing the elementwise TD errors loss.
        gamma=gamma,         #  A discount factor for future rewards.
        reward_scale_factor=reward_scale_factor,   #  Multiplicative scale for the reward.
        train_step_counter=train_step)      # An optional counter to increment every time the train op is run.  Defaults to the global_step.

  tf_agent.initialize()

# 为了跟踪从环境中收集的数据，我们将使用Deepmind的Reverb ，这是一个高效，可扩展且易于使用的经验回放系统。它存储了actor收集的经验数据，并在训练期间由学习者使用。
# 在本教程中，这不如max_size重要，但是在具有异步收集和训练的分布式设置中，您可能需要使用rate_limiters.SampleToInsertRatio进行实验，在2至1000之间使用samples_per_insert。
# rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0))
table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

# 经验回放缓冲区是使用描述要存储的张量的规范构造的，可以使用tf_agent.collect_data_spec从agent获取该tf_agent.collect_data_spec 。
# 由于SAC agent需要当前和下一个观测值来计算损失，因此我们设置sequence_length=2 。
reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

# 现在我们从Reverb 经验回放缓冲区生成TensorFlow数据集。我们会将其传递给学习者，以抽样经验进行训练。
dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

# 在TF-Agent中，策略代表RL中策略的标准概念：给定time_step产生action或action分布。
# 主要方法是policy_step = policy.step(time_step) ，其中policy_step是一个命名的元组PolicyStep(action, state, info) 。
# policy_step.action是要应用于环境的action ， state代表有状态（RNN）策略的状态， info可能包含辅助信息，例如动作的日志概率。
# Agents包含两个策略：
# agent.policy用于评估和部署的主要策略。
# agent.collect_policy用于数据收集的第二个策略。
tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)

# 可以独立于agent创建策略。例如，使用tf_agents.policies.random_py_policy创建一个策略，该策略将为每个time_step随机选择一个动作。
random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())

# actor管理策略和环境之间的交互。
# actor组件包含环境的实例（如py_environment ）和策略变量的副本。
# 给定策略变量的本地值，每个Actor都会执行一系列数据收集步骤。
# 在调用actor.run()之前，使用训练脚本中的变量容器客户端实例显式完成变量更新。
# 在每个数据收集步骤中，将观察到的体验写入重放缓冲区。
# 当Actor运行数据收集步骤时，他们将（状态，动作，奖励）的轨迹传递给观察者，观察者将其缓存并将其写入经验回放缓冲区。
# 我们正在存储帧[[t0，t1）（t1，t2）（t2，t3），...]的stride_length=1 ，因为stride_length=1 。
rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

# 我们使用随机策略创建一个Actor，并收集经验以为经验回放缓冲区提供种子。
initial_collect_actor = actor.Actor(
  collect_env,    # An instance of either a tf or py environment. Note the policy, andobservers should match the tf/pyness of the env.
  random_policy,  # An instance of a policy used to interact with the environment.
  train_step,     # A scalar tf.int64 `tf.Variable` which will keep track of the number of train steps. This is used for artifacts created like summaries.
  steps_per_run=initial_collect_steps, # Number of steps to evaluated per run call. See below.
  observers=[rb_observer])
initial_collect_actor.run()  

# 使用收集策略实例化actor，以训练期间收集更多经验。
env_step_metric = py_metrics.EnvironmentSteps()  # Counts the number of steps taken in the environment
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),    # A list of metric observers.
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

# 创建一个Actor，该Actor将在训练期间用于评估策略。我们传入actor.eval_metrics(num_eval_episodes)以稍后记录指标。
eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

# Learner组件包含代agent，并使用经验回放缓冲区中的经验数据对策略变量执行渐变步骤更新。经过一个或多个训练步骤后，学习者可以将一组新的变量值推送到变量容器。
saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# 策略保护
tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]
agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers)

# # 检查点
# checkpoint_dir = os.path.join(tempdir, 'checkpoint')
# train_checkpointer = common.Checkpointer(
#     ckpt_dir=checkpoint_dir,
#     max_to_keep=1,
#     agent=tf_agent,
#     policy=tf_agent.policy,
#     replay_buffer=reverb_replay,
#     global_step=train_step
# )
# train_checkpointer.initialize_or_restore()


# 我们使用上面的actor.eval_metrics实例化了eval Actor，它在策略评估期间创建了最常用的指标：
# 平均回报。回报是在环境中为某个episode运行策略时获得的报酬之和，我们通常将其平均化为几个情节。
# 平均episode长度。
def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results
metrics = get_eval_metrics()

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

# 训练循环涉及从环境中收集数据和优化agent网络。在此过程中，我们有时会评估agent的策略以了解我们的情况。
# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]
init_time=time.time()
for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1} time = {2}'.format(step, loss_info.loss.numpy(),time.time()-init_time))

  # if step % policy_save_interval == 0:
  #   tf_policy_saver.save('policy_%d' %_)
print("Train time is {0}".format((time.time()-init_time)/(60.0*60)))

rb_observer.close()
reverb_server.stop()

# 我们可以绘制平均回报率与总体步骤的关系图，以查看agent的表现。在Minitaur，奖励功能基于Minitaur行走1000步并惩罚能源消耗的距离。
steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.show()


