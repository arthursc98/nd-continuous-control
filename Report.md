[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"
[image3]: https://pylessons.com/static/images/Reinforcement-learning/09_A2C-reinforcement-learning/Actor-Critic-fun.png "Actor-Critic"
[image4]: https://pylessons.com/static/images/Reinforcement-learning/08_PG-reinforcement-learning/RL_Agents.png "Policy-Based Methods"
[image5]: https://paperswithcode.com/media/methods/b6cdb8f5-ea3a-4cca-9331-f951c984d63a_MBK7MUl.png "SARS Memory"
[image6]: https://miro.medium.com/max/700/1*vLFINWklJ0BtYtgzwK223g.png "Gradient Clipping"
[image7]: imgs/model_performance.png "Model Performance"
[image8]: imgs/iteration_metrics.png "Iteration Metrics"
[image9]: https://pylessons.com/static/images/Reinforcement-learning/09_A2C-reinforcement-learning/Actor-Critic-fun.png "Actor Critic"
[image10]: https://cdn-media-1.freecodecamp.org/images/1*SvSFYWx5-u5zf38baqBgyQ.png "Advantage Function"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Objective
The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Solution
Before we dive deep in my solution i'll cover the following agenda an try to cover all concepts used in the implementation.
- Policy-Based Methods
- Actor-Critic
- Deep Deterministic Policy Gradient
- Epsilon Greedy
- Experience Replay
- Ornstein-Uhlenbeck Process
- Gradient Clipping
- Batch Normalization

## Policy-Based Methods
In RL we have two methods that we want to optimize, we have value-based methods and policy-based methods but after all, what's the difference between those two methods? One exists but the other one don't? Well for our lucky there's some models that join the both of two worlds, but let's explain a little more about them, basically a value based method learn by acting with the best action in the state, so if we have a probability for each action we would choose the one that has the maximum probability, in the other hand we have a policy based method which learns by a policy function that maps each state to action, so if we are learning each state to action, let's say a input action 1 gives a state 1 and action 2 gives a state 2 and so on, this type of method will learn kinda like a state machine e.g. MDP (Markov Decision Process)<br>
![Policy-Based Methods][image4]<br>

## Actor-Critic
Have you ever heard about GAN's? Well they basically are the fake images that you see on the internet, one cool example is [this cat does not exist](https://thiscatdoesnotexist.com/) which shows up cat that are made by a GAN. GAN's extends for Generative Adversarial Network, by a given training set they try to generate something similar to their training set. This class of ML was developed by Ian Goodfellow and his colleagues in 2014 and throw years pass there some improvements in this field. The architecture of GAN's it's really interesting and it's replicated to our Actor-Critic model, in GAN's we have two models that work together, one it's the Generator and the other one it's the Discriminator, when the Generator creates new examples the Discriminator judge if it was a good representation based on the training set.<br>
In Actor-Critic we have a similar architecture, we have our Actor that will play around the enviroment and our Critic which will tell to our Actor if it was a good move to take in the enviroment, which it's pretty cool right?<br>
![Actor Critic][image9]<br>
Now we have a function called Advantage Function, which shows us how much better it is to take a specific action compared to the average, general action at the given state, and that what's the equation below tell us, if you wanna go deeper i highly recommend to check on some resources like [this](https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b) one that goes deep in math concepts with some pseudocode<br>
![Advantage Function][image10]

## Deep Deterministic Policy Gradient (DPPG)
DDPG is a RL technique that combines Q-Learning and Policy Gradients. While the actor is a policy network that takes the state as input and outputs the exact action instead of probability for each action to choose a strategy to pick one of them. The critic is a Q-value network that takes in state and action as input and outputs the Q-value.

## Epsilon Greedy
Before we talk about epsilon greedy we need to know a very popular dilemma called Exploration / Exploitation, this dilemma is one of the hardest to think about, let's say that each time we play we have to decide between explore even more our environment and see which series of actions would lead to a highest rewards or keep what we know about the environment and continue to do the action that belongs the highest rewards, now, what should we do? Start exploring our environment and more often begins to exploit it? It's there any possible way to estimate when we need to explore the environment? Or even a heuristic? Well you will see about Epsilon Greedy in the next section which tries to solve this problem.<br>
Now that you know more about exploration / exploitation dilemma we can explain how Epsilon Greedy works, let's say we have a probability for those two actions, what epsilon greedy tries to do it's to generate a randomness into the algorithm, which force the agent to try different actions and not get stuck at a local minimum, so to implemente epsilon greedy we set a epsilon value between 0 and 1 where 0 we never explore but always exploit the knowledge that we already have and 1 do the opposite, after we set a value for epsilon we generate a random value usually from a normal distribution and if that value is bigger than epsilon we will choose the current best action otherwise we will choose a random action to explore the environment.<br>
For this current project i used the following parameter to control the exploitation and generate a little noise to force the some exploration.<br>
```python
EPSILON = 1.0           # Epsilon Exploration / Exploit Parameter
EPSILON_DECAY = 1e-6    # Decay Epsilon while training
```

## Experience Replay
In order to try to solve rare events detection in our model, we store each experience from our agent in a memory and sample it randomly so our agent start to generalize better and recall rare occurrences. Also for better performance we could use mini-batch's to see how our model converge. The image below shows up how figuratively the memory would look.<br>
![SARS Memory][image5]

## Ornstein-Uhlenbeck Process
The Ornstein-Uhlenbeck process it's a stochastic process that envolves a modification random walk in continuous time, this process tends to converge to the mean function, which it's called mean-reverting, at first sight i never saw this before but it's really good since it's applied into continuous problems as like the one we are trying to solve<br>
In OU Process we have three highly important parameters which are they the $\mu$ that represents the mean where the noise will be generated will tend to drift towards, $\theta$ it's the speed that the noise will reach the mean and $\sigma$ it's volatility parameter, this came from physics concepts but it's really used in financial problems, one of current posts that i got based on it's [this one](https://www.maplesoft.com/support/help/maple/view.aspx?path=Finance%2FOrnsteinUhlenbeckProcess#bkmrk0).<br<>
I tried to replicate the parameters used in [this paper](https://arxiv.org/pdf/1509.02971.pdf) so i didn't tried too many variation from those parameters but should be an awesome action to take in the future, but for now i used the following parameters to our Ornstein-Uhlenbeck generator.<br>
```python
OU_MU = 0.0             # Ornstein-Uhlenbeck Mu Parameter
OU_THETA = 0.15         # Ornstein-Uhlenbeck Theta Parameter
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck Sigma Parameter
```

## Gradient Clipping
Gradient Clipping tries to solve one of biggest problems in Backpropagation that it's exploding/vanishing gradients which happens really often in RNN's for example, the idea behind it's pretty simple, if the gradient gets too large or too small, we can rescale to be between two values e.g. gradient represents a variable $g$ and clip represents a variable called $c$ should obey the following formula: $|g| >= c$<br>
![Gradient Clipping][image6]<br>
To implement it on our code it's pretty simple the following code it's from the implementation that i did on the project but this should give an idea.
```python
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
```

## Batch Normalization
This technique it's used really often to reduce overfitting into Neural Networks, for this purpose we could use Dropout Layers that would try to reduce it. But to try to keep it simple a Batch Norm it's basically a standard normalization for each d-dimension input, e.g. $x = (x^{(1)}, ..., x^{(d)})$ so if we remember the old formula from standard norm that is `(x - µ) / sqrt(σ**2)` we would have the same normalization but for every mini-batch and dimensions where each one respects the following formula for Batch Normalization:<br>
$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu^{(k)}}{\sqrt{\sigma^{(k)^{2}}}}$$
To implement this equation we can use a function from torch called `torch.nn.BatchNorm1d` which lead us to only specity the size of the input to this layer and use it on our forward function like the code used in actor model, if you wanna deep dive into it just access the `agent.py` file, since it's a short file should be easy to find the code that i mentioned.
```python
...
class Actor(nn.Module, BaseModel):
	def __init__(self, state_size, action_size, seed, fc_units, fc2_units):
		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.bn = nn.BatchNorm1d(fc_units)
		...
	
    def forward(self, state):
        x = F.relu(self.bn(self.fc1(state)))
		...
...

```

## Model Performance
Following all hyperparameters told before i builded the model and start to train it, for starting point i utilize Batch Normalization after each fully connected layer but it started takes too long to compute which for the time left to achieve the goal performance from the project would take a few hours computing it, so to deal with training time i used it after only the first fully connected layer that will try normalize and reduce overfitting in the first layer and voilà it takes much less time to compute and complete what the project challenge, so down below we can see the average and all performance metrics for our multiple agents<br>
![Iteration Metrics][image8]<br>
![Model Performance][image7]<br>
As far we can see our model metrics starts to envolving when we train the agents for longer and longer episodes which it's great right? But what really intrigates me it's the maximum values starts to stabilizing and we may need to deal with this further but let's leave it for new ideas to come shall we?

## Ideas for Future Work
Since a common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, I would probably study about Twin Delayed DDPG and try to implement a Clipped Double-Q Learning, “Delayed” Policy Updates and Target Policy Smoothing, those are the ones that i already saved to try in a near future

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **Twenty (20) Agents**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `nd-continuous-control` folder, and unzip (or decompress) the file and adapt the path in UnityEnviroment refering to the filename, just being generic to deal with OSX and other systems, me personally runned it on Udacity Enviroment but it's completely easy to do it on you're local machine.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
	
2. Since i already created a file with several dependencies that we need to run the project. First of all it's better to install the python folder which contains some stable libraries. To do so follow the next command lines.
```bash
cd python
pip install .
```

3. Now that the env already have some things that we need, let's install the other part of the dependencies
```bash
cd ../
pip install -r requirements.txt
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

6. Now that we have it all to run the project all you gotta do is to open the `Continuous_Control.ipynb` and run each cell, it's highly recommended to run it with GPU for faster results.