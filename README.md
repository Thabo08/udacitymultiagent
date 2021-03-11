[env]: tennis.png "env"
[point5]: tennis_point5.png "point5"
[point75]: tennis_point75.png "point75"
[point5_gif]: tennis_point5.gif "point5_gif"
[point75_gif]: tennis_point75.gif "point75_gif"

# Deep Deterministic Policy Gradient: Collaboration and Competition

In this project, two agents are taught how to play tennis, with the goal being to keep the tennis ball in play. By controlling
racquets, the agents hit the ball over net and must avoid the ball hitting the ground or hitting the ball out of bounds.

The algorithm used is the Deep Deterministic Policy Gradient (DDPG), which uses the Actor-Critic approach:
1. **Actor Model**: A policy function based model used to chose actions to take given a state in the environment,
1. **Critic Model**: A value function based model used to critic the chosen actions

### 1. The Environment
The Unity Tennis environment was used. The environment has two agents, as mentioned. With the goal of keeping the ball in
play, the agents each receive a score of +0.1 when they hit the ball over the net and, a score of when the ball hits the
ground or is hit out of bounds. 

For the agents to be considered to have been well-trained, they must receive an average score of at least +0.5 over 100
consecutive episodes.

Each agent receives its own state (observation) space, where the dimension of that space is 8, comprising position and velocity
of the ball and the racquet. The action space has two variables, representing the forward/backward move of the agent with
respect to the net, and jumping.

Here is an example of the environment:

![env][env]

### 2. The Model
As mentioned, the agent uses the Actor-Critic approach method of learning, where 2 models are used, one as an actor and 
the other as the actor's critic. The Actor model takes the current state as input and outputs a suitable action. The 
critic takes both a state, and the action returned by the actor, as input and returns Q values (hence value based).

The architectures of these models are defined in the _models.py_ file. The actor model has 2 hidden layers and outputs 
and using the *tanh* activation function outputs a value between -1 and 1, representing an action. The critic model also
has 2 hidden layers and uses a *relu* function to map states and action to Q values.

### 3. The Agent
The multi agent set up can be found in _agents.py_. Herein, the MultiDDPGAgent object exposes the functionality required
to do the training. The MultiDDPGAgent has two agents created and, it initialises them for training. This allows agents
to learn through self-play.

The multi agent has the following high level functionalities, which enable it to learn from interactions:
1. **Noise reset**: At the beginning of every episode, the agent resets the noise to start each episode's noise factor on a clean slate
1. **Act**: The agent uses this to take an action based the input state
1. **Step and Learn**: The agent uses this to 'step' into the environment and then learn from experiences sampled from the replay buffer

The two agents share a learned actor network, and a replay buffer. Each agent uses the same actor network to take actions
(self-play) and, contribute their experiences to the same replay buffer, so the experiences can be shared. The agents also
share an exploration noise object used to added exploration to the actions.

Each agent creates and maintains its own target actor network and, both local and target critic networks. These are updated
in the learning step of the agent.

### 4. The Trainer
The code to train the agent can be found in _multi_trainer.py_ file. This is where a Unity environment is loaded, which then 
provides the environment to use with training the agent.

Herein, an agent is trained over the desired number of episodes, with the target score also optionally specified. Additionally,
there's logic to test the trained agent.

### 5. Usage
The packages required to run the _multi_trainer.py_ script can be installed this way:
```
pip install -r requirements.txt
```
The links to download the environment for popular operating systems per version of the environment are as follows:

* Linux: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows 32-bit: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows 64-bit: [Click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Once downloaded, place the zip archive in the directory of your choice and decompress the file.

The script is command line based and can be used in the following ways:
```commandline
python multi_multi_trainer.py -h
```
The line above gives a help print out of the high level parameters that can be passed to run the trainer
```
python multi_trainer.py <agentFile> --mode <mode> --model <saved_model>
```
The line above shows how to run the agent, where ```<agentFile>``` is used to specify which environment file to load,
```<mode>``` indicates which mode to run the trainer in, with options being trained or tested, lastly, ```<saved_model>```
specifies where the trained model weights must be saved when training and where they must be loaded from when testing.

See below as examples of running the script test mode:
```commandline
python multi_trainer.py Tennis.app --mode test --model checkpoint.pth
```

### 6. Results
The multi agent was trained twice, with the aim of reaching the benchmark score of 0.5, and then with the aim of reaching
a higher score of 0.75. The results are shown below:

Target Score  | Episodes to target | Average Score | Training Time | Test Score |
------------- | ------------------ | ------------- | ------------- | ---------- |
+0.5 | 853 | 0.5 | ~ 45 min | 0.4
+0.75 | 1034 | 0.75 | ~ 1 hr | 1.9

It took longer to train and reach a score of 0.75 than it was for 0.5. The following plots show how the multi agent got
scores as episodes proceeded, starting with the 0.5 version and followed by the 0.75 plot.

![point5][point5]

![point75][point75]

Expectedly, when the target score to reach was higher, the agent got higher scores as well in the training. In fact, as
shown in the demonstrations below, the game goes for a longer period in test mode when using model weights of the agent 
trained with a higher goal target without either agent losing.

The agents' behaviours can be seen below, starting with running with model weights when the target was +0.5, followed by
when it was +0.75

![point5_gif][point5_gif]

![point75_gif][point75_gif]

