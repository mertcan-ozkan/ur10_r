import numpy as np
from environment import Env
import matplotlib.pyplot as plt
import torch
from collections import deque
import random 
import time
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_LENGTH = 10000
EPSILON = 1.0
MIN_EPSILON = 0.1 
EPSILON_DECAY = 0.001 
EPSILON_DECAY_ITER = 100 
LEARNING_RATE = 0.0005
UPDATE_FREQ = 4 
TARGET_NETWORK_UPDATE_FREQ = 1000 
N_ACTIONS = 4

env = Env()

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_features = int(4)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,N_ACTIONS)   
        )
    def forward(self,x):
        return self.net(x)
 
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype= torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index= torch.argmax(q_values,dim=1)[0]
        action = max_q_index.detach().item()
        return action
    
rew_buffer =  deque([0,0],maxlen = 200)
replay_buffer = deque(maxlen= BUFFER_LENGTH)
plot_buffer = deque([0,0])
device = 'cpu'
Online_Net = Network()
Target_Net = Network()
Online_Net = torch.load("e_5000_decay100_online_net.pth")
Online_Net.eval()
Target_Net = torch.load("e_5000_decay100_target_net.pth")
Target_Net.eval()
Online_Net.to(device)
Target_Net.to(device)
Target_Net.load_state_dict(Online_Net.state_dict())
optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=Online_Net.parameters())

not_first= False
while (True) :  #### while rospy not down
    episode_reward = 0.0  
    if not_first: 
        print("reseting .......")
        env.reset()
    not_first  = True
    done = False
    while not done: 
        old_state= env.high_level_state()
        action = Online_Net.act(torch.as_tensor(old_state).to(device) ) 
        state, reward, is_terminal, is_truncated = env.step(action)
        
        new_state= env.high_level_state()
        done = is_terminal or is_truncated 
        transition = (old_state, action,reward, done, new_state)
        replay_buffer.append(transition)
        episode_reward  = episode_reward + reward
        if done:
            rew_buffer.append(episode_reward)
            plot_buffer.append(np.mean(rew_buffer) )




# plt.plot(plot_buffer)
# plt.title("Running Avarage Of Rewards")
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.savefig('dqn_pybullet_decay100.png')
# plt.show()









### simulation task space limits (dqn)(mujoco):   x = [0.25,0.75] ,  y = [-0.3, 0.3]
## x 0.50
## y 0.6
## initial pos: [0.79798992 0.17399998]
## moves 
## action :     [0.05 0.  ]
## action :     [-0.025       0.04330127]
## action :     [-0.025      -0.04330127]
## action :     [ 5.0000000e-02 -1.2246468e-17]
###
# (-1.558, 0.125)
# (-1.6585, 0.19356034416666668)   -- dondu

#TASKSPACE_LIMITS = {'x':[-1.29, -0.62],'y':[-0.35, 0.6],'z':[0.24, 0.7]} # very very safe ;)
## x 0.67
## y 0.95
## initial pos: [-0.90021825  0.02926625]
## move if v == 1 :    0.03

#### her pozisyondan initial position a gelmeli episode bitince !! - bunun için demonstartiondaki playı kullanabılrısın
#### problemler:
### simulasyonda action space ı bolme olayını burda nasıl yaparız
### enumarte 
###  bunu kullanacaksan tam w_x ve w_y nin oranları neden esıt degıl bakman lazım
### burdakı task space sımulasyondakıyle aynı degıl ona bakman lazım - burdakı task space borderları neye gore belırlenmıs?
## burda v lere gore bır target position hesaplanıyor - direkt position nasıl verılır ona bakman lazım
### modele x y için goal ve gripper position vermelisin ona bak!
    ## right joy --> pendulum 
    ## left joy cartesian
    ## v1 pozitif sol
    ## v0 pozitif ileri
        #     print('v[0] :   ' , v[0] )
        # print('v[1] :   ' , v[1])
        # print('v[2]:   ' , v[2])
        # print('w_x:   ' , w_x)
        # print('w_y:   ' , w_y)
        # print('w_Z:   ' , w_Z)
        ###############################################################
# sol sag yukarı asagı ascent descent


# v[0] :    -0.0
# v[1] :    1.0

# p_target:   [-0.90019894  0.05812449  0.5196807 ]
# p_initial:  [-0.90019894  0.02925727  0.51968068]

# idx:  0
# axis:  x
# idx:  1
# axis:  y
# idx:  2
# axis:  z
# v[0] :    -0.0
# v[1] :    -1.0
# v[2]:    0.0
# w_x:    0.0
# w_y:    0.0
# w_Z:    0.0

# p_target:   [-0.9001957   0.02927696  0.519691  ]
# p_initial:  [-0.90019572  0.05814418  0.51969099]

# idx:  0
# axis:  x
# idx:  1
# axis:  y
# idx:  2
# axis:  z
# v[0] :    1.0
# v[1] :    -0.0

 
## time mevzusu için gidene kadar while koy +
## tajı manuel sılme  +
## init poziyonda değilse init e çek ototmaitk +
## actionlar task space de cok az yer degıstırtıyor - bunu nedenıne bak  +
## action -task space oranı aynı olmalı! +







