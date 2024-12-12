import torch
import torch.nn as nn



class MLP(nn.Module):
    def __init__(self, layer_sizes, init_method=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 2):
            if init_method is not None:
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]).apply(init_method))
            else:
                self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.ELU())  # ELU activation after each linear layer

        if init_method is not None:
            self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]).apply(init_method))
        else:
            self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))            

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class RND():
    def __init__(self, layer_sizes, device, optimzer_lr=1e-4, rnd_gamma=0.99):
        self.device = device
        self.target_network = MLP(layer_sizes, init_method=self.init_method_2).to(self.device)
        self.predictor_network = MLP(layer_sizes, init_method=self.init_method_1).to(self.device)     
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=optimzer_lr)

        # self.running_mean = 0
        # self.running_std = 1
        self.obs_running_mean = 0
        self.obs_running_std = 1
        self.gamma = rnd_gamma

    # @staticmethod
    def init_method_1(self, model):
        model.weight.data.uniform_()
        model.bias.data.uniform_()

    # @staticmethod
    def init_method_2(self, model):
        model.weight.data.normal_()
        model.bias.data.normal_()

    def compute_reward(self, obs):
        # with torch.no_grad(): 
        # target_output = self.target_network(obs)
        # predictor_output = self.predictor_network(obs)

        self.obs_running_mean = self.gamma* self.obs_running_mean + (1-self.gamma)*torch.mean(obs, dim=0)
        self.obs_running_std = self.gamma* self.obs_running_std + (1-self.gamma)*torch.std(obs, dim=0)
        obs = torch.div(obs - self.obs_running_mean, self.obs_running_std)
        obs = obs.clip(-5, 5)

        target_output = self.target_network(obs)
        predictor_output = self.predictor_network(obs)

        intrinsic_reward = torch.mean((target_output - predictor_output) ** 2, dim=1)
        # self.running_mean = self.gamma* self.running_mean + (1-self.gamma)*torch.mean(intrinsic_reward)
        # self.running_std = self.gamma* self.running_std + (1-self.gamma)*torch.std(intrinsic_reward)
        # intrinsic_reward = (intrinsic_reward - self.running_mean)/self.running_std
        return intrinsic_reward
    
    def update(self, obs):
        # torch.set_grad_enabled(True)
        with torch.enable_grad():
            obs = torch.div(obs - self.obs_running_mean, self.obs_running_std)
            obs = obs.clip(-5, 5)
            target_output = self.target_network(obs).detach()
            predictor_output = self.predictor_network(obs)
            loss = self.loss_fn(target_output, predictor_output)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




if __name__=='__main__':
    
    device = 'cuda'
    rnd = RND([2,5,2], device)
    obs1 = torch.tensor([5.,5.], device=device).reshape(1,2)
    obs2 = torch.tensor([50.,50.], device=device).reshape(1,2)
    print(rnd.compute_reward(obs1))
    print(rnd.compute_reward(obs2))
    print('==')
    for _ in range(10000):
        rnd.update(obs1)
    print(rnd.compute_reward(obs1))
    print(rnd.compute_reward(obs2))

