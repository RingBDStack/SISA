import torch

class InverseModel(torch.nn.Module):
    """
    Description:
        Network module that captures predicting the action given a state, next_state pair.

    Parameters:
        - params
        - n_actions : Int
            The number of actions in the environment
    """
    def __init__(self, params, n_actions, discrete=False):
        super(InverseModel, self).__init__()
        self.discrete = discrete
        self.body = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], params['layer_size']),
            torch.nn.ReLU(),
        )
        if self.discrete:
            self.log_pr_linear = torch.nn.Linear(params['layer_size'], n_actions)
        else:
            self.mean_linear = torch.nn.Linear(params['layer_size'], n_actions)
            self.log_std_linear = torch.nn.Linear(params['layer_size'], n_actions)
        self.log_sig_min = params['log_std_min']
        self.log_sig_max = params['log_std_max']

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        shared_vector = self.body(context)

        if self.discrete:
            return self.log_pr_linear(shared_vector)
        else:
            mean = self.mean_linear(shared_vector)
            log_std = self.log_std_linear(shared_vector)
            log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
            std = log_std.exp()
            return mean, std

class ContrastiveModel(torch.nn.Module):
    """
    Description:
        Network module that captures if a given state1, state2 pair belong in the same transition.

    Parameters:
        - params
    """
    def __init__(self, params):
        super(ContrastiveModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], 1),
        )

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)
    
class TL_Model(torch.nn.Module):
    """
    Description:
        Network module that captures transition probability between any two state.

    Parameters:
        - params
    """
    def __init__(self, params):
        super(TL_Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], 1),
        )

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class AL_Model(torch.nn.Module):
    """
    Description:
        Network module that captures action between any two state.

    Parameters:
        - params
    """
    def __init__(self, params):
        super(AL_Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], 1),
        )

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class RL_Model(torch.nn.Module):
    """
    Description:
        Network module that captures reward between any two state.

    Parameters:
        - params
    """
    def __init__(self, params):
        super(RL_Model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(params['latent_dim'] * 2, params['layer_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(params['layer_size'], 1),
        )

    def forward(self, z0, z1):
        context = torch.cat((z0, z1), -1)
        return self.model(context)

class SISAHead(torch.nn.Module):
    """
    Description:
        Network module that combines contrastive and inverse models.

    Parameters:
        - params
        - action_space : Int
            The environment's action space
    """
    def __init__(self, params, action_shape, log_freq=10000):
        super(SISAHead, self).__init__()

        self.n_actions = action_shape[0]
        self.discrete = False
        self.log_freq = log_freq

        self.inverse_model = InverseModel(params, self.n_actions, discrete=self.discrete)
        self.discriminator = ContrastiveModel(params)
        self.tl_model = TL_Model(params)
        self.al_model = AL_Model(params)
        self.rl_model = RL_Model(params)

        self.inverse_coef = params['inverse_coef']
        self.contrastive_coef = params['contrastive_coef']
        self.smoothness_coef = params['smoothness_coef']
        self.smoothness_max_dz = params['smoothness_max_dz']

        self.bce = torch.nn.BCEWithLogitsLoss()
        if self.discrete:
            self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def compute_loss(self, z0, a, z1):
        # Inverse loss
        if self.discrete:
            log_pr_actions = self.inverse_model(z0, z1)
            l_inverse = self.ce(input=log_pr_actions, target=a)
        else:
            mean, std = self.inverse_model(z0, z1)
            cov = torch.diag_embed(std, dim1=1, dim2=2)
            normal = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
            log_pr_action = normal.log_prob(a)
            l_inverse = -1 * log_pr_action.mean(dim=0)

        # Ratio loss
        with torch.no_grad():
            N = len(z1)
            idx = torch.randperm(N)  # shuffle indices of next states
        z1_neg = z1.view(N, -1)[idx].view(z1.size())

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_pos_neg = torch.cat([z1, z1_neg], dim=0)
        is_real_transition = torch.cat([torch.ones(N), torch.zeros(N)], dim=0).to(z0.device)
        log_pr_real = self.discriminator(z0_extended, z1_pos_neg)
        l_contr = self.bce(input=log_pr_real, target=is_real_transition.unsqueeze(-1).float())

        # Smoothness loss
        with torch.no_grad():
            dimensionality_scale_factor = torch.sqrt(torch.as_tensor(z0.shape).float()[-1])  # distance scales as ~sqrt(dim)
        dz = torch.norm(z1 - z0, dim=-1, p=2) / dimensionality_scale_factor
        excess = torch.nn.functional.relu(dz - self.smoothness_max_dz)
        l_smoothness = self.mse(excess, torch.zeros_like(excess))

        markov_loss = self.inverse_coef * l_inverse + self.contrastive_coef * l_contr + self.smoothness_coef * l_smoothness
        return markov_loss, l_inverse, l_contr, l_smoothness
    
    def compute_tl_loss(self, azs, bp_vector, cp_matrix):
        x0, x1 = azs[0].unsqueeze(0), azs[1].unsqueeze(0)
        y = torch.tensor(cp_matrix[0][1] / bp_vector[1]).unsqueeze(0)
        for id1 in range(azs.shape[0]):
            for id2 in range(azs.shape[0]):
                if id1 == id2:
                    continue
                if id1 == 0 and id2 == 1:
                    continue
                x0 = torch.cat((x0, azs[id1].unsqueeze(0)), dim=0)
                x1 = torch.cat((x1, azs[id2].unsqueeze(0)), dim=0)
                y = torch.cat((y, torch.tensor(cp_matrix[id1][id2] / bp_vector[id2]).unsqueeze(0)), dim=0)
        y = torch.reshape(y, (-1, 1))
        y_hat = self.tl_model(x0, x1)
        device = torch.device("cpu")
        return self.mse(y.to(device).float(), y_hat.to(device).float())
    
    def compute_al_loss(self, azs, cp_matrix):
        x0, x1 = azs[0].unsqueeze(0), azs[1].unsqueeze(0)
        y = torch.tensor(cp_matrix[0][1]).unsqueeze(0)
        for id1 in range(azs.shape[0]):
            for id2 in range(azs.shape[0]):
                if id1 == id2:
                    continue
                if id1 == 0 and id2 == 1:
                    continue
                x0 = torch.cat((x0, azs[id1].unsqueeze(0)), dim=0)
                x1 = torch.cat((x1, azs[id2].unsqueeze(0)), dim=0)
                y = torch.cat((y, torch.tensor(cp_matrix[id1][id2]).unsqueeze(0)), dim=0)
        y = torch.reshape(y, (-1, 1))
        y_hat = self.al_model(x0, x1)
        device = torch.device("cpu")
        return self.mse(y.to(device).float(), y_hat.to(device).float())
    
    def compute_rl_loss(self, azs, cp_matrix):
        x0, x1 = azs[0].unsqueeze(0), azs[1].unsqueeze(0)
        y = torch.tensor(cp_matrix[0][1]).unsqueeze(0)
        for id1 in range(azs.shape[0]):
            for id2 in range(azs.shape[0]):
                if id1 == id2:
                    continue
                if id1 == 0 and id2 == 1:
                    continue
                x0 = torch.cat((x0, azs[id1].unsqueeze(0)), dim=0)
                x1 = torch.cat((x1, azs[id2].unsqueeze(0)), dim=0)
                y = torch.cat((y, torch.tensor(cp_matrix[id1][id2]).unsqueeze(0)), dim=0)
        y = torch.reshape(y, (-1, 1))
        y_hat = self.rl_model(x0, x1)
        device = torch.device("cpu")
        return self.mse(y.to(device).float(), y_hat.to(device).float())

    def log(self, L, step):
        if step % self.log_freq != 0:
            return

        pass
