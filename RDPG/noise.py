import numpy as np
import copy

class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck noise generator.
    
    This class implements the Ornstein-Uhlenbeck process, which is commonly used in reinforcement learning algorithms
    to add exploration noise to the actions taken by an agent. The process generates temporally correlated noise that
    tends to revert back to a mean value over time.
    
    Parameters:
    - size (int): The size of the noise vector. Default is 1.
    - mu (float): The mean value of the noise. Default is 0.
    - theta (float): The rate at which the noise reverts back to the mean. Default is 0.15.
    - sigma (float): The volatility of the noise. Default is 0.2.
    """
    def __init__(self, size=1, mu=0, theta=0.15, sigma=0.2):
        self.mu = np.ones(size) * mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()
        
    def reset(self):
        """
        Reset the noise to the mean value.
        """
        self.state = copy.copy(self.mu)
    
    def sample(self):
        """
        Generate a sample from the Ornstein-Uhlenbeck process.
        
        Returns:
        - state (ndarray): The generated noise sample.
        """
        s = self.state
        ds = self.theta * (self.mu - s) + self.sigma * np.random.randn(len(s))
        self.state = s + ds
        return self.state