"""
Simulated Data Generator

This module provides simulated energy data when a real energy prediction model is not available,
to demonstrate the functionality of energy-aware trajectory planning.
"""

import numpy as np
import torch
import torch.nn as nn

class SimulatedEnergyModel(nn.Module):
    """Simulated Energy Prediction Model
    
    Generates approximate energy predictions based on physical rules when the actual model is unavailable.
    """
    
    def __init__(self):
        """Initialize simulated energy model"""
        super(SimulatedEnergyModel, self).__init__()
        
        # Simple linear layer, just to simulate PyTorch model interface
        self.linear = nn.Linear(20, 2)
    
    def forward(self, x):
        """Forward pass function
        
        Simulates energy consumption based on physical intuition:
        - Current is proportional to acceleration (F = ma)
        - Voltage varies with height and speed (potential and kinetic energy)
        
        Args:
            x: Input features [batch_size, seq_len, features]
            
        Returns:
            Predicted current and voltage [batch_size, 2]
        """
        batch_size = x.shape[0]
        
        # Extract features from last time step
        last_step = x[:, -1, :]
        
        # Extract position, velocity, and acceleration-related features
        pos = last_step[:, 3:6]  # pos_x, pos_y, pos_z
        vel = last_step[:, 6:9]  # vel_x, vel_y, vel_z
        
        # Calculate speed and height
        speed = torch.sqrt(torch.sum(vel**2, dim=1))
        height = pos[:, 2]  # z-axis height
        
        # Simulate current and voltage
        # Current proportional to speed (kinetic energy ~ 1/2 m v^2)
        current = 2.0 + 0.5 * speed + 0.01 * torch.randn(batch_size)
        
        # Voltage proportional to height (potential energy ~ mgh)
        voltage = 10.0 + 0.5 * height + 0.2 * speed + 0.01 * torch.randn(batch_size)
        
        # Ensure reasonable ranges
        current = torch.clamp(current, min=0.5, max=20.0)
        voltage = torch.clamp(voltage, min=9.0, max=12.0)
        
        # Combine results
        result = torch.stack([current, voltage], dim=1)
        
        return result

def create_simulated_model():
    """Create a simulated model
    
    Returns:
        Simulated energy prediction model
    """
    model = SimulatedEnergyModel()
    model.eval()  # Set to evaluation mode
    return model

class SimulatedScaler:
    """Simulated data scaler
    
    Mimics sklearn's StandardScaler interface
    """
    
    def transform(self, X):
        """'Normalize' input data (actually just a shallow copy)
        
        Args:
            X: Input data
            
        Returns:
            Copied data
        """
        return X.copy()
    
    def inverse_transform(self, X):
        """'Denormalize' data (actually just a shallow copy)
        
        Args:
            X: Input data
            
        Returns:
            Copied data
        """
        return X.copy()

def create_simulated_scalers():
    """Create simulated scalers
    
    Returns:
        Feature scaler and target scaler
    """
    feature_scaler = SimulatedScaler()
    target_scaler = SimulatedScaler()
    return feature_scaler, target_scaler

# Add simulation support to EnergyAwareTrajectory
def add_simulation_support_to_energy_trajectory(energy_trajectory):
    """Add simulated data support to energy trajectory object
    
    When no real model is available, add a simulated model
    
    Args:
        energy_trajectory: EnergyAwareTrajectory object
        
    Returns:
        Modified EnergyAwareTrajectory object
    """
    if energy_trajectory.model is None:
        print("Using simulated energy prediction model...")
        energy_trajectory.model = create_simulated_model()
        
        # Set simulated scalers
        feature_scaler, target_scaler = create_simulated_scalers()
        energy_trajectory.feature_scaler = feature_scaler
        energy_trajectory.target_scaler = target_scaler
    
    return energy_trajectory