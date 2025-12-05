import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trainer.constants import HIDDEN_DIM

def train_irreducible_loss_model(model, train_ds, device='cpu', batch_size=16, epochs=5):
    """
    Phase 1 Step 1: Train the IL Model on a holdout dataset.
    """
    model = model.__class__() # Could roll this into the rho-loss file

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
       for x, y in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
           x = x.view(x.size(0), -1).to(device)
           y = y.to(device)
           optimizer.zero_grad()
           y_pred = model(x)
           loss = loss_fn(y_pred, y)
           loss.backward()
           optimizer.step()
    return model

def compute_irreducible_losses(model, train_ds, device='cpu', batch_size=256):
    """
    Phase 1 Step 2 & 3: Compute Irreducible Loss for every datapoint.
    Returns a tensor of shape (len(train_ds),) containing loss per sample.
    """
    model.eval()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    all_losses = []
    
    # We need to iterate in order to map back to indices 0..N
    # DataLoader with shuffle=False
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            y = y.to(device)
            y_pred = model(x)
            # Compute loss per sample
            losses = loss_fn(y_pred, y)
            all_losses.append(losses.cpu())
            
    return torch.cat(all_losses)

def batch_sampler(train_ds, batch_size, model=None, loss_fn=None, irreducible_losses=None, device='cpu'):
    """
    Phase 2: RHO-LOSS Selection during training.
    """
    assert irreducible_losses is not None

    n_samples = len(train_ds)
    n_batches = n_samples // batch_size

    # Step 1 note: 2x larger than actual batch size
    large_batch_size = batch_size * 2
    small_batch_size = batch_size
    
    for _ in range(n_batches):
        # Step 1: Sample Large Batch (Bt)
        candidate_indices = np.random.choice(n_samples, size=large_batch_size, replace=False)
        
        # Prepare data for candidates
        batch_data = [train_ds[i] for i in candidate_indices]
        batch_x, batch_y = zip(*batch_data)
        
        # Stack and move to device
        x_tensor = torch.stack(batch_x).view(len(candidate_indices), -1).to(device)
        y_tensor = torch.tensor(batch_y).to(device)
        
        # Step 2: Compute Training Loss (current model)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_tensor)
            current_losses = loss_fn(y_pred, y_tensor) # Shape: (large_batch_size,)
        
        model.train() # Switch back if needed, though loop in vision.py sets model.train()
        
        # Step 3: Calculate RHO-LOSS
        il_losses = irreducible_losses[candidate_indices].to(device)
        rho_losses = current_losses - il_losses
        
        # Step 4: Select Best Points (highest RHO-LOSS)
        top_k_local_indices = torch.argsort(rho_losses, descending=True)[:small_batch_size]
        selected_indices = candidate_indices[top_k_local_indices.cpu().numpy()]
        
        yield selected_indices
