import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2, negative_samples):
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Positive similarity
        positive_sim = torch.sum(z1 * z2, dim=1) / self.temperature
        
        # Negative similarity
        z1_expanded = z1.unsqueeze(1)
        neg_samples_embedded = z2[negative_samples]
        negative_sim = torch.sum(z1_expanded * neg_samples_embedded, dim=2) / self.temperature
        
        # InfoNCE loss
        logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
        labels = torch.zeros(len(z1), device=z1.device, dtype=torch.long)
        
        return F.cross_entropy(logits, labels) 