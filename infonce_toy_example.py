import torch
import torch.nn.functional as F


B, S, U = 2, 16, 3 # batch_size, max_sequence_length, hidden_size
"""
Sentence 1:
0   /
1   a
2   bad
3   text
4   /
5   good
6   /
7   poor        [label]
8   /
9   moderate
10  /

Sentence 2:
0   /
1   a
2   mission
3   to
4   stop
5   the
6   war
7   /
8   finish up   [label]
9   /
10  continue
11  /
12  start
13  /
14  reopen
15  /
"""

#################### Manual Loss Calculation ####################
# Sentence 1
embeddings1 = torch.rand((S, U))
embeddings1 = F.normalize(embeddings1, dim=-1)
sim_2_5 = embeddings1[2, :] @ embeddings1[5, :].reshape(1, -1).T
sim_2_7 = embeddings1[2, :] @ embeddings1[7, :].reshape(1, -1).T
sim_2_9 = embeddings1[2, :] @ embeddings1[9, :].reshape(1, -1).T
manual_loss1 = - torch.log(torch.exp(sim_2_7) / (torch.exp(sim_2_5) + torch.exp(sim_2_7) + torch.exp(sim_2_9)))

# Sentence 2
embeddings2 = torch.rand((S, U))
embeddings2 = F.normalize(embeddings2, dim=-1)
sim_4_8 = embeddings2[4, :] @ embeddings2[8, :].reshape(1, -1).T
sim_4_10 = embeddings2[4, :] @ embeddings2[10, :].reshape(1, -1).T
sim_4_12 = embeddings2[4, :] @ embeddings2[12, :].reshape(1, -1).T
sim_4_14 = embeddings2[4, :] @ embeddings2[14, :].reshape(1, -1).T
manual_loss2 = - torch.log(torch.exp(sim_4_8) / (torch.exp(sim_4_8) + torch.exp(sim_4_10) + torch.exp(sim_4_12) + torch.exp(sim_4_14)))

# Mean manual loss
mean_manual_loss = (manual_loss1 + manual_loss2) / 2
print(f'Manual Loss\t{mean_manual_loss.item()}')



#################### Torch Loss Calculation ####################
embeddings = torch.rand((B, S, U)) # Values are not used(!), just for creating the tensor :)
embeddings[0, :, :] = embeddings1
embeddings[1, :, :] = embeddings2

# Create the mask
mask = torch.ones((B, S, S), dtype=torch.bool)

mask[0, 2, 5] = 0
mask[0, 2, 7] = 0
mask[0, 2, 9] = 0

mask[1, 4, 8] = 0
mask[1, 4, 10] = 0
mask[1, 4, 12] = 0
mask[1, 4, 14] = 0

# Compute the similarity matrix
sim_mat = torch.bmm(embeddings, embeddings.transpose(1, 2))
sim_mat = sim_mat.masked_fill(mask, - torch.inf)
sim_mat = sim_mat.transpose(0, 2).transpose(0, 1)

# Creating labels
target = torch.full((B, S), -100, dtype=torch.long)
target[0, 2] = 7
target[1, 4] = 8
target = target.T

# Calculate the loss
torch_loss = F.cross_entropy(sim_mat, target, reduction='sum') / B
print(f'Torch Loss\t{torch_loss.item()}')
