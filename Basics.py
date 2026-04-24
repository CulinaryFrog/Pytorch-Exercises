import torch

### PyTorch Fundamentals

###1 Read documentation
###2
sol2 = torch.rand(size = (7,7))
print("2: ", sol2)
###3
rand3 = torch.rand(size = (1,7))
sol3 = sol2 @ rand3.T
print("3: ", sol3)
###4
torch.random.manual_seed(seed = 0)
rand4 = torch.rand(size = (7,7))
rand4_2 = torch.rand(size = (1,7))
sol4 = rand4 @ rand4_2.T
print("4: ",sol4)
###5
torch.cuda.manual_seed(seed = 1234)
###6
torch.random.manual_seed(seed = 1234)
device = "cuda" if torch.cuda.is_available() else "cpu"
sol6 = torch.rand(size = (2,3)).to(device)
sol6_2 = torch.rand(size = (2,3)).to(device)
###7
sol7 = sol6 @ sol6_2.T
print("7: ", sol7)
###8
print("Min: ", sol7.min())
print("Max: ", sol7.max())
###9
print("Min index: ", sol7.argmin())
print("Max index: ", sol7.argmax())
###10
torch.random.manual_seed(seed = 7)
rand10 = torch.rand(size = (1,1,1,10))
sol10 = rand10.squeeze()
print("10: ", sol10)


