import torch

if __name__ == "__main__":

    m = torch.nn.Linear(20, 30).to("cuda")
    input = torch.randn(128, 20).to("cuda")
    output = m(input)
    print('output', output.size())
    exit()