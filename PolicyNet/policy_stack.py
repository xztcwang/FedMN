from PolicyNet.policy_net import PolicyNet

def create_policy(args_, policy_hdim_x, policy_hdim_y, policy_outdim, hard):
    policy_net = PolicyNet(args_, policy_hdim_x, policy_hdim_y, policy_outdim, hard)
    return policy_net
