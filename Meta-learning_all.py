from Main_algorithm_GCN.CR_MGC import CR_MGC
from Configurations import *
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.optim import Adam
import Utils

# the range of the number of remained UAVs
meta_type = [i for i in range(2, 201)]

print("Meta Learning Starts...")
print("-----------------------------------")

for mt in meta_type:
    meta_cr_gcm_n = CR_MGC()
    # list of tuples [('', ...), ('',...)]
    meta_params = dict(meta_cr_gcm_n.gcn_network.named_parameters())
    # param name list
    param_name = meta_cr_gcm_n.gcn_network.state_dict().keys()

    # meta training
    num_remain = mt
    meta_seed = 0
    loss_list = []
    for epi in range(config_meta_training_epi):
        # create the training gcn
        training_cr_gcm_n = CR_MGC()
        training_cr_gcm_n.optimizer = Adam(training_cr_gcm_n.gcn_network.parameters(), lr=0.001)
        # decrease the learning rate as the meta learning moves on
        if epi > 100:
            training_cr_gcm_n.optimizer = Adam(training_cr_gcm_n.gcn_network.parameters(), lr=0.0001)
        if epi > 250:
            training_cr_gcm_n.optimizer = Adam(training_cr_gcm_n.gcn_network.parameters(), lr=0.00001)

        # generate the support set of the training task
        meta_training_support = np.zeros((num_remain, 3))
        while True:
            meta_training_support[:, 0] = np.random.rand(num_remain) * config_width
            meta_training_support[:, 1] = np.random.rand(num_remain) * config_length
            meta_training_support[:, 2] = np.random.rand(num_remain) * config_height
            meta_seed += 1
            np.random.seed(meta_seed)
            cf, nc = Utils.check_if_a_connected_graph(meta_training_support, num_remain)
            if not cf:
                # print(cf)
                break
                # endow the initial values of the GCN with the meta parameter
        for key in training_cr_gcm_n.gcn_network.state_dict().keys():
            training_cr_gcm_n.gcn_network.state_dict()[key].copy_(meta_params[key].data)
        # train the network on the support set
        training_cr_gcm_n.train_support_set_single(meta_training_support, num_remain)
        # generate the query set of the training task
        meta_training_query = np.zeros((num_remain, 3))
        while True:
            meta_training_query[:, 0] = np.random.rand(num_remain) * config_width
            meta_training_query[:, 1] = np.random.rand(num_remain) * config_length
            meta_training_query[:, 2] = np.random.rand(num_remain) * config_height
            meta_seed += 1
            np.random.seed(meta_seed)
            cf, nc = Utils.check_if_a_connected_graph(meta_training_query, num_remain)
            if not cf:
                # print(cf)
                break
                # train on the query set and return the gradient
        gradient, loss = training_cr_gcm_n.train_query_set_single(meta_training_query, num_remain)
        print("%d episode %d remain UAVs -- destroy %d UAVs -- loss %f" % (
        epi, num_remain, config_num_of_agents - num_remain, loss))
        loss_list.append(deepcopy(loss))
        # update the meta parameter
        for key in param_name:
            meta_params[key].data += gradient[key].data
        if epi >= 1:
            x_axis = [i for i in range(epi + 1)]
            fig = plt.figure()
            plt.plot(x_axis, loss_list, linewidth=2.0)
            plt.xlim((0, epi + 1))
            plt.ylim((0, 1400))
            plt.savefig('Meta_Learning_Results/meta_loss_pic/meta_%d.png' % num_remain)
            plt.close()
            # plt.show()
    for key in meta_params.keys():
        meta_params[key] = meta_params[key].cpu().data.numpy()
    np.save('Meta_Learning_Results/meta_parameters/meta_%d.npy' % num_remain, meta_params)
