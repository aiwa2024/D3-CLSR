import time
import argparse
import pickle
from model import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')
parser.add_argument('--dropout_global', type=float, default=0.0, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--corDecay', type=int, default=5,help='Distance Correlation Weight')
parser.add_argument('--dropout', action='store', default=0.1, type=float, help='dropout')
parser.add_argument('--beta', type=float, default=0.1, help='beta in DisConv')
parser.add_argument('--lamda', type=float, default=1e-4, help='aux loss weight')
parser.add_argument('--iterations', type=int, default=2, help='iterations of each DisConv')
parser.add_argument('--weightshare', type=bool, default=False,help='wether layer share weight')

opt = parser.parse_args()
opt.channels = [5]
opt.c_dims = [20]
opt.in_dims = [100]


def main():
    init_seed()

    if opt.dataset == 'diginetica':
        num_node = 43098  #(43097+1)
        opt.dropout_local = 0.0
        n_category = 996    #(995+1)
    elif opt.dataset == 'Nowplaying':
        num_node = 60417  #(60416+1)
        opt.dropout_local = 0.0
        n_category = 11462 #(11461 + 1)
    elif opt.dataset == 'Tmall':
        num_node = 40728   #40727 + 1
        opt.dropout_local = 0.6
       # opt.dropout_local = 0
        n_category = 712 #711 + 1
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data

    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))



    category = pickle.load(open('datasets/' + opt.dataset + '/category.txt', 'rb'))


    train_data = Data(train_data,category)
    test_data = Data(test_data,category)

    model = trans_to_cuda(CombineGraph(opt, num_node+n_category-1, n_category, category))
    Contrast_loss_model = trans_to_cuda(Contrast_loss(opt.channels[-1], opt.in_dims[-1], opt.c_dims[-1]))

    start = time.time()
    best_result_k10 = [0, 0]
    best_result_k20 = [0, 0]


    best_epoch_k10 = [0, 0]
    best_epoch_k20 = [0, 0]
    bad_counter_k20 = bad_counter_k10 = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
   #     hit, mrr = train_test(model, train_data, test_data)
        hit_k10, mrr_k10, hit_k20, mrr_k20 = train_test(model, train_data, test_data, Contrast_loss_model)

        flag_k10 = 0
        if hit_k10 >= best_result_k10[0]:
            best_result_k10[0] = hit_k10
            best_epoch_k10[0] = epoch
            flag_k10 = 1
        if mrr_k10 >= best_result_k10[1]:
            best_result_k10[1] = mrr_k10
            best_epoch_k10[1] = epoch
            flag_k10 = 1
        print("\n")
        print('Best @10 Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (
            best_result_k10[0], best_result_k10[1]))
        bad_counter_k10 += 1 - flag_k10

        flag_k20 = 0
        if hit_k20 >= best_result_k20[0]:
            best_result_k20[0] = hit_k20
            best_epoch_k20[0] = epoch
            flag_k20 = 1
        if mrr_k20 >= best_result_k20[1]:
            best_result_k20[1] = mrr_k20
            best_epoch_k20[1] = epoch
            flag_k20 = 1
        print('Best @20 Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (
            best_result_k20[0], best_result_k20[1]))
        bad_counter_k20 += 1 - flag_k20

        if ((bad_counter_k20 >= opt.patience) and (bad_counter_k10 >= opt.patience)):
            break


    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
