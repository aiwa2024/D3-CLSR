import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GNN
from torch.nn import Module, Parameter
from einops import rearrange
import torch.nn.functional as F
from utils import *


class Contrast_loss(Module):
    def __init__(self, channels, in_dims, c_dims):
        super(Contrast_loss, self).__init__()
        decoupled_contrastive_learning = True
        self.channels = channels
        self.in_dims = in_dims
        self.c_dims = c_dims
        multiview_loss_weight = 0.1
        self.temperature = nn.Parameter(torch.tensor(1.))
        self.decoupled_contrastive_learning = decoupled_contrastive_learning
        self.multiview_loss_weight = multiview_loss_weight


    def forward(self, item_hidden, cate_hidden):
        item_hidden, cate_hidden = map(l2norm, (item_hidden, cate_hidden))

        item_hiddens = torch.chunk(item_hidden, self.channels, dim=2)
        cate_hiddens = torch.chunk(cate_hidden, self.channels, dim=2)
        imean_hiddens = trans_to_cuda(torch.tensor([], dtype=torch.float))
        cmean_hiddens = trans_to_cuda(torch.tensor([], dtype=torch.float))
        for i in range(self.channels):
            imean_hidden = torch.mean(item_hiddens[i], dim=1, keepdim=False)
            cmean_hidden = torch.mean(cate_hiddens[i], dim=1, keepdim=False)
            imean_hiddens = torch.cat([imean_hiddens, imean_hidden], dim=1)
            cmean_hiddens = torch.cat([cmean_hiddens, cmean_hidden], dim=1)
        b = item_hidden.shape[0]

        temp = self.temperature.exp()

        imean_latents = rearrange(imean_hiddens, 'b (m h) ->m b h', m=5)
        cmean_latents = rearrange(cmean_hiddens, 'b (m h) ->m b h', m=5)
        item_to_cate = torch.einsum('m t d, n i d -> m n t i', imean_latents, cmean_latents) * temp
        cate_to_item = rearrange(item_to_cate, '... t i -> ... i t')

        item_to_cate = rearrange(item_to_cate, 'm n ... -> (m n) ...')
        cate_to_item = rearrange(cate_to_item, 'm n ... -> (m n) ...')

        item_to_cate_exp, cate_to_item_exp = map(torch.exp, (item_to_cate, cate_to_item))
        item_to_cate_pos, cate_to_item_pos = map(matrix_diag, (item_to_cate_exp, cate_to_item_exp))

        if self.decoupled_contrastive_learning:
            pos_mask = trans_to_cuda(torch.eye(b, dtype=torch.bool))
            item_to_cate_exp, cate_to_item_exp = map(lambda t: t.masked_fill(pos_mask, 0.),
                                                     (item_to_cate_exp, cate_to_item_exp))

        item_to_cate_denom, cate_to_item_denom = map(lambda t: t.sum(dim=-1), (item_to_cate_exp, cate_to_item_exp))

        item_to_cate_loss = -log(item_to_cate_pos / item_to_cate_denom).mean(dim=-1)
        cate_to_item_loss = -log(cate_to_item_pos / cate_to_item_denom).mean(dim=-1)

        cl_loss = (item_to_cate_loss + cate_to_item_loss) / 2

        contrast_loss = cl_loss.mean()

        return contrast_loss


class CombineGraph(Module):
    def __init__(self, opt, num_total, n_category, category):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_total = num_total
        self.dim = opt.hiddenSize
        self.in_dims = opt.in_dims[-1]
        self.channels = opt.channels[-1]
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample

        self.n_category = n_category
        self.category = category

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_total, self.dim)
        self.pos = nn.Embedding(200, self.dim)

        self.c_dims = self.dim // self.channels
        self.split_sections = [self.c_dims] * self.channels

        self.disen_agg = nn.ModuleList([LocalAggregator(self.c_dims, self.opt.alpha) for i in range(self.channels)])
        self.gnn = nn.ModuleList([GNN(self.c_dims) for i in range(self.channels)])
        self.disen_mixagg = nn.ModuleList([LocalAggregator(self.c_dims, self.opt.alpha) for i in range(self.channels)])

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3 * self.dim, 1))
        self.w_s = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.glu1 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=True)
        self.glu3 = nn.Linear(self.dim, self.dim, bias=True)

        self.w_k = 10


        self.bbb = Parameter(torch.Tensor(1))
        self.ccc = Parameter(torch.Tensor(1))

        self.classifier = nn.Linear(self.c_dims, self.channels)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.intent_loss = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step,
                                                         gamma=opt.lr_dc)
        self.reset_parameters()

        item = []
        for x in range(1, num_total + 1 - n_category):
            item += [category[x]]
        item = np.asarray(item)
        self.item = trans_to_cuda(torch.Tensor(item).long())

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_disentangle_loss(self, intents_item, intents_cate):

        labels1 = [torch.ones(f.shape[0]) * i for i, f in enumerate(intents_item)]
        labels1 = trans_to_cuda(torch.cat(tuple(labels1), 0)).long()
        intents_item = torch.cat(tuple(intents_item), 0)

        pred1 = self.classifier(intents_item)
        discrimination_loss1 = self.loss_function(pred1, labels1)

        labels2 = [torch.ones(f.shape[0]) * i for i, f in enumerate(intents_cate)]
        labels2 = trans_to_cuda(torch.cat(tuple(labels2), 0)).long()
        intents_cate = torch.cat(tuple(intents_cate), 0)

        pred2 = self.classifier(intents_cate)
        discrimination_loss2 = self.loss_function(pred2, labels2)

        discrimination_loss = (discrimination_loss1 + discrimination_loss2) / 2

        return discrimination_loss

    def mix(self, hidden1, hidden2, hidden1_mix, hidden2_mix):
        item_hidden = hidden1 + hidden1_mix * self.bbb
        cate_hidden = hidden2 + hidden2_mix * self.ccc

        return item_hidden, cate_hidden


    def compute_scores(self, item_hidden, cate_hidden, mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = item_hidden.shape[0]
        len = item_hidden.shape[1]


        pos_emb = self.pos.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        item_hs = torch.sum(item_hidden * mask, -2) / torch.sum(mask, 1)
        item_hs = item_hs.unsqueeze(-2).repeat(1, len, 1)
        item_ht = item_hidden[:, 0, :]
        item_ht = item_ht.unsqueeze(-2).repeat(1, len, 1)  # (b, N, dim)

        item_nh = torch.matmul(torch.cat([pos_emb, item_hidden], -1), self.w_1)
        item_nh = torch.tanh(item_nh)

        item_hs = torch.cat([item_hs, item_ht], -1).matmul(self.w_s)

        item_feat = item_hs * item_hidden
        item_nh = torch.sigmoid(torch.cat([self.glu1(item_nh), self.glu2(item_hs), self.glu3(item_feat)], -1))

        beta1 = torch.matmul(item_nh, self.w_2)
        beta1 = beta1 * mask


        select_item = torch.sum(beta1 * item_hidden, 1)

        score_item = []
        select_item = torch.split(select_item, self.split_sections, dim=-1)
        b = torch.split(self.embedding.weight[1:self.num_total - self.n_category + 1], self.split_sections, dim=-1)
        for i in range(self.channels):
            sess_emb_item = self.w_k * select_item[i]
            item_embeddings_int = b[i]
            scores_int_item = torch.mm(sess_emb_item, torch.transpose(item_embeddings_int, 1, 0))
            score_item.append(scores_int_item)

        item_score = torch.stack(score_item, dim=1)  # (b ,k, item_num)
        item_scores = item_score.sum(1)

        cate_hs = torch.sum(cate_hidden * mask, -2) / torch.sum(mask, 1)
        cate_hs = cate_hs.unsqueeze(-2).repeat(1, len, 1)
        cate_ht = cate_hidden[:, 0, :]
        cate_ht = cate_ht.unsqueeze(-2).repeat(1, len, 1)  # (b, N, dim)

        cate_nh = torch.matmul(torch.cat([pos_emb, cate_hidden], -1), self.w_1)
        cate_nh = torch.tanh(cate_nh)

        cate_hs = torch.cat([cate_hs, cate_ht], -1).matmul(self.w_s)

        cate_feat = cate_hs * cate_hidden
        cate_nh = torch.sigmoid(torch.cat([self.glu1(cate_nh), self.glu2(cate_hs), self.glu3(cate_feat)], -1))

        beta2 = torch.matmul(cate_nh, self.w_2)
        beta2 = beta2 * mask


        select_cate = torch.sum(beta2 * cate_hidden, 1)

        score_cate = []
        select_cate = torch.split(select_cate, self.split_sections, dim=-1)
        item_category = torch.split(self.embedding(self.item), self.split_sections, dim=-1)
        for i in range(self.channels):
            sess_emb_cate = self.w_k * select_cate[i]
            cate_embeddings_int = item_category[i]
            scores_cate = torch.mm(sess_emb_cate, torch.transpose(cate_embeddings_int, 1, 0))
            score_cate.append(scores_cate)

        cate_score = torch.stack(score_cate, dim=1)  # (b ,k, item_num)
        cate_scores = cate_score.sum(1)

        scores = item_scores + cate_scores

        return scores

    def forward(self, inputs, adj, mask_item, items, items_ID, adj_ID, total_items, total_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        hidden1 = self.embedding(inputs)
        hidden2 = self.embedding(items_ID)
        hidden_mix = self.embedding(total_items)
        hidden1 = torch.split(hidden1, self.split_sections, dim=-1)
        hidden2 = torch.split(hidden2, self.split_sections, dim=-1)
        hidden_mix = torch.split(hidden_mix, self.split_sections, dim=-1)
        hidden1s = []
        hidden2s = []
        hidden_mixs = []
        intents_item = []
        intents_cate = []

        # local
        for i in range(self.channels):
            hidden1_i = self.gnn[i](hidden1[i], adj)
            hidden2_c = self.gnn[i](hidden2[i], adj_ID)

            hidden_m = self.disen_mixagg[i](hidden_mix[i], total_adj)
            hidden1s.append(hidden1_i)
            hidden2s.append(hidden2_c)
            hidden_mixs.append(hidden_m)

            h_init = torch.mean(hidden1_i, dim=1)
            h_inca = torch.mean(hidden2_c, dim=1)
            intents_item.append(h_init)
            intents_cate.append(h_inca)

        hidden1s_item = torch.stack(hidden1s, dim=2)
        hidden2s_cate = torch.stack(hidden2s, dim=2)
        hidden_mixs_all = torch.stack(hidden_mixs, dim=2)

        h1_local = hidden1s_item.reshape(batch_size, seqs_len, self.dim)
        h2_local = hidden2s_cate.reshape(batch_size, seqs_len, self.dim)
        h_mix = hidden_mixs_all.reshape(batch_size, 2*seqs_len, self.dim)

        # combine
        h1_local = F.dropout(h1_local, self.dropout_local, training=self.training)
        h2_local = F.dropout(h2_local, self.dropout_local, training=self.training)
        h_mix = F.dropout(h_mix, self.dropout_local, training=self.training)
        self.intent_loss = self.compute_disentangle_loss(intents_item, intents_cate)

        return h1_local, h2_local, h_mix


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward_cor(model, data):
    (alias_inputs, adj, items, mask, targets, targets_ID, inputs, alias_inputs_ID, adj_ID, items_ID,
     alias_items, alias_category, total_adj, total_items) = data
    alias_items = trans_to_cuda(alias_items).long()
    alias_category = trans_to_cuda(alias_category).long()
    total_adj = trans_to_cuda(total_adj).float()
    total_items = trans_to_cuda(total_items).long()

    alias_inputs_ID = trans_to_cuda(alias_inputs_ID).long()
    items_ID = trans_to_cuda(items_ID).long()
    adj_ID = trans_to_cuda(adj_ID).float()

    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()


    hidden1, hidden2, hidden_mix = model(items, adj, mask, inputs, items_ID, adj_ID, total_items, total_adj)

    get1 = lambda i: hidden1[i][alias_inputs[i]]
    seq_hidden1 = torch.stack([get1(i) for i in torch.arange(len(alias_inputs)).long()])
    get2 = lambda i: hidden2[i][alias_inputs_ID[i]]
    seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs_ID)).long()])

    get1_mix = lambda i: hidden_mix[i][alias_items[i]]
    seq_hidden1_mix = torch.stack([get1_mix(i) for i in torch.arange(len(alias_items)).long()])
    get2_mix = lambda i: hidden_mix[i][alias_category[i]]
    seq_hidden2_mix = torch.stack([get2_mix(i) for i in torch.arange(len(alias_category)).long()])

    item_hidden, cate_hidden = model.mix(seq_hidden1, seq_hidden2, seq_hidden1_mix, seq_hidden2_mix)

    scores = model.compute_scores(item_hidden, cate_hidden, mask)

    return targets, targets_ID, scores, item_hidden, cate_hidden


def train_test(model, train_data, test_data, contrast_model):
    print('start training: ', datetime.datetime.now())
    model.train()
    contrast_model.train()
    total_loss = 0.0
    total_loss_2 = 0.0
    total_loss_3 = 0.0
    lamda = 1e-4
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.batch_size,
                                               shuffle=True,
                                               pin_memory=True)

    # for data in tqdm(train_loader):
    #     model.optimizer.zero_grad()
    #     targets, scores, item_hidden, cate_hidden = forward_cor(model, data)
    #     targets = trans_to_cuda(targets).long()
    #     item_hidden = trans_to_cuda(item_hidden)
    #     cate_hidden = trans_to_cuda(cate_hidden)
    #     loss_1 = model.loss_function(scores, targets - 1)
    #     loss_2 = contrast_model(item_hidden, cate_hidden)
    #     loss = loss_1 + loss_2 + model.intent_loss
    #     loss.backward()
    #     model.optimizer.step()
    #     total_loss_3 += model.intent_loss
    #     total_loss_2 += loss_2
    #     total_loss += loss
    # print('\tLoss:\t%.3f' % total_loss)
    # print('\tcontrast_Loss:\t%.3f' % total_loss_2)
    # print('\tcor_Loss\t%.3f' % total_loss_3)
    # model.scheduler.step()
    #
    # print('start predicting: ', datetime.datetime.now())
    # model.eval()
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
    #                                           shuffle=False, pin_memory=True)
    # result = []
    # hit_k10, mrr_k10, hit_k20, mrr_k20 = [], [], [], []
    #
    # for data in test_loader:
    #     targets, scores, item_hidden, cate_hidden = forward_cor(model, data)
    #     sub_scores_k20 = scores.topk(20)[1]
    #     sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
    #     sub_scores_k10 = scores.topk(10)[1]
    #     sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()
    #     targets = targets.numpy()
    #
    #     for score, target, mask in zip(sub_scores_k20, targets, test_data.mask):
    #         hit_k20.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr_k20.append(0)
    #         else:
    #             mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))
    #
    #     for score, target, mask in zip(sub_scores_k10, targets, test_data.mask):
    #         hit_k10.append(np.isin(target - 1, score))
    #         if len(np.where(score == target - 1)[0]) == 0:
    #             mrr_k10.append(0)
    #         else:
    #             mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))


    # 类别预测
    for data in tqdm(train_loader):  # 每个data中存放100条会话的信息（随机取样且不重复）
        model.optimizer.zero_grad()  # 上一步的损失清零
        targets, targets_ID, scores, item_hidden, cate_hidden = forward_cor(model, data)
        targets = trans_to_cuda(targets).long()
        targets_ID = trans_to_cuda(targets_ID).long()
        item_hidden = trans_to_cuda(item_hidden)
        cate_hidden = trans_to_cuda(cate_hidden)
        loss_1 = model.loss_function(scores, targets_ID-997) #交叉熵损失
        # loss_4 = model.loss_function(scores, targets-1)
        loss_2 = contrast_model(item_hidden, cate_hidden) #对比学习损失
        loss = loss_1 + loss_2 + model.intent_loss
        loss.backward()  # 反向传播
        model.optimizer.step()  # 优化
        total_loss_3 += model.intent_loss
        total_loss_2 += loss_2
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    print('\tcontrast_Loss:\t%.3f' % total_loss_2)
    print('\tcor_Loss\t%.3f' % total_loss_3)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()  # 指定模型为计算模式
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    #    hit, mrr = [], []
    hit_k10, mrr_k10, hit_k20, mrr_k20 = [], [], [], []

    for data in test_loader:
        targets, targets_ID, scores, item_hidden, cate_hidden = forward_cor(model, data)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()
        targets = targets.numpy()
        targets_ID = targets_ID.numpy()

        for score, target, mask in zip(sub_scores_k20, targets_ID, test_data.mask):
            hit_k20.append(np.isin(target - 997, score))
            if len(np.where(score == target - 997)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 997)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets_ID, test_data.mask):
            hit_k10.append(np.isin(target - 997, score))
            if len(np.where(score == target - 997)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 997)[0][0] + 1))

    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)

    return result