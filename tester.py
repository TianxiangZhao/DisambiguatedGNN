from GCL.eval import get_split
import torch
import ipdb

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F
import utils
from GCL.eval import BaseEvaluator
from plots import plot_chart, plot_bar, plot_tsne_anchored, plot_pca_anchored
import numpy as np
from scipy.sparse import coo_matrix

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval
        self.classifier = None
        self.best_logits = None

        

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict, class_wise=False):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        self.classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(self.classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0
        self.best_logits = None
        best_acc = 0

        if class_wise:
            #
            class_list, class_count = torch.unique(y,return_counts=True)
            ordered_class = torch.argsort(class_count, descending=True)
            class_group_length = class_list.shape[0]//4

            y_test = y[split['test']]
            test_maj = np.array([y_i in ordered_class[:class_group_length] for y_i in y_test],dtype=bool)
            test_midmaj = np.array([y_i in ordered_class[class_group_length:class_group_length*2] for y_i in y_test],dtype=bool)
            test_midmin = np.array([y_i in ordered_class[class_group_length*2:class_group_length*3] for y_i in y_test],dtype=bool)
            test_min = np.array([y_i in ordered_class[class_group_length*3:] for y_i in y_test],dtype=bool)

            # measure the balancedness
            best_balancedness = 0



        #with tqdm(total=self.num_epochs, desc='(LR)',
        #          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
        for epoch in range(self.num_epochs):
            self.classifier.train()
            optimizer.zero_grad()

            output = self.classifier(x[split['train']])
            loss = criterion(output_fn(output), y[split['train']])

            loss.backward()
            optimizer.step()

            if (epoch + 1) % self.test_interval == 0:
                self.classifier.eval()
                y_test = y[split['test']].detach().cpu().numpy()
                y_logit = self.classifier(x[split['test']]).detach()
                y_pred = y_logit.argmax(-1).detach().cpu().numpy()
                test_micro = f1_score(y_test, y_pred, average='micro')
                test_macro = f1_score(y_test, y_pred, average='macro')
                test_acc = utils.accuracy(y_logit.cpu().numpy(), y_test)
                
                y_val = y[split['valid']].detach().cpu().numpy()
                y_pred = self.classifier(x[split['valid']]).argmax(-1).detach().cpu().numpy()
                val_micro = f1_score(y_val, y_pred, average='macro')
                
                if y_test.max() > 1:
                    auc_score = roc_auc_score(y_test, F.softmax(y_logit, dim=-1).detach().cpu(), average='macro', multi_class='ovr')
                else:
                    auc_score = roc_auc_score(y_test, F.softmax(y_logit, dim=-1)[:,1].detach().cpu(), average='macro')

                if val_micro > best_val_micro:
                    best_val_micro = val_micro
                    best_test_micro = test_micro
                    best_test_macro = test_macro
                    best_epoch = epoch
                    best_acc = test_acc
                    best_auc = auc_score
                    with torch.no_grad():
                        self.best_logits = F.log_softmax(self.classifier(x), dim=-1).detach()

                    #pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    #pbar.update(self.test_interval)

                    balancedness = utils.balance_precision(y_logit.cpu().numpy(), y_test)
                    best_balancedness = balancedness

                    if class_wise:
                        test_maj_acc = utils.accuracy(y_logit.cpu().numpy()[test_maj], y_test[test_maj])
                        test_midmaj_acc = utils.accuracy(y_logit.cpu().numpy()[test_midmaj], y_test[test_midmaj])
                        test_midmin_acc = utils.accuracy(y_logit.cpu().numpy()[test_midmin], y_test[test_midmin])
                        test_min_acc = utils.accuracy(y_logit.cpu().numpy()[test_min], y_test[test_min])

        if class_wise:
            return {
                'micro_f1': best_test_micro,
                'macro_f1': best_test_macro,
                'accuracy': best_acc,
                'auc': best_auc,
                'maj_acc': test_maj_acc,
                'midmaj_acc': test_midmaj_acc,
                'midmin_acc': test_midmin_acc,
                'min_acc': test_min_acc,
                'balancedness': best_balancedness,
            }

        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'accuracy': best_acc,
            'auc': best_auc,
            'balancedness': best_balancedness,
        }

def test_emb(model, test_dataloader, return_data=False, train_ratio=0.1, val_ratio=0.1, split=None, analyze_neighbor=False, class_wise=False):
    model.eval()

    zs = []
    ys = []
    group1s = []
    group2s = []
    edge_indexes= [] 
    cs_labels = []
    for data in test_dataloader:
        with torch.no_grad():
            data = data.to(next(model.parameters()).device)
            z = model.embedding(data.x, data.edge_index, data.edge_attr)
        zs.append(z)
        ys.append(data.y)
        if return_data:
            cs_labels.append(data.cs_label)
            group1s.append(data.group_label)
            group2s.append(data.group_label2)
            edge_indexes.append(data.edge_index)

    z = torch.cat(zs, dim=0)
    y = torch.cat(ys, dim=0)
    if return_data:
        cs_label = torch.cat(cs_labels, dim=0)
        group1 = torch.cat(group1s, dim=0)
        group2 = torch.cat(group2s, dim=0)
        edge_indexes = torch.cat(edge_indexes, dim=-1)

    if split is None:
        split = get_split(num_samples=z.size()[0], train_ratio=train_ratio, test_ratio=val_ratio)
    evaluator = LREvaluator()
    result = evaluator.evaluate(z, y, split, class_wise=class_wise)


    if return_data:
        #collect class prototype weight
        class_weights = torch.norm(evaluator.classifier.fc.weight, dim=-1).detach()
        class_biases = torch.abs(evaluator.classifier.fc.bias).detach()
        grouped_weights = []
        grouped_biases = []
        for cs_id in range(len(set(cs_label.cpu().numpy()))):
            sel_ins = cs_label==cs_id
            sel_classes = np.unique(y[sel_ins].cpu().numpy())
            grouped_weights.append(class_weights[sel_classes].mean().item())
            grouped_biases.append(class_biases[sel_classes].mean().item())
        fig_weight = plot_chart([np.array(grouped_weights)], name_list=['group weight norm'], x_start=1, x_name='group main', y_name='value')
        result['fig_weight'] = fig_weight
        fig_bias = plot_chart([np.array(grouped_biases)], name_list=['group bias norm'], x_start=1, x_name='group main', y_name='value')
        result['fig_bias'] = fig_bias


        # group-wise perform
        logits = evaluator.best_logits.cpu().numpy()
        y = y.cpu().numpy()
        group1 = group1.cpu().numpy()
        group2 = group2.cpu().numpy()
        edge_indexes = edge_indexes.cpu()

        group_acc1 = utils.grouped_accuracy(logits[split['test']], y[split['test']],group1[split['test']])

        fig1 = plot_chart([np.array(group_acc1)], name_list=['group main accuracy'], x_start=1, x_name='group main', y_name='acc')
        result['fig_1'] = fig1

        group_acc2 = utils.grouped_accuracy(logits[split['test']], y[split['test']],group2[split['test']])
        fig2 = plot_chart([np.array(group_acc2)], name_list=['group margin accuracy'], x_start=1, x_name='group margin', y_name='acc')
        result['fig_2'] = fig2

        # group-wise embedding similarity across different types of neighbors. similarity measured as cosine distance
        if analyze_neighbor:
            # group-wise embedding magnitude
            emb_scale = z.norm(dim=-1).cpu().numpy()
            group_scale = utils.grouped_measure(emb_scale, group1)
            fig3 = plot_chart([np.array(group_scale)], name_list=['group emb scale'], x_start=1, x_name='group', y_name='F norm')
            result['fig_scale'] = fig3

            adj_coo = coo_matrix((torch.ones(edge_indexes.shape[1]), (edge_indexes[0],edge_indexes[1])),shape=(y.shape[0],y.shape[0]))
            adj_csr = adj_coo.tocsr()
            group_names=['maj-same', 'maj-min', 'maj-othr','min-same', 'min-min', 'min-othr','mid-same', 'mid-min', 'mid-othr']
            with torch.no_grad():
                embedding = z.detach()
                similarity = embedding @ embedding.T
            group_sims, grid_names = utils.grided_similarity(similarity.cpu().numpy(), group1, y, adj_csr)
            fig4 = plot_bar(group_sims, grid_names, y_name='Inner-Product', x_name=group_names)
            result['grided_sim'] = fig4
            
            '''
            embedding = F.normalize(z.detach())
            with torch.no_grad():
                similarity = embedding @ embedding.T
            group_sims, grid_names = utils.grided_similarity(similarity.cpu().numpy(), group1, y, adj_csr)
            fig5 = plot_bar(group_sims, grid_names, y_name='Inner-Product', x_name=group_names)
            result['grided_sim_normed'] = fig5   
            '''     
        
            # TSNE visualization
            sel_index = np.random.choice(z.shape[0], 2000, replace=False)
            fig_tsne = plot_tsne_anchored(z.cpu().numpy()[sel_index], y[sel_index], evaluator.classifier.fc.weight.detach().cpu().numpy(), anchor_label=np.unique(y))
            result['fig_tsne'] = fig_tsne
            
            fig_pca = plot_pca_anchored(z.cpu().numpy()[sel_index], y[sel_index], evaluator.classifier.fc.weight.detach().cpu().numpy(), anchor_label=np.unique(y))
            result['fig_pca'] = fig_pca



        return result
    else:
        return result