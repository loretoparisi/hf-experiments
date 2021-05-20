import os
import sklearn.metrics
import numpy as np
import sys
import time
from . import word_encoder
from . import data_loader
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP

from .viterbi import ViterbiDecoder


def get_abstract_transitions(train_fname):
    """
    Compute abstract transitions on the training dataset for StructShot
    """
    samples = data_loader.FewShotNERDataset(train_fname, None, 1,1,1,1).samples
    tag_lists = [sample.tags for sample in samples]

    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1

    trans = []
    trans.append(s_o / (s_o + s_i))
    trans.append(s_i / (s_o + s_i))
    trans.append(o_o / (o_o + o_i))
    trans.append(o_i / (o_o + o_i))
    trans.append(i_o / (i_o + i_i + x_y))
    trans.append(i_i / (i_o + i_i + x_y))
    trans.append(x_y / (i_o + i_i + x_y))
    return trans

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotNERModel(nn.Module):
    def __init__(self, my_word_encoder):
        '''
        word_encoder: Sentence encoder
        
        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.word_encoder = nn.DataParallel(my_word_encoder)
        # self.word_encoder = my_word_encoder
        self.cost = nn.CrossEntropyLoss(ignore_index=-1)

    
    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class] 
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label,[])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __transform_label_to_tag__(self, pred, query):
        '''
        flatten labels and transform them to string tags
        '''
        pred_tag = []
        label_tag = []
        current_sent_idx = 0 # record sentence index in the batch data
        current_token_idx = 0 # record token indes in the batch data
        assert len(query['sentence_num']) == len(query['label2tag'])
        # iterate by each query set
        for idx, num in enumerate(query['sentence_num']):
            true_label = torch.cat(query['label'][current_sent_idx:current_sent_idx+num], 0).cpu().numpy().tolist()
            set_token_length = len(true_label)
            # use the idx-th label2tag dict
            pred_tag += [query['label2tag'][idx][label] for label in pred[current_token_idx:current_token_idx + set_token_length]]
            label_tag += [query['label2tag'][idx][label] for label in true_label]
            # update sentence and token index
            current_sent_idx += num
            current_token_idx += set_token_length
        assert len(pred_tag) == len(label_tag)
        assert len(pred_tag) == len(pred)
        return pred_tag, label_tag

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error and coarse type error
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span / total_correct_span, wrong_outer_span / total_correct_span
                
    def metrics_by_entity(self, pred, label):
        '''
        return entity level precision, recall, f1 score
        '''
        pred = pred.view(-1).cpu().numpy().tolist()
        label = label.view(-1).cpu().numpy().tolist()
        pred_class_span = self.__get_class_span_dict__(pred)
        label_class_span = self.__get_class_span_dict__(label)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        precision = correct_cnt / (pred_cnt + 1e-6)
        recall = correct_cnt / label_cnt
        f1 = 2*precision*recall / (precision + recall + 1e-6)
        return precision, recall, f1

    def error_analysis(self, pred, label, query):
        '''
        return 
        token level false positive rate and false negative rate
        entity level within error and outer error 
        '''
        fp = torch.mean(((pred.view(-1) != 0) & (label.view(-1) == 0)).type(torch.FloatTensor))
        fn = torch.mean(((pred.view(-1) == 0) & (label.view(-1) != 0)).type(torch.FloatTensor))
        pred = pred.view(-1).cpu().numpy().tolist()
        label = label.view(-1).cpu().numpy().tolist()
        within, outer = self.__get_type_error__(pred, label, query)
        return fp, fn, within, outer


class FewShotNERFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, viterbi=False, N=None, train_fname=None, tau=0.05):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.viterbi = viterbi
        if viterbi:
            abstract_transitions = get_abstract_transitions(train_fname)
            self.viterbi_decoder = ViterbiDecoder(N+2, abstract_transitions, tau)
    
    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)
    
    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self,
              model,
              model_name,
              B, N_for_train, N_for_eval, K, Q,
              learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False):
        '''
        model: a FewShotREModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        '''
        print("Start training...")
    
        # Init optimizer
        print('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if use_sgd_for_bert:
            optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
        else:
            optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 
        
        # load model
        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()

        # Training
        best_f1 = 0
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        iter_precision = 0.0
        iter_recall = 0.0
        iter_f1 = 0.0
        for it in range(start_iter, start_iter + train_iter):
            support, query = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    if k != 'label' and k != 'sentence_num':
                        support[k] = support[k].cuda()
                        query[k] = query[k].cuda()
                label = torch.cat(query['label'], 0)
                label = label.cuda()
            else:
                for k in support:
                    if k != 'label' and k != 'sentence_num':
                        support[k] = support[k]
                        query[k] = query[k]
                label = torch.cat(query['label'], 0)
                label = label

            logits, pred = model(support, query, 
                    N_for_train, K, Q * N_for_train)
            loss = model.loss(logits, label) / float(grad_iter)
            right = model.accuracy(pred, label)
            precision, recall, f1 = model.metrics_by_entity(pred, label)
                
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_precision += precision
            iter_recall += recall
            iter_f1 += f1
            iter_sample += 1
            if (it + 1) % 100 == 0 or (it + 1) % val_step == 0:
                sys.stdout.write('step: {0:4} | loss: {1:2.6f}, token-level accuracy: {2:3.2f}% | [ENTITY] precision: {3:3.4f}, recall: {4:3.4f}, f1: {5:3.4f}'\
                    .format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, \
                                        iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                _, _, f1, _, _, _, _ = self.eval(model, B, N_for_eval, K, Q, val_iter)
                model.train()
                if f1 > best_f1:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_f1 = f1
                iter_loss = 0.
                iter_loss_dis = 0.
                iter_right = 0.
                iter_right_dis = 0.
                iter_sample = 0.
                iter_precision = 0.0
                iter_recall = 0.0
                iter_f1 = 0.0
                
        print("\n####################\n")
        print("Finish training " + model_name)
    
    def __get_emmissions__(self, logits, tags_list):
        # split [num_of_query_tokens, num_class] into [[num_of_token_in_sent, num_class], ...]
        emmissions = []
        current_idx = 0
        for tags in tags_list:
            emmissions.append(logits[current_idx:current_idx+len(tags)])
            current_idx += len(tags)
        assert current_idx == logits.size()[0]
        return emmissions

    def viterbi_decode(self, logits, query_tags):
        emissions_list = self.__get_emmissions__(logits, query_tags)
        pred = []
        for i in range(len(query_tags)):
            sent_scores = emissions_list[i].cpu()
            sent_len, n_label = sent_scores.shape
            sent_probs = F.softmax(sent_scores, dim=1)
            start_probs = torch.zeros(sent_len) + 1e-6
            # start_probs = start_probs.cuda()
            sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
            feats = self.viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label+1))
            vit_labels = self.viterbi_decoder.viterbi(feats)
            vit_labels = vit_labels.view(sent_len)
            vit_labels = vit_labels.detach().cpu().numpy().tolist()
            for label in vit_labels:
                pred.append(label-1)
        return torch.tensor(pred).cuda()

    def eval(self,
            model,
            B, N, K, Q,
            eval_iter,
            ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        iter_precision = 0.0
        iter_recall = 0.0
        iter_f1 = 0.0

        iter_fp = 0.0 # misclassify O as I-
        iter_fn = 0.0 # misclassify I- as O
        iter_within = 0.0 # span correct but of wrong fine-grained type 
        iter_outer = 0.0 # span correct but of wrong coarse-grained type
        with torch.no_grad():
            for it in range(eval_iter):
                support, query = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        if k != 'label' and k != 'sentence_num':
                            support[k] = support[k].cuda()
                            query[k] = query[k].cuda()
                    label = torch.cat(query['label'], 0)
                    label = label.cuda()
                logits, pred = model(support, query, N, K, Q * N)
                if self.viterbi:
                    pred = self.viterbi_decode(logits, query['label'])

                right = model.accuracy(pred, label)
                precision, recall, f1 = model.metrics_by_entity(pred, label)
                fp, fn, within, outer = model.error_analysis(pred, label, query)
                iter_right += self.item(right.data)
                iter_precision += precision
                iter_recall += recall
                iter_f1 += f1

                iter_fn += self.item(fn.data)
                iter_fp += self.item(fp.data)
                iter_outer += outer
                iter_within += within
                iter_sample += 1

            sys.stdout.write('[EVAL] step: {0:4} | token-level accuracy: {1:3.2f}% | [ENTITY] precision: {2:3.4f}, recall: {3:3.4f}, f1: {4:3.4f}'\
                .format(it + 1, 100 * iter_right / iter_sample, iter_precision / iter_sample, \
                                    iter_recall / iter_sample, iter_f1 / iter_sample) + '\r')
            sys.stdout.flush()
            print("")
        return iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample, iter_fp / iter_sample, iter_fn / iter_sample, iter_within / iter_sample, iter_outer / iter_sample
