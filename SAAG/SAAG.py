import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.basemodel import BaseModel
from nnet.attention import SelfAttention
from transformers import *

from nnet.modules import Classifier, EncoderLSTM, EmbedLayer, LockedDropout
from nnet.rgcn import RGCN_Layer
from utils.tensor_utils import rm_pad, split_n_pad
import os
from nnet.attention import MultiHeadAttention

class SAAG(BaseModel):
    def __init__(self, params, pembeds, loss_weight=None, sizes=None, maps=None, lab2ign=None):
        super(SAAG, self).__init__(params, pembeds, loss_weight, sizes, maps, lab2ign)
        # contextual semantic information
        self.more_lstm = params['more_lstm']
        if self.more_lstm:
            if 'bert-large' in params['pretrain_l_m'] or 'albert-large' in params['pretrain_l_m']:
                lstm_input = 1024
            elif 'bert-base' in params['pretrain_l_m'] or 'albert-base' in params['pretrain_l_m']:
                lstm_input = 768
            elif 'albert-xlarge' in params['pretrain_l_m']:
                lstm_input = 2048
            elif 'albert-xxlarge' in params['pretrain_l_m']:
                lstm_input = 4096
            elif 'xlnet-large' in params['pretrain_l_m']:
                lstm_input = 1024
        else:
            lstm_input = params['word_dim'] + params['type_dim']
        self.encoder = EncoderLSTM(input_size=lstm_input,
                                   num_units=params['lstm_dim'],
                                   nlayers=params['bilstm_layers'],
                                   bidir=True,
                                   dropout=params['drop_i'])
        pretrain_hidden_size = params['lstm_dim'] * 2
        if 'bert-large' in params['pretrain_l_m'] and 'albert-large-v2' != params['pretrain_l_m']:
            if self.more_lstm:
                pretrain_hidden_size = params['lstm_dim']*2
            else:
                pretrain_hidden_size = 1024
                self.pretrain_lm = BertModel.from_pretrained('bert-large-uncased-whole-word-masking') # bert-base-uncased
        elif params['pretrain_l_m'] == 'bert-base' and params['pretrain_l_m']!='albert-base-v2':
            if self.more_lstm:
                pretrain_hidden_size = params['lstm_dim']*2
            else:
                pretrain_hidden_size = 768
            if params['dataset']=='docred' and os.path.exists('./bert_base') \
                    or params['dataset']=='cdr' and os.path.exists('./biobert_base'):
                if params['dataset']=='docred':
                    self.pretrain_lm = BertModel.from_pretrained('./bert_base/')
                else:
                    self.pretrain_lm = BertModel.from_pretrained('./biobert_base/')
            else:
                self.pretrain_lm = BertModel.from_pretrained('bert-base-uncased') # bert-base-uncased
      
        elif params['pretrain_l_m'] == 'albert-large-v2':
            if self.more_lstm:
                pretrain_hidden_size = params['lstm_dim']*2
            else:
                pretrain_hidden_size = 1024
            self.pretrain_lm = AlbertModel.from_pretrained('albert-large-v2')
        elif params['pretrain_l_m'] == 'albert-xlarge-v2':
            if self.more_lstm:
                pretrain_hidden_size = params['lstm_dim']*2
            else:
                pretrain_hidden_size = 2048
            if os.path.exists('./albert-xlarge-v2/'):
                self.pretrain_lm = AlbertModel.from_pretrained('./albert-xlarge-v2/')
            else:
                self.pretrain_lm = AlbertModel.from_pretrained('albert-xlarge-v2')
        self.pretrain_l_m_linear_re = nn.Linear(pretrain_hidden_size, params['lstm_dim'])

        if params['types']:
            self.type_embed = EmbedLayer(num_embeddings=3,
                                         embedding_dim=params['type_dim'],
                                         dropout=0.0)

        rgcn_input_dim = params['lstm_dim']
        if params['types']:
            rgcn_input_dim += params['type_dim']

        self.rgcn_layer = RGCN_Layer(params, rgcn_input_dim, params['rgcn_hidden_dim'], params['rgcn_num_layers'], relation_cnt=5)
        self.rgcn_linear_re = nn.Linear(params['rgcn_hidden_dim']*2, params['rgcn_hidden_dim'])

        if params['rgcn_num_layers'] == 0:
            input_dim = rgcn_input_dim * 2
        else:
            input_dim = params['rgcn_hidden_dim'] * 2

        if params['local_rep']:
            self.local_rep_layer = Local_rep_layer(params)
            if not params['global_rep']:
                input_dim = params['lstm_dim'] * 2
            else:
                input_dim += params['lstm_dim'] * 2

        if params['finaldist']:
            input_dim += params['dist_dim'] * 2


        if params['context_att']:
            self.self_att = SelfAttention(input_dim, 1.0)
            input_dim = input_dim * 2

        self.mlp_layer = params['mlp_layers']
        if self.mlp_layer>-1:
            hidden_dim = params['mlp_dim']
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(params['mlp_layers'] - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            self.out_mlp = nn.Sequential(*layers)
            input_dim = hidden_dim

        self.classifier = Classifier(in_size=input_dim,
                                     out_size=sizes['rel_size'],
                                     dropout=params['drop_o'])

        self.rel_size = sizes['rel_size']
        self.finaldist = params['finaldist']
        self.context_att = params['context_att']
        self.pretrain_l_m = params['pretrain_l_m']
        self.local_rep = params['local_rep']
        self.query = params['query']
        assert self.query == 'init' or self.query == 'global'
        self.global_rep = params['global_rep']
        self.lstm_encoder = params['lstm_encoder']

    def encoding_layer(self, word_vec, seq_lens):
       
        ys, _ = self.encoder(torch.split(word_vec, seq_lens.tolist(), dim=0), seq_lens)  # 20, 460, 128
        return ys

    def forward(self, batch):

        input_vec = self.input_layer(batch['words'], batch['ners'])

        if self.pretrain_l_m == 'none':
            encoded_seq = self.encoding_layer(input_vec, batch['section'][:, 3])
            encoded_seq = rm_pad(encoded_seq, batch['section'][:, 3])
            encoded_seq = self.pretrain_l_m_linear_re(encoded_seq)
        else:
            context_output = self.pretrain_lm(batch['bert_token'], attention_mask=batch['bert_mask'])[0]

            context_output = [layer[starts.nonzero().squeeze(1)] for layer, starts in
                              zip(context_output, batch['bert_starts'])]
            context_output_pad = []
            for output, word_len in zip(context_output, batch['section'][:, 3]):
                if output.size(0) < word_len:
                    padding = Variable(output.data.new(1, 1).zero_())
                    output = torch.cat([output, padding.expand(word_len - output.size(0), output.size(1))], dim=0)
                context_output_pad.append(output)

            context_output = torch.cat(context_output_pad, dim=0)

            if self.more_lstm:
                context_output = self.encoding_layer(context_output, batch['section'][:, 3])
                context_output = rm_pad(context_output, batch['section'][:, 3])
            encoded_seq = self.pretrain_l_m_linear_re(context_output)

        encoded_seq = split_n_pad(encoded_seq, batch['word_sec'])

        if self.pretrain_l_m == 'none':
            assert self.lstm_encoder
            nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])
        else:
            nodes = self.node_layer(encoded_seq, batch['entities'], batch['word_sec'])

        init_nodes = nodes
        nodes, nodes_info = self.graph_layer(nodes, batch['entities'], batch['section'][:, 0:3])
        nodes, _ = self.rgcn_layer(nodes, batch['rgcn_adjacency'], batch['section'][:, 0:3])
        entity_size = batch['section'][:, 0].max()
        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                      torch.arange(entity_size).to(self.device))
        relation_rep_h = nodes[:, r_idx]
        relation_rep_t = nodes[:, c_idx]
        # relation_rep = self.rgcn_linear_re(relation_rep)  # global node rep

        if self.local_rep:
            entitys_pair_rep_h, entitys_pair_rep_t = self.local_rep_layer(batch['entities'], batch['section'], init_nodes, nodes)
            if not self.global_rep:
                relation_rep_h = entitys_pair_rep_h
                relation_rep_t = entitys_pair_rep_t
            else:
                relation_rep_h = torch.cat((relation_rep_h, entitys_pair_rep_h), dim=-1)
                relation_rep_t = torch.cat((relation_rep_t, entitys_pair_rep_t), dim=-1)

        if self.finaldist:
            dis_h_2_t = batch['distances_dir'] + 10
            dis_t_2_h = -batch['distances_dir'] + 10
            dist_dir_h_t_vec = self.dist_embed_dir(dis_h_2_t)
            dist_dir_t_h_vec = self.dist_embed_dir(dis_t_2_h)
            relation_rep_h = torch.cat((relation_rep_h, dist_dir_h_t_vec), dim=-1)
            relation_rep_t = torch.cat((relation_rep_t, dist_dir_t_h_vec), dim=-1)
        graph_select = torch.cat((relation_rep_h, relation_rep_t), dim=-1)

        if self.context_att:
            relation_mask = torch.sum(torch.ne(batch['multi_relations'], 0), -1).gt(0)
            graph_select = self.self_att(graph_select, graph_select, relation_mask)

        # Classification
        r_idx, c_idx = torch.meshgrid(torch.arange(nodes_info.size(1)).to(self.device),
                                      torch.arange(nodes_info.size(1)).to(self.device))
        select, _ = self.select_pairs(nodes_info, (r_idx, c_idx), self.dataset)
        graph_select = graph_select[select]
        if self.mlp_layer>-1:
            graph_select = self.out_mlp(graph_select)
        graph = self.classifier(graph_select)

        loss, stats, preds, pred_pairs, multi_truth, mask, truth = self.estimate_loss(graph, batch['relations'][select],
                                                                                      batch['multi_relations'][select])

        return loss, stats, preds, select, pred_pairs, multi_truth, mask, truth


class Local_rep_layer(nn.Module):
    def __init__(self, params):
        super(Local_rep_layer, self).__init__()
        self.query = params['query']
        input_dim = params['rgcn_hidden_dim']
        self.device = torch.device("cuda" if params['gpu'] != -1 else "cpu")

        self.multiheadattention = MultiHeadAttention(input_dim, num_heads=params['att_head_num'], dropout=params['att_dropout'])
        self.multiheadattention1 = MultiHeadAttention(input_dim, num_heads=params['att_head_num'],
                                                     dropout=params['att_dropout'])


    def forward(self, info, section, nodes, global_nodes):

        entities = split_n_pad(entities, section[:, 0]) 
        if self.query == 'global':
            entities = global_nodes

        entity_size = section[:, 0].max()
        mentions = split_n_pad(mentions, section[:, 1])

        mention_sen_rep = F.embedding(info[:, 4], sentences) 
        mention_sen_rep = split_n_pad(mention_sen_rep, section[:, 1])

        eid_ranges = torch.arange(0, max(info[:, 0]) + 1).to(self.device)
        eid_ranges = split_n_pad(eid_ranges, section[:, 0], pad=-2) 


        r_idx, c_idx = torch.meshgrid(torch.arange(entity_size).to(self.device),
                                          torch.arange(entity_size).to(self.device))
        query_1 = entities[:, r_idx]  
        query_2 = entities[:, c_idx]

        info = split_n_pad(info, section[:, 1], pad=-1)
        m_ids, e_ids = torch.broadcast_tensors(info[:, :, 0].unsqueeze(1), eid_ranges.unsqueeze(-1))
        index_m = torch.ne(m_ids, e_ids).to(self.device)  
        index_m_h = index_m.unsqueeze(2).repeat(1, 1, entity_size, 1).reshape(index_m.shape[0], entity_size*entity_size, -1).to(self.device)
        index_m_t = index_m.unsqueeze(1).repeat(1, entity_size, 1, 1).reshape(index_m.shape[0], entity_size*entity_size, -1).to(self.device)

        entitys_pair_rep_h, h_score = self.multiheadattention(mention_sen_rep, mentions, query_2, index_m_h)
        entitys_pair_rep_t, t_score = self.multiheadattention1(mention_sen_rep, mentions, query_1, index_m_t)
        return entitys_pair_rep_h, entitys_pair_rep_t
