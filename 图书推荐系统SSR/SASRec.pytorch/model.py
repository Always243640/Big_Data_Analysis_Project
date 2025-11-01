import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args, metadata=None):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        metadata = metadata or {}

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        author_count = metadata.get('author_count', 0)
        item_author_ids = metadata.get('item_author_ids')
        if author_count > 1 and item_author_ids:
            self.author_emb = torch.nn.Embedding(author_count, args.hidden_units, padding_idx=0)
            self.register_buffer('item_author_map', torch.LongTensor(item_author_ids))
        else:
            self.author_emb = None
            self.item_author_map = None

        category_count = metadata.get('category_count', 0)
        item_category_ids = metadata.get('item_category_ids')
        if category_count > 1 and item_category_ids:
            self.category_emb = torch.nn.Embedding(category_count, args.hidden_units, padding_idx=0)
            self.register_buffer('item_category_map', torch.LongTensor(item_category_ids))
        else:
            self.category_emb = None
            self.item_category_map = None

        dept_count = metadata.get('dept_count', 0)
        user_dept_ids = metadata.get('user_dept_ids')
        if dept_count > 1 and user_dept_ids:
            self.dept_emb = torch.nn.Embedding(dept_count, args.hidden_units, padding_idx=0)
            self.register_buffer('user_dept_map', torch.LongTensor(user_dept_ids))
        else:
            self.dept_emb = None
            self.user_dept_map = None

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, user_ids, log_seqs):
        item_tensor = torch.LongTensor(log_seqs).to(self.dev)
        seqs = self.item_emb(item_tensor)
        if self.author_emb is not None and self.item_author_map is not None:
            author_ids = self.item_author_map[item_tensor]
            seqs = seqs + self.author_emb(author_ids)
        if self.category_emb is not None and self.item_category_map is not None:
            category_ids = self.item_category_map[item_tensor]
            seqs = seqs + self.category_emb(category_ids)
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        if self.dept_emb is not None and self.user_dept_map is not None and user_ids is not None:
            user_tensor = torch.LongTensor(user_ids).to(self.dev)
            dept_ids = self.user_dept_map[user_tensor]
            seqs = seqs + self.dept_emb(dept_ids).unsqueeze(1)

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs): # for training
        log_feats = self.log2feats(user_ids, log_seqs) # user_ids hasn't been used yet

        pos_tensor = torch.LongTensor(pos_seqs).to(self.dev)
        pos_embs = self.item_emb(pos_tensor)
        if self.author_emb is not None and self.item_author_map is not None:
            pos_embs = pos_embs + self.author_emb(self.item_author_map[pos_tensor])
        if self.category_emb is not None and self.item_category_map is not None:
            pos_embs = pos_embs + self.category_emb(self.item_category_map[pos_tensor])

        neg_tensor = torch.LongTensor(neg_seqs).to(self.dev)
        neg_embs = self.item_emb(neg_tensor)
        if self.author_emb is not None and self.item_author_map is not None:
            neg_embs = neg_embs + self.author_emb(self.item_author_map[neg_tensor])
        if self.category_emb is not None and self.item_category_map is not None:
            neg_embs = neg_embs + self.category_emb(self.item_category_map[neg_tensor])

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices): # for inference
        log_feats = self.log2feats(user_ids, log_seqs) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_tensor = torch.LongTensor(item_indices).to(self.dev)
        item_embs = self.item_emb(item_tensor) # (U, I, C)
        if self.author_emb is not None and self.item_author_map is not None:
            item_embs = item_embs + self.author_emb(self.item_author_map[item_tensor])
        if self.category_emb is not None and self.item_category_map is not None:
            item_embs = item_embs + self.category_emb(self.item_category_map[item_tensor])

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
