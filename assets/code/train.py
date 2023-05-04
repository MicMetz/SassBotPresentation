# put init, layer definition, and forward statement for models here.
import torch.nn as nn
import torch
from transformers import AutoModel
import torch.nn.functional as F

# attention classifier class as a secondary set of layers.
class AttentionWithContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionWithContext, self).__init__()

        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.contx = nn.Linear(hidden_dim, 1, bias=False)

    # internal tanh and softmax layer for calculations.
    def forward(self, inp):
        u = torch.tanh_(self.attn(inp))
        a = F.softmax(self.contx(u), dim=1)
        s = (a * inp).sum(1)
        return s

# initalizes weights based on the class specified.
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# transformer layer used after embeddings are processed.
class TransformerLayer(nn.Module):
    def __init__(self,both=True,
                 pretrained_path='xlm-roberta-base'):
        super(TransformerLayer, self).__init__()

        self.both = both
        self.transformer = AutoModel.from_pretrained(pretrained_path, output_hidden_states=True)


    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return outputs

    def output_num(self):
        return self.transformer.config.hidden_size

# main attention classifier used by the model, used to adjust focus as data updates.
class ATTClassifier(nn.Module):
    def __init__(self, in_feature, class_num=1, dropout_prob=0.2):
        super(ATTClassifier, self).__init__()
        self.attention = AttentionWithContext(in_feature)

        self.Classifier = nn.Sequential(
            nn.Linear(2 * in_feature, 512),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.Linear(512, class_num)
        )

        self.apply(init_weights)

    def forward(self, x):
        att = self.attention(x[0])

        xx = torch.cat([att, x[1]], 1)

        out = self.Classifier(xx)
        return out
