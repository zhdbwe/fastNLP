
import torch
from torch import nn

from ..models.base_model import BaseModel
from ..modules.decoder.mlp import MLP
from .. import seq_len_to_mask
from ..modules import Embedding
from ..modules import LSTM
from ..modules.decoder.crf import ConditionalRandomField
from ..modules.decoder.crf import allowed_transitions

class _CWSBiLSTMEncoder(BaseModel):
    def __init__(self, char_embed, bigram_embed=None, num_bigram_per_char=None,
                 hidden_size=200, embed_drop_p=0.2, num_layers=1):
        super().__init__()

        self.input_size = 0
        self.num_bigram_per_char = num_bigram_per_char
        self.num_layers = num_layers
        self.hidden_size = hidden_size//2
        self.num_directions = 2

        if not bigram_embed is None:
            assert not bigram_embed is None, "Specify num_bigram_per_char."

        self.char_embedding = Embedding(char_embed, dropout=embed_drop_p)
        self.input_size += self.char_embedding.weight.size(1)

        if bigram_embed is not None:
            self.bigram_embedding = Embedding(bigram_embed, dropout=embed_drop_p)
            self.input_size += self.bigram_embedding.weight.size(1)


        self.lstm = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, bidirectional=True,
                    batch_first=True, num_layers=self.num_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            if 'bias_hh' in name:
                nn.init.constant_(param, 0)
            elif 'bias_ih' in name:
                nn.init.constant_(param, 1)
            else:
                nn.init.xavier_uniform_(param)


    def forward(self, chars, bigrams=None, seq_len=None):

        batch_size, max_len = chars.size()

        x_tensor = self.char_embedding(chars)

        if hasattr(self, 'bigram_embedding'):
            bigram_tensor = self.bigram_embedding(bigrams).view(batch_size, max_len, -1)
            x_tensor = torch.cat([x_tensor, bigram_tensor], dim=2)

        outputs, _ = self.lstm(x_tensor, seq_len)
        return outputs


class CWSBiLSTMCRF(BaseModel):
    """
    BiLSTM+CRF的分词模型，tag为BMES模式。tag与index的对应关系应该为B:0, M:1, E:2, S:3; 提供forward函数用于训练，predict
        函数用于预测。forward()函数返回的dict包含{'loss': torch.Tensor}(Tensorshape为[batch_size,]), predict()函数返回的dict包含
        {'pred': torch.Tensor}(Tensor shape为[batch_size, max_len])

    :param char_embed: 可以是 tuple:(num_embedings, embedding_dim), 即embedding的大小和每个词的维度;也可以传入
        nn.Embedding 对象, 此时就以传入的对象作为embedding; 传入np.ndarray也行，将使用传入的ndarray作为作为Embedding
        初始化; 传入orch.Tensor, 将使用传入的值作为Embedding初始化。
    :param bigram_embed: 可以是 tuple:(num_embedings, embedding_dim), 即embedding的大小和每个词的维度;也可以传入
        nn.Embedding 对象, 此时就以传入的对象作为embedding; 传入np.ndarray也行，将使用传入的ndarray作为作为Embedding
        初始化; 传入orch.Tensor, 将使用传入的值作为Embedding初始化。
    :param int num_bigram_per_char: 每个character处的bigram的数量
    :param int hidden_size: BiLSTM的隐层大小
    :param float embed_drop_p: char_embed和bigram_embed的dropout概率。
    :param int num_layers: BiLSTM的层数
    """
    def __init__(self, char_embed, bigram_embed=None, num_bigram_per_char=None,
                 hidden_size=200, embed_drop_p=0.2, num_layers=1, tag_size=4):
        super(CWSBiLSTMCRF, self).__init__()

        self.tag_size = tag_size

        self.encoder_model = _CWSBiLSTMEncoder(char_embed, bigram_embed, num_bigram_per_char,
                                               hidden_size, embed_drop_p, num_layers)

        size_layer = [hidden_size, 200, tag_size]
        self.decoder_model = MLP(size_layer)
        allowed_trans = allowed_transitions({0:'b', 1:'m', 2:'e', 3:'s'}, encoding_type='bmes')
        self.crf = ConditionalRandomField(num_tags=tag_size, include_start_end_trans=False,
                                          allowed_transitions=allowed_trans)


    def forward(self, chars, target, seq_len, bigrams=None):
        device = self.parameters().__next__().device
        chars = chars.to(device).long()
        if not bigrams is None:
            bigrams = bigrams.to(device).long()
        else:
            bigrams = None
        seq_len = seq_len.to(device).long()
        masks = seq_len_to_mask(seq_len)
        feats = self.encoder_model(chars, bigrams, seq_len)
        feats = self.decoder_model(feats)
        losses = self.crf(feats, target, masks)

        pred_dict = {}
        pred_dict['loss'] = losses

        return pred_dict

    def predict(self, chars, seq_len, bigrams=None):
        """

        :param torch.Tensor chars: batch_size x max_len,
        :param torch.Tensor seq_len: batch_size,
        :param torch.Tensor bigrams: batch_size x *max_len
        :return: {'pred': torch.Tensor}, pred的shape为batch_size x max_len。每个sample都是从该sample的最大长度处开始往回
            解码的。
        """
        device = self.parameters().__next__().device
        chars = chars.to(device).long()
        if not bigrams is None:
            bigrams = bigrams.to(device).long()
        else:
            bigrams = None
        seq_len = seq_len.to(device).long()
        masks = seq_len_to_mask(seq_len)
        feats = self.encoder_model(chars, bigrams, seq_len)
        feats = self.decoder_model(feats)
        paths, _ = self.crf.viterbi_decode(feats, masks)

        return {'pred': paths}

