
__all__ = ['CWS', 'ChinesePOS']

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from itertools import chain

import torch

from ..core.dataset import DataSet
from .utils import _load_url
from ..models.cws_bilstm_crf import CWSBiLSTMCRF
from ..models.sequence_labeling import AdvSeqLabel
from ..core.vocabulary import Vocabulary
from ..core.const import Const
from ..core.batch import Batch
from ..core.sampler import SequentialSampler
from ..core.utils import _move_dict_value_to_device
from ..core.utils import _get_model_device
from ..core.utils import _build_args

# TODO add pretrain urls
model_urls = {
    "cws": " http://123.206.98.91:8888/download/cws_20190523-d72b766f.pkl",
    "chinese_pos": " http://123.206.98.91:8888/download/chinese_pos_tag_20190523-f7a55486.pkl",
    "parser": "http://123.206.98.91:8888/download/parser_20190204-c72ca5c0.pkl"
}



class API:
    def __init__(self):
        self.pipeline = None
        self._dict = None
        self.model = None
        self.vocab = None

    def predict(self, *args, **kwargs):
        """Do prediction for the given input.
        """
        raise NotImplementedError

    def test(self, file_path):
        """Test performance over the given data set.

        :param str file_path:
        :return: a dictionary of metric values
        """
        raise NotImplementedError

    def load(self, path, device):
        if os.path.exists(os.path.expanduser(path)):
            _dict = torch.load(path, map_location=device)
        else:
            _dict = _load_url(path, map_location=device)
        self._dict = _dict

        # self.pipeline = _dict['pipeline']
        # for processor in self.pipeline.pipeline:
        #     if isinstance(processor, ModelProcessor):
        #         processor.set_model_device(device)


class ChinesePOS(API):
    """初始化后可用于直接进行词性标注的类，会自动下载fastNLP预训练的模型。该模型采用fastNLP中AdvSeqLabel作为底层模型。
        预测请使用predict()方法

    :param str model_path: 从哪里读取预训练的权重，默认为None，fastNLP会尝试自动下载。
    :param str device: 在哪个设备进行推断。与pytorch对device设定一致
    """

    def __init__(self, model_path=None, device='cpu'):
        super(ChinesePOS, self).__init__()
        if model_path is None:
            model_path = model_urls['chinese_pos']
        self.vocab = None
        self.load(model_path, device)
        self._prepare()

    def _prepare(self):
        vocab = Vocabulary()
        vocab_label = Vocabulary()
        vocab.__dict__ = self._dict["vocab"]
        vocab_label.__dict__ = self._dict["vocab_label"]
        model = AdvSeqLabel(init_embed=np.zeros([len(vocab), 200]), hidden_size=200,
                            num_classes=len(vocab_label))
        model.load_state_dict(self._dict["model"])

        self.model = model
        self.vocab = vocab
        self.vocab_label = vocab_label

    def _process(self, dataset):
        dataset.apply_field(lambda x: len(x), field_name=Const.INPUT, new_field_name=Const.INPUT_LEN)
        self.vocab.index_dataset(dataset, field_name=Const.INPUT)
        dataset.set_input(Const.INPUT, Const.INPUT_LEN)

        data_iterator = Batch(dataset, 16, sampler=SequentialSampler())
        model_device = _get_model_device(self.model)
        with torch.no_grad():
            unpad_preds = []
            for batch_x, batch_y in data_iterator:
                _move_dict_value_to_device(batch_x, batch_y, device=model_device)
                if hasattr(self.model, 'predict'):
                    x = _build_args(self.model.predict, **batch_x)
                    pred_dict = self.model.predict(**x)
                else:
                    x = _build_args(self.model.forward, **batch_x)
                    pred_dict = self.model.predict(**x)
                seq_len = batch_x[Const.INPUT_LEN]
                preds = pred_dict['pred'].tolist()
                for _len, pred in zip(seq_len, preds):
                    unpad_preds.append(pred[:_len])

        output_tag = [[self.vocab_label.to_word(tag_idx) for tag_idx in tag_idxes ] for tag_idxes in unpad_preds]

        return output_tag

    def predict(self, content):
        """
        传入需要词性标注的数据，返回对应的结果。支持输入为

            1. list of str. 返回list。list中的元素的tuple，第一个为当前词语，第二个为该词语的词性

                Example::

                    pos = ChinesePOS()
                    sentence = ['新华社', '北京', '十二月', '二十九', '电']
                    print(pos.predict(sentence))
                    # [('新华社', 'NR'), ('北京', 'NR'), ('十二月', 'NT'), ('二十九', 'NT'), ('电', 'NN')]

            2. list of list str, 返回是嵌套的list。

                Example::

                    pos = ChinesePOS()
                    sentences = [
                                    ['新华社', '北京', '十二月', '二十九', '电'],
                                    ['唐家璇', '举行', '新年', '招待会']
                                ]
                    # [
                         [('新华社', 'NR'), ('北京', 'NR'), ('十二月', 'NT'), ('二十九', 'CD'), ('电', 'NN')],
                         [('唐家璇', 'NR'), ('举行', 'VV'), ('新年', 'NT'), ('招待会', 'NN')]
                      ]

        :param content: list of str或者list of list of str.
        :return : 如函数介绍中所示
        """

        # 1. 检查sentence的类型
        assert len(content)!=0, "Empty input is not allowed."
        if isinstance(content, list):
            ele_type = set(map(type, content))
            if len(ele_type)!=1:
                raise TypeError("Mixed type of elements encountered.")
            if isinstance(content[0], str):
                sents = [content]
            elif isinstance(content[0], list):
                ele_type = set(chain(*[[type(ele) for ele in content_i] for content_i in content]))
                if len(ele_type)!=1:
                    raise TypeError("Mixed type of elements encountered.")
                if isinstance(content[0][0], str):
                    for sent in content:
                        assert len(sent), "One of your input has no words."
                    sents = content
                else:
                    raise TypeError("Not support `{}` type of element.".format(ele_type.pop()))
            else:
                raise TypeError("Not support `{}` type of element.".format(ele_type.pop()))
        else:
            raise TypeError("Not support `{}` type of input.".format(type(content)))


        # 2. 组建dataset
        dataset = DataSet()
        dataset.add_field(Const.INPUT, sents)

        # 3. 使用pipeline
        tags_list = self._process(dataset)

        def merge_tag(sents, tags_list):
            rtn = []
            for words, tags in zip(sents, tags_list):
                rtn.append([(word, tag) for word, tag in zip(words, tags)])
            return rtn

        output = merge_tag(sents, tags_list)

        return output[0] if isinstance(content[0], str) else output


class CWS(API):
    """初始化后可用于直接进行中文分词的类，会自动下载fastNLP预训练的模型。该模型采用fastNLP中CWSBiLSTMCRF作为底层模型。
        预测请使用predict()方法。

    :param str model_path: 从哪里读取预训练的权重，默认为None，fastNLP会尝试自动下载。
    :param str device: 在哪个设备进行推断。与pytorch对device设定一致
    """

    def __init__(self, model_path=None, device='cpu'):

        super(CWS, self).__init__()
        if model_path is None:
            model_path = model_urls['cws']
        self.vocab_bi = None
        self.word2id = Vocabulary(unknown=None, padding=None)
        self.word2id.add_word_lst(['B'] * 4 + ['M'] * 3 + ['E'] * 2 + ['S'] * 1)
        self.load(model_path, device)
        self._prepare()

    def _prepare(self):
        vocab_uni = Vocabulary()
        vocab_bi = Vocabulary()
        vocab_uni.__dict__ = self._dict["vocab_uni"]
        vocab_bi.__dict__ = self._dict["vocab_bi"]
        model = CWSBiLSTMCRF(char_embed=np.zeros([len(vocab_uni), 100]),
                             bigram_embed=np.zeros([len(vocab_bi), 100]),
                             num_bigram_per_char=1)
        model.load_state_dict(self._dict["model"])

        self.model = model
        self.vocab = vocab_uni
        self.vocab_bi = vocab_bi

    def _process(self, dataset):
        dataset.apply(CWS._cws_tokenize_uni, new_field_name='raw_words')
        dataset.apply(CWS._cws_tokenize_bi, new_field_name='raw_words_bi')
        dataset.apply_field(lambda x: len(x), field_name='raw_words', new_field_name='seq_len')
        self.vocab.index_dataset(dataset, field_name='raw_words', new_field_name='chars')
        self.vocab_bi.index_dataset(dataset, field_name='raw_words_bi', new_field_name='bigrams')
        dataset.set_input(Const.CHAR_INPUT, "bigrams", Const.INPUT_LEN)

        data_iterator = Batch(dataset, 16, sampler=SequentialSampler())
        model_device = _get_model_device(self.model)
        with torch.no_grad():
            unpad_preds = []
            for batch_x, batch_y in data_iterator:
                _move_dict_value_to_device(batch_x, batch_y, device=model_device)
                if hasattr(self.model, 'predict'):
                    x = _build_args(self.model.predict, **batch_x)
                    pred_dict = self.model.predict(**x)
                else:
                    x = _build_args(self.model.forward, **batch_x)
                    pred_dict = self.model.predict(**x)
                seq_len = batch_x[Const.INPUT_LEN]
                preds = pred_dict['pred'].tolist()
                for _len, pred in zip(seq_len, preds):
                    unpad_preds.append(pred[:_len])
        return unpad_preds

    @staticmethod
    def _cws_tokenize_uni(x):
        raw_sentence_list = x["raw_words"]
        res = []
        for word in raw_sentence_list:
            res.append(word)
        return res

    @staticmethod
    def _cws_tokenize_bi(x):
        raw_sentence_list = x["raw_words"]
        res = []
        for i in range(len(raw_sentence_list) - 1):
            res.append(raw_sentence_list[i] + raw_sentence_list[i + 1])
        res.append(raw_sentence_list[-1] + "<eos>")
        return res

    def _cws_create_tags(self, x):
        raw_sentence_list = x["raw_words"]
        res = []
        for phrase in raw_sentence_list:
            if len(phrase) == 1:
                res.append(self.word2id["S"])
                continue
            for word_idx in range(len(phrase) - 1):
                if word_idx == 0:
                    res.append(self.word2id["B"])
                else:
                    res.append(self.word2id["M"])
            res.append(self.word2id["E"])
        return res

    def predict(self, content):
        """
        传入需要分词的数据，返回对应的结果。支持输入为

            1. str. 返回为list，元素为一个词。

                Example::

                    cws = CWS()
                    sentence = '新华社北京十二月二十九电'
                    print(cws.predict(sentence))
                    # ['新华社', '北京', '十二月', '二十九', '电']

            2. list of str, 返回是list of list, 里层list中每个元素为一个词。

                Example::

                    cws = CWS()
                    sentence = [
                                '特朗普总统将华为列入了实体清单。',
                                '中美贸易跌入冰点。'
                                ]
                    cws.predict(sentence)
                    # [
                        ['特朗普', '总统', '将', '华为', '列入', '了', '实体', '清单', '。'],
                        ['中', '美', '贸易', '跌入', '冰点', '。']
                      ]

        :param content: list of str或者list of list of str.
        :return : 如函数介绍中所示
        """
        assert len(content) != 0, "Empty input is not allowed."
        if isinstance(content, str):
            sents = [content]
        elif isinstance(content, list):
            ele_type = set(map(type, content))
            assert len(ele_type)==1, "Only list of str supported, mixed input."
            assert isinstance(content[0], str), "Only str supported, not `{}`.".format(type(content[0]))
            for sent in content:
                assert len(sent)!=0, "Only of your input has no char."
            sents = content
        else:
            raise TypeError('Unsupported type.')

        dataset = DataSet()
        dataset.add_field('raw_words', sents)

        def convert_to_output(sents, tag_list):
            output = []
            for chars, tags in zip(sents, tag_list):
                tmp = []
                words = []
                for char, tag in zip(chars, tags):
                    tmp.append(char)
                    if tag==3 or tag==2:
                        words.append(''.join(tmp))
                        tmp = []
                if len(tmp)!=0:
                    words.append(''.join(tmp))
                output.append(words)
            return output

        # 3. process data
        tag_list = self._process(dataset)
        output = convert_to_output(sents, tag_list)
        return output[0] if isinstance(content, str) else output