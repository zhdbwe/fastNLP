import unittest

from fastNLP.api.basic_chinese_nlp import ChinesePOS
from fastNLP.api.basic_chinese_nlp import CWS

class TestPOS(unittest.TestCase):
    def test_usage(self):
        # pos = ChinesePOS()
        # print(pos.predict(['新华社', '北京', '十二月', '二十九', '电']))
        # print(pos.predict([
        #                     ['新华社', '北京', '十二月', '二十九', '电'],
        #                     ['唐家璇', '举行', '新年', '招待会'],
        #                     ['Trump', '将', '华为', '列入', '了', '实体', '清单', '。']
        #                   ]))
        pass


class TestCWS(unittest.TestCase):
    def test_usage(self):
        # cws = CWS()
        # s1 = ''.join(['新华社', '北京', '十二月', '二十九', '电'])
        # s2 = ''.join(['唐家璇', '举行', '新年', '招待会'])
        # print(s1, s2)
        # print(cws.predict(s1))
        # print(cws.predict([
        #     s1,
        #     s2,
        #     '特朗普总统将华为列入了实体清单。',
        #     '中美贸易跌入冰点。'
        # ]))
        pass


if __name__ == '__main__':
    TestPOS().test_usage()
    TestCWS().test_usage()