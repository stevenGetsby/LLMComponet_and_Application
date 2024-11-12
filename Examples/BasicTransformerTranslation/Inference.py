import torch
import numpy as np
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Examples.BasicTransformerTranslation.dataset import PrepareData
from Transformers.model import make_model
from Transformers.utils import greedy_decode


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(data, model):
    # 梯度清零
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for i in range(5):
            # 打印待翻译的英文句子
            en_sent = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            print("\n" + en_sent)
            # 打印对应的中文句子答案
            cn_sent = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])
            print("".join(cn_sent))

            # 将当前以单词id表示的英文句子数据转为tensor，并放如DEVICE中
            src = torch.from_numpy(np.array(data.dev_en[i])).long().to(DEVICE)
            # 增加一维
            src = src.unsqueeze(0)
            # 设置attention mask
            src_mask = (src != 0).unsqueeze(-2)
            # 用训练好的模型进行decode预测
            out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=data.cn_word_dict["BOS"])
            # 初始化一个用于存放模型翻译结果句子单词的列表
            translation = []
            # 遍历翻译输出字符的下标（注意：开始符"BOS"的索引0不遍历）
            for j in range(1, out.size(1)):
                # 获取当前下标的输出字符
                sym = data.cn_index_dict[out[0, j].item()]
                # 如果输出字符不为'EOS'终止符，则添加到当前句子的翻译结果列表
                if sym != 'EOS':
                    translation.append(sym)
                # 否则终止遍历
                else:
                    break
            # 打印模型翻译输出的中文句子结果
            print("translation: %s" % " ".join(translation))

model = make_model(
                    src_vocab = 5493,
                    tgt_vocab = 3194,
                    N = 6,
                    d_model = 256,
                    h = 8,
                    d_ff = 1024,
                    dropout = 0.1,
                )
model.load_state_dict(torch.load('Examples/BasicTransformerTranslation/model.pth'))

TRAIN_FILE = 'Examples/BasicTransformerTranslation/nmt/en-cn/train.txt'  # 训练集数据文件
DEV_FILE = "Examples/BasicTransformerTranslation/nmt/en-cn/dev.txt"  # 验证(开发)集数据文件
data = PrepareData(TRAIN_FILE, DEV_FILE)

print(">>>>>>> start evaluate")
evaluate_start  = time.time()
evaluate(data, model)
print(f"<<<<<<< finished evaluate, cost {time.time()-evaluate_start:.4f} seconds")
print(1)