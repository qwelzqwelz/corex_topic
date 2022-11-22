from corextopic import corextopic as ct
from corextopic import vis_topic as vt

import numpy as np
import os

DATA_ROOT = r"/var/www/alhs-dw-data"


def _read_data_file(name: str, path=None, int_csv=False):
    path = path or os.path.join(DATA_ROOT, name)
    with open(path, "rt", encoding="utf8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            if int_csv:
                line = [int(x.strip()) for x in line.split(",") if x.strip()]
            yield line


def train(epoch=200):
    anchors = _read_data_file("anchors.csv", int_csv=True)
    # 读数据
    print("[start] read X")
    X = []
    for line in _read_data_file("doc-word-matrix.csv"):
        X.append([int(x) for x in line.split(",")])
    X = np.array(X, dtype=int)
    print(f"[end] read X, shape={X.shape}")
    # 构建模型
    topic_model = ct.Corex(n_hidden=len(anchors), verbose=True)
    topic_model.fit(X, anchors=anchors, anchor_strength=10)
    # 输出信息
    vt.vis_rep(topic_model, column_label=WORDS, prefix='topic-model-example')


if __name__ == '__main__':
    # nltk, networkx
    train(epoch=200)
