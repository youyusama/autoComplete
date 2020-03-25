#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>
import sys
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\src')
import json
import os
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

class objgpt2:
    enc = encoder.get_encoder('117M')
    hparams = model.default_hparams()
    with open(os.path.join(r'D:\CNM\NTeat\autoCom\autoComapp\src\models', '117M', 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if 3 > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

    sess = tf.Session(config=config)
    context = tf.placeholder(tf.int32, [1, None])

    tf_sample = sample.sample_sequence(
        hparams=hparams,
        length=3,
        context=context,
        batch_size=1,
        temperature=1.0,
        top_k=40,
        top_p=0.0)

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]

    saver = tf.train.Saver(
        var_list=all_vars,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=2)
    sess.run(tf.global_variables_initializer())

    # Get fresh GPT weights if new run.
    ckpt = tf.train.latest_checkpoint(
        os.path.join(r'D:\CNM\NTeat\autoCom\autoComapp\src\models', '117M'))
    saver.restore(sess, ckpt)


    # def __init__(self):

    def doautocom(self,words):
        context_tokens = self.enc.encode(words)
        # print(context_tokens)
        all_text = []
        index = 0
        while index < 5:
            out = self.sess.run(
                self.tf_sample,
                feed_dict={self.context: 1 * [context_tokens]})
            for i in range(min(5 - index, 1)):
                text = self.enc.decode(out[i])
                text=text.replace('\n', ' ')[len(words):-1]
                all_text.append(text)
            index += 1
        print(all_text)
        return all_text


if __name__ == '__main__':
    obj=objgpt2()
    obj.doautocom('ATP shall')
