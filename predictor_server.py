#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import socket

import tensorflow as tf
import numpy as np
import data_helpers
from multi_class_data_loader import MultiClassDataLoader
from word_data_processor import WordDataProcessor
from config import FLAGS
import csv

data_loader = MultiClassDataLoader(tf.flags, WordDataProcessor())
data_loader.define_flags()

# checkpoint_dir이 없다면 가장 최근 dir 추출하여 셋팅
if FLAGS.checkpoint_dir == "":
    all_subdirs = ["./runs/" + d for d in os.listdir('./runs/.') if os.path.isdir("./runs/" + d)]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    FLAGS.checkpoint_dir = latest_subdir + "/checkpoints/"


# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = data_loader.restore_vocab_processor(vocab_path)

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Make Server
        sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '127.0.0.1'
        port = int(3000)
        sock.bind((host, port))
        sock.listen(1)
        print("text_predictor Start...\n")
        while True:
            connection, client_addr = sock.accept()

            # print(connection, client_addr)
            data = connection.recv(1024)
            data = data.decode("utf-8")
            print("data > "+data)
            arr = []
            arr.append(data.strip())
            # print(arr)
            x_test = np.array(list(vocab_processor.transform(arr)))
            # print(x_test)
            pred = sess.run(predictions, {input_x: x_test,dropout_keep_prob: 1.0})
            # print(pred)
            answer = np.concatenate([pred])
            # print(answer)
            answer = data_loader.class_labels(answer.astype(int))
            print("answ > "+answer[0])
            connection.sendall(answer[0].encode("utf-8"))
            connection.close()
