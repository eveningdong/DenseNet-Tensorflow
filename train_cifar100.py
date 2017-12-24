import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import time

from config import *
from datetime import datetime
from densenet_v1 import densenet
from libs.datasets.cifar import cifar100_input as cifar100

slim = tf.contrib.slim

def save(saver, sess, logdir, step):
  '''Save weights. 
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    logdir: path to the snapshots directory.
    step: current training step.
  '''
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
  
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  saver.save(sess, checkpoint_path, global_step=step)
  print('The checkpoint has been created.')

def load(saver, sess, ckpt_dir):
  '''Load trained weights.
  
  Args:
    saver: TensorFlow Saver object.
    sess: TensorFlow session.
    ckpt_path: path to checkpoint file with parameters.
  ''' 
  if args.ckpt == 0:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    ckpt_path = ckpt.model_checkpoint_path
  else:
    ckpt_path = ckpt_dir+'/model.ckpt-%i' % args.ckpt
  saver.restore(sess, ckpt_path)
  print("Restored model parameters from {}".format(ckpt_path))

def main():
  tf.set_random_seed(args.random_seed)
  coord = tf.train.Coordinator()
  # Train input
  images, labels = cifar100.distorted_inputs(data_dir=args.data_dir,
                                            batch_size=args.batch_size)
  
  # Prepare parameters for DenseNet
  assert (args.num_layers -4) % 3 == 0, 'The number of layers is wrong'
  num_units = (args.num_layers - 4) // 3
  blocks = [num_units, num_units, num_units]
  
  # Training
  net, end_points = densenet(images, 
                             blocks=blocks, 
                             growth=args.growth_rate, 
                             drop=0.2,
                             num_classes=10,
                             scope='densenet_L{}_k{}'.format(args.num_layers,
                               args.growth_rate))
  
  logits = tf.squeeze(net, [1,2])
  ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
    logits=logits)
  cls_loss = tf.reduce_mean(ce)
  cls_loss_sum = tf.summary.scalar('loss/cls', cls_loss)

  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  reg_loss = tf.add_n(reg_losses)
  reg_loss_sum = tf.summary.scalar('loss/reg', reg_loss)

  tot_loss = cls_loss + reg_loss
  tot_loss_sum = tf.summary.scalar('loss/tot', tot_loss)
  
  preds = tf.argmax(logits, axis=1)
  train_acc, train_acc_update_op = tf.metrics.accuracy(labels=labels, 
    predictions=preds, name='train_acc')
  train_acc_sum = tf.summary.scalar('acc/train_acc', train_acc)

  train_initializer = tf.variables_initializer(var_list=tf.get_collection(
    tf.GraphKeys.LOCAL_VARIABLES, scope="train_acc"))

  restore_var = [v for v in tf.global_variables() if 'fc' not in v.name 
    or not args.not_restore_last]
  if args.freeze_bn:
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in 
      v.name and 'gamma' not in v.name]
  else:
    all_trainable = [v for v in tf.trainable_variables()]
  conv_trainable = [v for v in all_trainable if 'fc' not in v.name]

  # Validation
  images_val, labels_val = cifar100.inputs(data_dir=args.data_dir,
                                          batch_size=100,
                                          is_training=False)

  net_val, _ = densenet(images_val, 
                        blocks=blocks, 
                        growth=args.growth_rate,
                        num_classes=100, 
                        is_training=False,  
                        reuse=True, 
                        scope='densenet_L{}_k{}'.format(args.num_layers,
                          args.growth_rate))

  logits_val = tf.squeeze(net_val, [1,2])

  preds_val = tf.argmax(logits_val, axis=1)
  val_acc, val_acc_update_op = tf.metrics.accuracy(labels=labels_val, 
    predictions=preds_val, name='val_acc')
  val_acc_sum = tf.summary.scalar('acc/val_acc', val_acc)

  val_initializer = tf.variables_initializer(var_list=tf.get_collection(
    tf.GraphKeys.LOCAL_VARIABLES, scope="val_acc"))
  test_sum_op = tf.summary.merge([val_acc_sum])
  
  # Optimization
  global_step = tf.train.get_or_create_global_step()
  learning_rate = args.learning_rate
  learning_rates = [learning_rate, learning_rate*0.1, learning_rate*0.01]
  num_epochs = 300
  num_steps = num_epochs * cifar100.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // \
    args.batch_size
  steps = [int(num_steps * 0.5), int(num_steps * 0.75)]
  learning_rate = tf.train.piecewise_constant(tf.to_int32(global_step),
                                              steps, learning_rates)
  lr_sum = tf.summary.scalar('params/learning_rate', learning_rate)
  train_sum_op = tf.summary.merge([cls_loss_sum, reg_loss_sum, 
    tot_loss_sum, train_acc_sum, lr_sum])

  opt = tf.train.MomentumOptimizer(learning_rate, args.momentum) 
  grads_conv = tf.gradients(tot_loss, conv_trainable)
  train_op = slim.learning.create_train_op(
    tot_loss, opt,
    global_step=global_step,
    variables_to_train=conv_trainable,
    summarize_gradients=True)
  
  # Set up tf session and initialize variables. 
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  # Saver for storing checkpoints of the model.
  saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=20)
  
  # Load variables if the checkpoint is provided.
  if args.ckpt > 0 or args.restore_from is not None:
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.snapshot_dir)
  
  # Start queue threads.
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)
  
  # tf.get_default_graph().finalize()
  summary_writer = tf.summary.FileWriter(args.snapshot_dir,
                                         sess.graph)
  
  # Iterate over training steps.
  for step in range(args.ckpt, args.num_steps):
    start_time = time.time()
    tot_loss_float, cls_loss_float, reg_loss_float, _, lr_float, _,train_summary = sess.run([tot_loss, cls_loss, reg_loss, train_op,
      learning_rate, train_acc_update_op, train_sum_op])
    train_acc_float = sess.run(train_acc)
    duration = time.time() - start_time
    sys.stdout.write('step {:d}, tot_loss = {:.6f}, cls_loss = {:.6f}, ' \
      'reg_loss = {:.6f}, acc = {:.6f}, lr: {:.6f}({:.3f}sec/step)\n'.format(
      step, tot_loss_float, cls_loss_float, reg_loss_float, train_acc_float, 
      lr_float, duration))
    sys.stdout.flush()

    if step % args.save_pred_every == 0 and step > args.ckpt:
      summary_writer.add_summary(train_summary, step)
      sess.run(val_initializer)
      # for val_step in range(cifar100.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL//100):
      for val_step in range(5):
        _, test_summary = sess.run([val_acc_update_op, test_sum_op])
        
      summary_writer.add_summary(test_summary, step)
      val_acc_float= sess.run(val_acc)

      save(saver, sess, args.snapshot_dir, step)
      sys.stdout.write('step {:d}, train_acc: {:.6f}, val_acc: {:.6f}\n'.format(step, train_acc_float, val_acc_float))
      sys.stdout.flush()
      sess.run(train_initializer)

    if coord.should_stop():
      coord.request_stop()
      coord.join(threads)
    
if __name__ == '__main__':
  main()