import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import numpy as np
import random
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization
import tensorflow as tf
import time 
import tensorflow_addons as tfa 
import tensorflow.keras.backend as K 
import datetime


def extract_first_features(filters, size, strides, apply_batchnorm=True):
  initializer = tf.keras.initializers.he_normal(seed=None)

  result = tf.keras.Sequential()
  result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                            kernel_initializer=initializer, use_bias=False,
                            kernel_regularizer=tf.keras.regularizers.l2(0.001)))

  if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())
      result.add(tfa.layers.InstanceNormalization())

  result.add(tf.keras.layers.ReLU())

  return result

def base_model(left_patch,right_patch):
  layer1 = extract_first_features(32, 3, 1, True)
  layer2 = extract_first_features(64, 3, 1, True)
  layer3 = extract_first_features(128, 3, 1, True)
  layer4 = extract_first_features(128, 5, 2, True)
  layer5 = extract_first_features(256, 3, 1, True)
  layer6 = extract_first_features(256, 5, 2, True)


  left_patch = layer1(left_patch)
  left_patch = layer2(left_patch)
  left_patch = layer3(left_patch)
  left_patch = layer4(left_patch)
  left_patch = layer5(left_patch)
  left_patch = layer6(left_patch)
  left_patch = layers.Flatten()(left_patch)

  # for right_patch
  right_patch = layer1(right_patch)
  right_patch = layer2(right_patch)
  right_patch = layer3(right_patch)
  right_patch = layer4(right_patch)
  right_patch = layer5(right_patch)
  right_patch = layer6(right_patch)
  right_patch = layers.Flatten()(right_patch)

  x = tf.abs(left_patch - right_patch)

  return x


def siamese_network():
  left_patch = layers.Input(shape=[32, 32, 3])
  right_patch = layers.Input(shape=[32, 32, 3])

  # matching
  x = base_model(left_patch, right_patch)
  # metric learning
  x = layers.Dense(1024)(x)
  x = layers.Dense(128)(x)
  #x = layers.Dropout(0.25)(x)
  x = layers.Dense(1)(x)

  model = tf.keras.Model(inputs=[left_patch, right_patch], outputs=[x])
  return model


@tf.function
def train_step(left_patch_positive, right_patch_positive,left_patch_negative,right_patch_negative,epoch):
    with tf.GradientTape() as tape:

        # left_feature_pos = model(left_patch_positive,training=True) # (BATCH_SIZE,32,32,3)
        # right_feature_pos = model(right_patch_positive,training=True) # (BATCH_SIZE,32,32,3)

        # # Negative examples
        # left_feature_neg = model(left_patch_negative,training=True) # (BATCH_SIZE,32,32,3)
        # right_feature_neg = model(right_patch_negative,training=True) # (BATCH_SIZE,32,32,3)
        # # Inner product
        # inner_product_pos = map_inner_product(left_feature_pos,right_feature_pos) # (BATCH_SIZE,1) 
        # inner_product_neg = map_inner_product(left_feature_neg,right_feature_neg) # (BATCH_SIZE,1) 

        pos_output = model([left_patch_positive,right_patch_positive],training=True)
        neg_output = model([left_patch_negative,right_patch_negative],training=True)
        
        loss_pos = cross_entropy(tf.ones_like(pos_output),pos_output)
        loss_neg = cross_entropy(tf.zeros_like(neg_output),neg_output)

        #total_loss = tfa.losses.contrastive_loss(y_true=tf.ones_like(inner_product_pos),y_pred=inner_product_pos,margin = 1.0)#loss_pos + loss_neg
        #total_loss = tf.reduce_sum(total_loss)
        total_loss = loss_pos + loss_neg
        #Gradient descent
    grads = tape.gradient(total_loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    with summary_writer.as_default():
      tf.summary.scalar('total_loss', total_loss, step=epoch)
      tf.summary.scalar('loss_pos', loss_pos, step=epoch)
      tf.summary.scalar('loss_neg', loss_neg, step=epoch)

    return total_loss,loss_pos,loss_neg,pos_output,neg_output
if __name__ == '__main__':
    EPOCHS = 20    
    disp_range = 201 # For kitti 2015
    log_dir = "logs/"
    
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    images = np.load('../generation/images.npy', allow_pickle=True)  

    # Normalize 
    images = (images - images.mean()) / images.std()
    #print(images[0][0]) # Test for normalization
    print("DATASET SHAPE: ",images.shape)
    BUFFER_SIZE = images.shape[0]
    BATCH_SIZE = 256

    train_dataset = tf.data.Dataset.from_tensor_slices(images)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    # DEFINE MODEL 
    model = siamese_network() #base_model((32,32,3))
    model.summary()
    # Create optimizer and checkpoint
    learning_rate = 0.001
    optimizer = optimizers.Adam(learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './checkpoint', max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.01)    

    # Train
    for epoch in range(EPOCHS):
        start = time.time()
        ckpt.step.assign_add(1)   
        print("Epoch: ", epoch+1)

        # Train
        average_loss = 0
        average_posl = 0
        average_negl = 0

        count = 0
        count_ones_pos = 0
        count_ones_neg = 0

        for n, images in train_dataset.enumerate():
            #print(images.shape)
            left_patch_positive,right_patch_positive,left_patch_negative,right_patch_negative = tf.split(images,4,axis=1)
            
            # Remove single channel 
            left_patch_positive = tf.squeeze(left_patch_positive,axis=1)
            right_patch_positive = tf.squeeze(right_patch_positive,axis=1)
            left_patch_negative = tf.squeeze(left_patch_negative,axis=1)
            right_patch_negative = tf.squeeze(right_patch_negative,axis=1)

            #print("PATCH: ",left_patch_positive.shape)
            # print('.', end='')
            # if (n+1) % 100 == 0:
            #     print()
            total_loss,loss_pos,loss_neg,pos_output,neg_output = train_step(left_patch_positive, right_patch_positive,left_patch_negative,right_patch_negative,epoch)
            
            # --------- compute training acc ---------
            bool_pos_output = pos_output > 0
            ones_pos_output = tf.reduce_sum(tf.cast(bool_pos_output, tf.float32))
            count_ones_pos = count_ones_pos + ones_pos_output

            bool_neg_output = neg_output < 0
            ones_neg_output = tf.reduce_sum(tf.cast(bool_neg_output, tf.float32))
            count_ones_neg = count_ones_neg + ones_neg_output
            
            if (n+1) % 100 == 0:
                print("TOTAL LOSS: \t",total_loss.numpy(),"\t LOSS POS: ", loss_pos.numpy(), "\t LOSS NEG",loss_neg.numpy())

            average_loss = average_loss + total_loss
            average_posl = average_posl + loss_pos
            average_negl = average_negl + loss_neg

            count += 1
        
        average_loss = average_loss / count
        average_posl = average_posl / count
        average_negl = average_negl / count

        print('epoch {}  average_loss {}'.format(epoch, average_loss))
        print('normal loss {}  perceptual loss {}  '.format(average_posl, average_negl))

        pos_acc = (count_ones_pos * 100.0) / BUFFER_SIZE
        neg_acc = (count_ones_neg * 100.0) / BUFFER_SIZE
        print('train acc (pos) {} - acc (neg) {}'.format(pos_acc, neg_acc))


        print()

        # saving (checkpoint) the model every 20 epochs
        #if (epoch + 1) % 20 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)
        save_path = manager.save()
        print("\n----------Saved checkpoint for epoch {}: {}-----------\n".format(epoch+1, save_path))
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))

    
