import os
import sys

import numpy as np
import tensorflow as tf

from dataloader import Gen_Data_loader, Dis_dataloader, F_Dis_dataloader, F_Gen_Data_loader, AB_Dis_dataloader
from discriminator import Discriminator
from generator import Generator
from midi_io import MIDI_IO
from rollout import ROLLOUT

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32  # embedding dimension
HIDDEN_DIM = 32  # hidden state dimension of lstm cell
SEQ_LENGTH = 36  # sequence length
START_TOKEN = 0
SEED = 88
BATCH_SIZE = 64
vocab_size = 100

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_batch_size = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

dis_dropout_keep_prob = 0.5
dis_l2_reg_lambda = 0.001
dis_wgan_reg_lambda = 1.0
dis_ce_lambda = 0.0
dis_grad_clip = 1.0

if len(sys.argv) > 1:
    dis_l2_reg_lambda, dis_wgan_reg_lambda, dis_ce_lambda, dis_grad_clip = map(float, sys.argv[1:])

#########################################################################################
#  Basic Training Parameters
#########################################################################################
MAIN_EPOCH = 5
PRE_GEN_EPOCH = MAIN_EPOCH * 5
PRE_DIS_EPOCH = MAIN_EPOCH * 1
GAN_OUTER_EPOCH = MAIN_EPOCH
GAN_G_EPOCH = 20
GAN_D_EPOCH = 1
EXCHANGE_EPOCH = MAIN_EPOCH
FUSION_EPOCH = 10
FUSION_F_EPOCH = MAIN_EPOCH
FUSION_AB_EPOCH = MAIN_EPOCH
FUSION_G_EPOCH = 10
FUSION_D_EPOCH = 1
generated_num = 10000
gen_len = 500

#########################################################################################
#  Tensorboard parameters
#########################################################################################
tb_d_id = 0
tb_g_id = 0
print_inter = 100
logdir = '/home/danny/PycharmProjects/hybridmusic/save/l2={},w={},e={},clip={}'.format(dis_l2_reg_lambda,
                                                                                       dis_wgan_reg_lambda,
                                                                                       dis_ce_lambda, dis_grad_clip)
suffix = '_{}_{}_{}_{}'.format(dis_l2_reg_lambda, dis_wgan_reg_lambda, dis_ce_lambda, dis_grad_clip)
print logdir


def rout(*args):
    refresh = True
    to_print = ' '.join(map(str, args))
    if refresh:
        sys.stdout.write('%s\r' % to_print)
        sys.stdout.flush()
    else:
        print(to_print)


def init_var(file):
    positive_file = file
    negative_file = positive_file.split('.')[0] + '_tmp'
    output_music_gan = positive_file.split('.')[0] + '_gan.out'
    output_music_mle = positive_file.split('.')[0] + '_mle.out'

    return positive_file, negative_file, output_music_gan, output_music_mle


def init_data_loader(positive_file):
    dis_data_loader = Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)
    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    gen_data_loader.create_batches(positive_file)
    return gen_data_loader, dis_data_loader


def gen(domain):
    return Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN, domain_name=domain)


def dis(domain, num_class=2):
    return Discriminator(sequence_length=SEQ_LENGTH, num_classes=num_class, vocab_size=vocab_size,
                         embedding_size=dis_embedding_dim,
                         filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                         l2_reg_lambda=dis_l2_reg_lambda, wgan_reg_lambda=dis_wgan_reg_lambda,
                         ce_lambda=dis_ce_lambda, grad_clip=dis_grad_clip,
                         domain_name=domain)


def pre_train_epoch(sess, generator, data_loader, tb_write, pre_epochs):
    for epoch in xrange(pre_epochs):
        print 'Pre-Training {} Gen at epoch {}, g_count is {}'.format(generator.domain_name, epoch, generator.g_count)
        # Pre-train the generator using MLE for one epoch
        # supervised_g_losses = []
        data_loader.reset_pointer()

        for it in xrange(data_loader.num_batch):
            batch = data_loader.next_batch()
            _, g_loss = generator.pretrain_step(sess, batch)
            # supervised_g_losses.append(g_loss)

            # return np.mean(supervised_g_losses)
            if generator.g_count % print_inter == 0:
                rout('Printing tensorboard', generator.g_count)
                _sid, _summ = generator.generate_pretrain_summary(sess, batch)
                tb_write.add_summary(_summ, _sid)
            generator.g_count += 1


def gan_g(sess, generator, discriminator, rollout, tb_writer, g_epochs):
    for it in range(g_epochs):
        print 'GAN Training {} Gen at epoch {}, g_count is {}'.format(generator.domain_name, it, generator.g_count)
        samples = generator.generate(sess)
        rewards = rollout.get_reward(sess, samples, 16, discriminator)
        feed = {generator.x: samples, generator.rewards: rewards}
        print 'rewards sum is {}'.format(np.sum(abs(rewards)))  # , np.array(rewards).shape,
        _ = sess.run(generator.g_updates, feed_dict=feed)

        if generator.g_count % print_inter == 0:
            rout('Printing tensorboard', generator.g_count)
            _sid, _summ = generator.generate_gan_summary(sess, samples, rewards)
            tb_writer.add_summary(_summ, _sid)
        generator.g_count += 1


def fusion_g(sess, generator, d_a, d_b, d_f, rollout, tb_writer, g_epochs):
    for it in range(g_epochs):
        print 'Fusion Training {} Gen at epoch {}, g_count is {}'.format(generator.domain_name, it, generator.g_count)
        samples = generator.generate(sess)
        rewards_a = rollout.get_reward(sess, samples, 16, d_a)
        rewards_b = rollout.get_reward(sess, samples, 16, d_b)
        rewards_f = rollout.get_reward(sess, samples, 16, d_f)
        print 'rewards', rewards_a.shape, np.sum(rewards_a), np.sum(rewards_b), np.sum(rewards_f)
        # considering lambdas
        rewards = rewards_a + rewards_b + rewards_f
        feed = {generator.x: samples, generator.rewards: rewards}
        _ = sess.run(generator.g_updates, feed_dict=feed)

        if generator.g_count % print_inter != 0:
            rout('Printing tensorboard', generator.g_count)
            _sid, _summ = generator.generate_gan_summary(sess, samples, rewards)
            tb_writer.add_summary(_summ, _sid)
        generator.g_count += 1


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for __ in generated_samples:
            buffer = ' '.join([str(x) for x in __]) + '\n'
            fout.write(buffer)


def train_d(sess, dis_data_loader, positive_file, negative_file, generator, discriminator, tb_write, d_epoch):
    for _ in range(d_epoch):
        print 'Training {} Dis at epoch {}, d_count is {}'.format(discriminator.domain_name, _, discriminator.d_count)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        for __ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                loss = discriminator.train(sess, x_batch, y_batch, dis_dropout_keep_prob)
                if discriminator.d_count % print_inter == 0:
                    rout('Printing tensorboard', discriminator.d_count)
                    _sid, _summ = discriminator.generate_summary(sess, x_batch, y_batch, dis_dropout_keep_prob)
                    tb_write.add_summary(_summ, _sid)
                discriminator.d_count += 1


def f_train_d(sess, dis_data_loader, positive_file_a, positive_file_b, negative_file, generator, discriminator,
              tb_write, d_epoch):
    for _ in range(d_epoch):
        print 'Training {} Dis at epoch {}, d_count is {}'.format(discriminator.domain_name, _, discriminator.d_count)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file_a, positive_file_b, negative_file)
        for __ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                loss = discriminator.train(sess, x_batch, y_batch, dis_dropout_keep_prob)
                if discriminator.d_count % print_inter == 0:
                    rout('Printing tensorboard', discriminator.d_count)
                    _sid, _summ = discriminator.generate_summary(sess, x_batch, y_batch, dis_dropout_keep_prob)
                    tb_write.add_summary(_summ, _sid)
                discriminator.d_count += 1


def f_fusion_d(sess, dis_data_loader, positive_file_a, positive_file_b, negative_file, generator, discriminator,
               tb_write, d_epoch):
    for _ in range(d_epoch):
        print 'Fusion Training {} Dis at epoch {}, d_count is {}'.format(discriminator.domain_name, _,
                                                                         discriminator.d_count)
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file_a, positive_file_b, negative_file)
        for __ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                loss = discriminator.train(sess, x_batch, y_batch, dis_dropout_keep_prob)
                if discriminator.d_count % print_inter == 0:
                    rout('Printing tensorboard', discriminator.d_count)
                    _sid, _summ = discriminator.generate_summary(sess, x_batch, y_batch, dis_dropout_keep_prob)
                    tb_write.add_summary(_summ, _sid)
                discriminator.d_count += 1


def ab_fusion_d(sess, dis_data_loader, positive_file_a, positive_file_b, negative_file, generator,
                discriminator, generator_f,
                tb_write, d_epoch):
    for _ in range(d_epoch):
        print 'Fusion Training {} Dis at epoch {}, d_count is {}'.format(discriminator.domain_name, _,
                                                                         discriminator.d_count)
        neutral_file = 'neutral_file'
        generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        generate_samples(sess, generator_f, BATCH_SIZE, generated_num, neutral_file)

        dis_data_loader.load_train_data(positive_file_a, positive_file_b, negative_file, neutral_file)
        for __ in range(3):
            dis_data_loader.reset_pointer()
            for it in xrange(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                loss = discriminator.get_score(sess, x_batch, dis_dropout_keep_prob)
                # refresh_output('XXXXXXX', np.array(loss).shape)
                loss = loss[0]

                ppnn = np.array(y_batch)
                ppnn_pa = np.vectorize(lambda x: bool(x))(ppnn[:, 0])
                ppnn_pb = np.vectorize(lambda x: bool(x))(ppnn[:, 1])
                ppnn_ng = np.vectorize(lambda x: bool(x))(ppnn[:, 2])
                ppnn_ne = np.vectorize(lambda x: bool(x))(ppnn[:, 3])

                loss_pa = loss[ppnn_pa, :]
                loss_pb = loss[ppnn_pb, :]
                loss_ng = loss[ppnn_ng, :]
                loss_ne = loss[ppnn_ne, :]

                additional_loss = 0.0
                # XA>GF
                if len(loss_pa) != 0 and len(loss_ne) != 0:
                    diff = np.mean(loss_pa[:, 0]) - np.mean(loss_ne[:, 0])
                    if str(diff).isdigit():
                        print 'XA>GF'
                        additional_loss -= diff

                # GF>XB
                if len(loss_ne) != 0 and len(loss_pb) != 0:
                    diff = np.mean(loss_ne[:, 0]) - np.mean(loss_pb[:, 0])
                    if str(diff).isdigit():
                        print 'GF>XB'
                        additional_loss -= diff

                rout('additional_loss', additional_loss)

                x_batch_pa = x_batch[ppnn_pa, :]
                x_batch_pb = x_batch[ppnn_pb, :]
                x_batch_ng = x_batch[ppnn_ng, :]
                x_batch_ne = x_batch[ppnn_ne, :]

                x_batch = np.concatenate((x_batch_pa, x_batch_ng), axis=0)
                y_batch = y_batch[ppnn_pa + ppnn_ng, :][:, [0, 2]]
                loss = discriminator.train(sess, x_batch, y_batch, dis_dropout_keep_prob,
                                           additional_loss=additional_loss)

                if discriminator.d_count % print_inter == 0:
                    rout('Printing tensorboard', discriminator.d_count)
                    _sid, _summ = discriminator.generate_summary(sess, x_batch, y_batch, dis_dropout_keep_prob)
                    tb_write.add_summary(_summ, _sid)
                discriminator.d_count += 1


def seqgan(pos_file_a, pos_file_b):
    print 'Init Variable ###########################################'
    # random.seed(SEED)
    # np.random.seed(SEED)
    assert START_TOKEN == 0

    positive_file_a, negative_file_a, output_music_gan_a, output_music_mle_a = init_var(pos_file_a)
    gen_data_loader_a, dis_data_loader_a = init_data_loader(positive_file_a)
    generator_a = gen('a')
    discriminator_a = dis('a')

    positive_file_b, negative_file_b, output_music_gan_b, output_music_mle_b = init_var(pos_file_b)
    gen_data_loader_b, dis_data_loader_b = init_data_loader(positive_file_b)
    generator_b = gen('b')
    discriminator_b = dis('b')

    negative_file_f = 'tmp_f'
    dis_data_loader_f = F_Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)
    gen_data_loader_f = F_Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    gen_data_loader_f.create_batches(positive_file_a, positive_file_b)
    generator_f = gen('f')
    discriminator_f = dis('f', num_class=3)

    dis_data_loader_ab = AB_Dis_dataloader(BATCH_SIZE, SEQ_LENGTH)

    print 'Init TensorFlow ###########################################'
    # init TensorFlow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # for tensorboard, debug
    tb_write = tf.summary.FileWriter(logdir)
    tb_write.add_graph(sess.graph)
    mio = MIDI_IO()

    print '#########################################################################'
    with tf.name_scope('pretrain-G-A'):
        print 'Start pre-training generator A'
        pre_train_epoch(sess, generator_a, gen_data_loader_a, tb_write, PRE_GEN_EPOCH)
    
        # with tf.name_scope('pretrain-D-A'):
        print 'Start pre-training discriminator A'
        train_d(sess, dis_data_loader_a, positive_file_a, negative_file_a, generator_a, discriminator_a, tb_write,
                PRE_DIS_EPOCH * 3)
    
        # with tf.name_scope('rollout-A'):
        rollout_a = ROLLOUT(generator_a, 0.8, SEQ_LENGTH)
    
    # sample sequence after MLE
    generate_samples(sess, generator_a, BATCH_SIZE, gen_len, output_music_mle_a)
    
    print '-------------------------------------------------------------------------'
    print 'Start Adversarial Training A'
    with tf.name_scope('GANtrain-A'):
        for total_batch in range(GAN_OUTER_EPOCH):
            print 'Adversarial Training Progress', total_batch
    
             # Train the generator
             gan_g(sess, generator_a, discriminator_a, rollout_a, tb_write, GAN_G_EPOCH)
    
             # Update roll-out parameters
             rollout_a.update_params()
    
             # Train the discriminator
             train_d(sess, dis_data_loader_a, positive_file_a, negative_file_a, generator_a, discriminator_a, tb_write,
                     GAN_D_EPOCH)

    generate_samples(sess, generator_a, BATCH_SIZE, gen_len, output_music_gan_a)

    print '#########################################################################'
    print '#########################################################################'
    with tf.name_scope('pretrain-G-B'):
        print 'Start pre-training generator B'
        pre_train_epoch(sess, generator_b, gen_data_loader_b, tb_write, PRE_GEN_EPOCH)

        # with tf.name_scope('pretrain-D-B'):
        print 'Start pre-training discriminator B'
        train_d(sess, dis_data_loader_b, positive_file_b, negative_file_b, generator_b, discriminator_b, tb_write,
                PRE_DIS_EPOCH)

        # with tf.name_scope('rollout-B'):
        rollout_b = ROLLOUT(generator_b, 0.8, SEQ_LENGTH)

    # sample sequence after MLE
    generate_samples(sess, generator_b, BATCH_SIZE, gen_len, output_music_mle_b)

    print '-------------------------------------------------------------------------'
    print 'Start Adversarial Training B'
    with tf.name_scope('GANtrain-B'):
        for total_batch in range(GAN_OUTER_EPOCH):
            print 'Adversarial Training Progress', total_batch

            # Train the generator
            gan_g(sess, generator_b, discriminator_b, rollout_b, tb_write, GAN_G_EPOCH)

            # Update roll-out parameters
            rollout_b.update_params()

            # Train the discriminator
            train_d(sess, dis_data_loader_b, positive_file_b, negative_file_b, generator_b, discriminator_b,
                    tb_write, GAN_D_EPOCH)

    # generate_samples(sess, generator_b, BATCH_SIZE, gen_len, output_music_gan_b)

    print '#########################################################################'
    print '#########################################################################'
    with tf.name_scope('pretrain-G-F'):
        print 'Start pre-training generator F'
        pre_train_epoch(sess, generator_f, gen_data_loader_f, tb_write, PRE_GEN_EPOCH)

        # with tf.name_scope('pretrain-D-F'):
        print 'Start pre-training discriminator F'
        f_train_d(sess, dis_data_loader_f, positive_file_a, positive_file_b, negative_file_f, generator_f,
                  discriminator_f, tb_write,
                  PRE_DIS_EPOCH)

        # with tf.name_scope('rollout-B'):
        rollout_f = ROLLOUT(generator_f, 0.8, SEQ_LENGTH)

    # sample sequence after MLE
    generate_samples(sess, generator_f, BATCH_SIZE, gen_len, 'pretrain-f')
    mio.trans_generated_to_midi('pretrain-f')

    # print '-------------------------------------------------------------------------'
    # print 'Start Adversarial Training F'
    # with tf.name_scope('GANtrain-F'):
    #     for total_batch in range(GAN_OUTER_EPOCH):
    #         print 'Adversarial Training Progress', total_batch
    #
    #         # Train the generator
    #         gan_g(sess, generator_f, discriminator_f, rollout_f, tb_write, GAN_G_EPOCH)
    #
    #         # Update roll-out parameters
    #         rollout_f.update_params()
    #
    #         # Train the discriminator
    #         train_d(sess, dis_data_loader_f, positive_file_f, negative_file_f, generator_f, discriminator_f,
    #                 tb_write, GAN_D_EPOCH)

    for fusion_total_batch in range(FUSION_EPOCH):
        print '#########################################################################'
        print 'Start Fusion GAN', fusion_total_batch
        print '-------------------------------------------------------------------------'
        print 'Start Fusion GAN Training F'
        with tf.name_scope('Fusion-A-B'):
            for total_batch in range(FUSION_F_EPOCH):
                fusion_g(sess, generator_f, discriminator_a, discriminator_b, discriminator_f, rollout_f, tb_write,
                         FUSION_G_EPOCH)
                rollout_f.update_params()
                f_fusion_d(sess, dis_data_loader_f, positive_file_a, positive_file_b, negative_file_f, generator_f,
                           discriminator_f,
                           tb_write,
                           FUSION_D_EPOCH)

        generate_samples(sess, generator_f, BATCH_SIZE, gen_len, 'fusion_' + str(fusion_total_batch))
        mio.trans_generated_to_midi('fusion_' + str(fusion_total_batch))
        # at last iteration, A and B do not need training
        print '++++++'
        print 'CHK PNT', fusion_total_batch, FUSION_EPOCH
        print '++++++'
        if fusion_total_batch == FUSION_EPOCH - 1:
            break
        print '-------------------------------------------------------------------------'
        print 'Start Fusion GAN Training A'
        with tf.name_scope('GANtrain-A'):
            for total_batch in range(FUSION_AB_EPOCH):
                print 'Adversarial Training Progress', total_batch

                # Train the generator
                fusion_g(sess, generator_a, discriminator_a, discriminator_b, discriminator_f, rollout_a, tb_write,
                         FUSION_G_EPOCH)

                # Update roll-out parameters
                rollout_a.update_params()

                # Train the discriminator
                ab_fusion_d(sess, dis_data_loader_ab, positive_file_a, positive_file_b, negative_file_a, generator_a,
                            discriminator_a,
                            generator_f,
                            tb_write,
                            GAN_D_EPOCH)

        print '-------------------------------------------------------------------------'
        print 'Start Fusion GAN Training B'
        with tf.name_scope('GANtrain-A'):
            for total_batch in range(FUSION_AB_EPOCH):
                print 'Adversarial Training Progress', total_batch

                # Train the generator
                fusion_g(sess, generator_b, discriminator_a, discriminator_b, discriminator_f, rollout_b, tb_write,
                         FUSION_G_EPOCH)

                # Update roll-out parameters
                rollout_b.update_params()

                # Train the discriminator
                ab_fusion_d(sess, dis_data_loader_ab, positive_file_b, positive_file_a, negative_file_b, generator_b,
                            discriminator_b,
                            generator_f,
                            tb_write,
                            GAN_D_EPOCH)

    output_music_fusion = 'fusion_gan{}'.format(suffix)
    generate_samples(sess, generator_f, BATCH_SIZE, gen_len, output_music_fusion)

    mio.trans_generated_to_midi(output_music_fusion)

if __name__ == '__main__':
    seqgan('./jazz_maj_midi.pk','./folk_maj_midi.pk')
