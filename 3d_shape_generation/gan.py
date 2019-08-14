from network import build_generator, build_discriminator
from visualize import *
import tensorflow as tf
from tensorflow.train import AdamOptimizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
import time
import numpy as np

tf.enable_eager_execution()


# hyperparameters
gen_learning_rate = 2.5e-3
disc_learning_rate = 1e-5
beta = 0.5
batch_size = 16
z_size = 200
DIR_PATH = '../Datasets/3DShapeNets/volumetric_data/'
generated_volumes_dir = 'generated_volumes'
log_dir = 'logs'

epochs = 100

generator = build_generator(verbose=False)
discriminator = build_discriminator(verbose=False)

gen_optimizer = AdamOptimizer(learning_rate=gen_learning_rate, beta1=beta)
disc_optimizer = AdamOptimizer(learning_rate=disc_learning_rate, beta1=0.9)

generator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.compile(loss='binary_crossentropy', optimizer=disc_optimizer)

discriminator.trainable = False
adversarial_model = Sequential()
adversarial_model.add(generator)
adversarial_model.add(discriminator)
adversarial_model.compile(loss='binary_crossentropy',
						  optimizer=gen_optimizer)

volumes = get_3images_for_category(obj='airplane', train=True, obj_ratio=1.0)
volumes = volumes[..., np.newaxis].astype(np.float)

tensorboard = TensorBoard(log_dir='{}/{}'.format(log_dir, time.time()))
tensorboard.set_model(generator)
#tensorboard.set_model(discriminator)

for epoch in range(epochs):
	print('Epoch:', epoch)
	
	gen_losses = []
	disc_losses = []

	number_batches = volumes.shape[0] // batch_size
	print('Number of batches:', number_batches)

	for index in range(number_batches):
		print('Batch:', index+1)

		z_sample = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)
		volumes_batch = volumes[index * batch_size: (index+1) * batch_size, :, :, :]

		gen_volumes = generator.predict(z_sample, verbose=3)

		labels_real = np.reshape([1] * batch_size, (-1, 1, 1, 1, 1))
		labels_fake = np.reshape([0] * batch_size, (-1, 1, 1, 1, 1))

		discriminator.trainable = True

		loss_real = discriminator.train_on_batch(volumes_batch, labels_real)
		loss_fake = discriminator.train_on_batch(gen_volumes, labels_fake)

		d_loss = 0.5 * (loss_real + loss_fake)

		z = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype(np.float32)

		g_loss = adversarial_model.train_on_batch(z, np.reshape([1] * batch_size, (-1, 1, 1, 1, 1)))

		gen_losses.append(g_loss)
		disc_losses.append(d_loss)

		if index % 10 == 0:
			z_sample2 = np.random.normal(0, 0.33, size=[batch_size, 1, 1, 1, z_size]).astype('float32')
			generated_volumes = generator.predict(z_sample2, verbose=3)

			for i, generated_volume in enumerate(generated_volumes[:5]):
				voxels = np.squeeze(generated_volume)
				voxels[voxels < 0.5] = 0.
				voxels[voxels >= 0.5] = 1.
				plot_voxels(voxels, 'results/img_{}_{}_{}'.format(epoch, index, i))

	write_log(tensorboard, 'g_loss', np.mean(gen_losses), epoch)
	write_log(tensorboard, 'd_loss', np.mean(disc_losses), epoch)

generator.save_weights(os.path.join(generated_volumes_dir, 'generator_weights.h5'))
discriminator.save_weights(os.path.join(generated_volumes_dir, 'discriminator_weights.h5'))
