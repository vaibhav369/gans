import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LeakyReLU, Conv3D, Conv3DTranspose, BatchNormalization, Activation
from tensorflow.keras.models import Model


def build_generator(verbose=True):

	z_size = 200
	gen_filters = [512, 256, 128, 64, 1]
	gen_kernel_sizes = [4, 4, 4, 4, 4]
	gen_strides = [1, 2, 2, 2, 2]
	gen_input_shape = (1, 1, 1, z_size)
	gen_activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid']
	gen_convolutional_blocks = 5


	input_layer = Input(shape=gen_input_shape)
	a = Conv3DTranspose(filters=gen_filters[0],
						kernel_size=gen_kernel_sizes[0],
						strides=gen_strides[0])(input_layer)
	a = BatchNormalization()(a, training=True)
	a = Activation(activation=gen_activations[0])(a)

	for i in range(gen_convolutional_blocks - 1):
		a = Conv3DTranspose(filters=gen_filters[i+1],
							kernel_size=gen_kernel_sizes[i+1],
							strides=gen_strides[i+1],
							padding='same')(a)
		a = BatchNormalization()(a, training=True)
		a = Activation(activation=gen_activations[i+1])(a)

	model = Model(inputs=input_layer, outputs=a)
	if verbose:
		print(model.summary())

	return model


def build_discriminator(verbose=True):

	disc_input_shape = (64, 64, 64, 1)
	disc_filters = [64, 128, 256, 512, 1]
	disc_kernel_sizes = [4, 4, 4, 4, 4]
	disc_strides = [2, 2, 2, 2, 2]
	disc_padding = ['same', 'same', 'same', 'same', 'valid']
	disc_alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
	disc_activations = ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid']
	disc_convolutional_blocks = 5

	input_layer = Input(shape=disc_input_shape)

	a = Conv3D(filters=disc_filters[0],
			   kernel_size=disc_kernel_sizes[0],
			   strides=disc_strides[0],
			   padding=disc_padding[0])(input_layer)
	a = BatchNormalization()(a, training=True)
	a = LeakyReLU(alpha=disc_alphas[0])(a)

	for i in range(disc_convolutional_blocks - 1):
		a = Conv3D(filters=disc_filters[i+1],
				   kernel_size=disc_kernel_sizes[i+1],
				   strides=disc_strides[i+1],
				   padding=disc_padding[i+1])(a)
		a = BatchNormalization()(a, training=True)
		a = LeakyReLU(alpha=disc_alphas[i+1])(a)

	model = Model(inputs=input_layer, outputs=a)
	if verbose:
		print(model.summary())

	return model


if __name__ == '__main__':

	generator = build_generator(verbose=True)
	discriminator = build_discriminator(verbose=True)

