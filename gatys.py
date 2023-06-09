import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()


target_image_path = "/content/content_1.jpg"
style_ref_image_path = "/content/002.jpg"

# downsize measurements
width, height = load_img(target_image_path).size

IMG_HEIGHT = 400
IMG_WIDTH = int(width * IMG_HEIGHT / height)


def load_and_downsize_img(img_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))


def preprocess_img(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def plot_img(img):
    plt.imshow(img)


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    val = K.dot(features, K.transpose(features))

    return val


def style_loss(style, combination):
    style_gram = gram_matrix(style)
    content_gram = gram_matrix(combination)

    channels = 3
    size = IMG_HEIGHT * IMG_WIDTH

    return K.sum(K.square(style_gram - content_gram)) / (4. * (channels**2) * (size**2))


def total_variation_loss(x):
    a = K.square(
        x[:, :img_height-1, :img_width-1, :] -
        x[:, 1:, :img_width-1, :]
    )
    b = K.square(
        x[:, :img_height-1, :img_width-1, :] -
        x[:, :img_height-1, 1:, :]
    )

    return K.sum(K.pow(a + b, 1.25))


outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = "block5_conv2"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1"
]

total_variation_weight = 1e-4
style_weight = 0.025
content_weight = 1.

loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * \
    content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_ref_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_ref_features, combination_features)
    loss = loss + (style_weight/len(style_layers)) * sl

loss = loss + total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]

fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object):

    def __init__(self) -> None:
        self.loss_val = None
        self.grad_val = None

    def loss(self, x):
        assert self.loss_val is None

        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])

        loss_val = outs[0]
        grad_val = outs[1].flatten().astype('float64')

        self.loss_val = loss_val
        self.grad_val = grad_val

        return self.loss_val

    def grads(self, x):
        assert self.loss_val is not None

        grad_val = np.copy(self.grad_val)
        self.loss_val = None
        self.grad_val = None

        return grad_val


evaluator = Evaluator()


result_prefix = 'my_result'
iterations = 20

x = load_and_downsize_img(target_image_path)
x = preprocess_image(x)
x = x.flatten()

losses = []

for i in range(iterations):
    print('Start of iteration', i)

    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(
        evaluator.loss, x, fprime=evaluator.grads, maxfun=20)

    print('Current loss value:', min_val)
    losses.append(min_val)

    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)

    print('Image saved as', fname)

    end_time = time.time()

    print('Iteration %d completed in %ds' % (i, end_time - start_time))

plt.plot(losses, color='red', label="Training Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title("Training Loss")
