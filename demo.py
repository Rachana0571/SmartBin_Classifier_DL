import numpy as np
import cv2
import torch
from torchvision import models, transforms
from PIL import Image

import matplotlib.pyplot as plt

model= models.vgg16(pretrained=False)
def grad_cam(model, img_array, layer_name):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    # Compute weights
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = cv2.resize(cam.numpy(), (img_array.shape[2], img_array.shape[1]))
    cam = np.maximum(cam, 0)
    heatmap = cam / cam.max()

    return heatmap

# Example usage
img = cv2.imread("test.jpg")
img_resized = cv2.resize(img, (224,224))
img_array = np.expand_dims(img_resized/255.0, axis=0)

heatmap = grad_cam(model, img_array, "block5_conv3")  # pick last conv layer

# Threshold + find contours
heatmap = np.uint8(255 * heatmap)
_, thresh = cv2.threshold(heatmap, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
output = img.copy()
cv2.drawContours(output, contours, -1, (0,255,0), 2)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()