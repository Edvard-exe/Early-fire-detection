# Data analysis and calculation
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

plt.rcParams["figure.dpi"] = 200

# Machine learning prediction
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch import topk
from torchvision import transforms as T
from torchvision import models, transforms


from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans


# Keras and TensorFlow
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.resnet50 import preprocess_input


# models
from keras.models import Model

import os
import io


def images_to_df(df: pd.DataFrame, fire: str, none: str) -> pd.DataFrame:
    """
    Takes existing empty DataFrame.
    Selects images from the path.
    Return DataFrame with fire and none images.
    """

    for dirname, _, filenames in os.walk(
        f"/content/gdrive/MyDrive/Fire_data/{fire}"
    ):
        for filename in filenames:
            df = df.append(
                pd.DataFrame(
                    [[os.path.join(dirname, filename), "fire"]],
                    columns=["path", "label"],
                )
            )

    for dirname, _, filenames in os.walk(
        f"/content/gdrive/MyDrive/Fire_data/{none}"
    ):
        for filename in filenames:
            df = df.append(
                pd.DataFrame(
                    [[os.path.join(dirname, filename), "none"]],
                    columns=["path", "label"],
                )
            )

    df = df.sample(frac=1).reset_index(drop=True)

    return df

def scatt_plot(df: pd.DataFrame, x: str, y: str, hue: str, title: str) -> None:
    """
    Takes DataFrame, x, y, hue and title values.
    Returns scatter plot
    """

    plt.figure(figsize=(16, 7))

    palette = ["#7800FF", "#0A6C00"]
    sns.set_palette(palette)
    sns.set_style("ticks")

    ax = sns.scatterplot(data=df, x=x, y=y, hue=hue, s=100, linewidth=0.1)

    sns.despine()
    ax.set(ylabel="")
    plt.legend([], [], frameon=False)
    plt.title(title, fontsize=18)
    plt.show()


def plot_images(label: str, num_img: int, verbose=0) -> None:
    """
    Takes label string and number of images.
    Returns plot of images
    """

    all_plot = fire_df[fire_df["label"] == label].sample(num_img)["path"].tolist()

    labels = [label for i in range(len(all_plot))]

    size = np.sqrt(num_img)
    if int(size) * int(size) < num_img:
        size = int(size) + 1

    plt.figure(figsize=(16, 10))

    for ind, (path_img, label) in enumerate(zip(all_plot, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(path_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")

    plt.show()


def feat_extract(path: str, model: nn.Module) -> pd.DataFrame:
    """
    Takes path to image and model
    Extracts features
    """

    file = path
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    new_img = img.reshape(1, 224, 224, 3)
    prep_img = preprocess_input(new_img)
    features = model.predict(prep_img, use_multiprocessing=True)

    return features


def kmean_clust(df: pd.DataFrame, label: str) -> dict:
    """
    Takes dataframe and image label
    Uses pretrained ResNet50 to cluster
    Return 5 clusters
    """

    model = ResNet50()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    new_df = df[df["label"] == label]
    new_df["features"] = new_df["path"].progress_apply(lambda x: feat_extract(x, model))

    features = np.array(new_df["features"].values.tolist()).reshape(-1, 2048)
    path = np.array(new_df["path"].values.tolist())

    kmeans = KMeans(n_clusters=5, random_state=22)
    kmeans.fit(features)

    groups = {}
    for file, cluster in zip(path, kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

    return groups


def clust_plot(kmean_clust: dict, cluster: int) -> None:
    """
    Takes cluster data and cluster number
    Returns plot of selected cluster images
    """

    plt.figure(figsize=(16, 7))

    k_clust = kmean_clust
    images = k_clust[cluster]
    if len(images) > 30:
        start = np.random.randint(0, len(images))
        images = images[start : start + 25]
    for index, file in enumerate(images):
        plt.subplot(5, 5, index + 1)
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis("off")
        plt.title("Fire")


def line_plot(y1: list, y2: list, title: str, legend_list: list) -> None:
    """
    Takes two lists, title and legend string
    Returns a line graph of a certain number of epochs
    """

    x = np.linspace(0, EPOCHS, EPOCHS)

    fig, ax1 = plt.subplots(1, figsize=(16, 6))
    sns.set_style("ticks")

    sns.lineplot(
        x=x,
        y=y1,
        color="seagreen",
        ax=ax1,
    )

    sns.lineplot(
        x=x,
        y=y2,
        color="darkorchid",
        ax=ax1,
    )

    plt.title(title)
    plt.legend(legend_list)
    sns.despine()
    plt.show()


def conf_matrix(true_label: list, predict_label: list) -> None:
    """
    Takes predicted and true values lists
    Returns confusion matrix
    """

    plt.figure(figsize=(16, 6))
    plt.rc("font", size=9)
    x_axis_labels = ["none", "fire"]
    y_axis_labels = ["none", "fire"]
    sns.heatmap(
        confusion_matrix(true_label, predict_label),
        annot=True,
        cmap="PRGn",
        cbar=False,
        fmt="d",
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
    )
    plt.ylabel("")
    plt.xlabel("")
    sns.despine()
    plt.show()

def transform_image(image_bytes):
    my_transforms = T.Compose(
        [
            T.Resize((229, 229)),
            T.ToTensor(),
            T.Normalize([0.4634, 0.4497, 0.4287], [0.2745, 0.2748, 0.2922]),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.to(device)
    output = model18.forward(tensor)

    probs = torch.nn.functional.softmax(output, dim=1)
    conf, classes = torch.max(probs, 1)
    return conf.item(), classes.item()


def missclasified_images(model: nn.Module) -> list:
    """
    Takes neural network
    Evaluets it
    Returns list of misclassified images
    """

    model.eval()
    images = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0]
            label = batch[1]
            label = torch.argmax(label, dim=1)
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            for sampleno in range(batch[0].shape[0]):
                if label[sampleno] != predictions[sampleno]:
                    images.append(inputs[sampleno].cpu())

        return images


def showimg(images: list) -> None:
    """
    Takes list of image tensors
    Resizes them
    Returns plot of misclassified images
    """

    fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    counter = 0
    for image in images:
        mean = np.repeat([0.485, 0.456, 0.406], 224 * 224).reshape(3, 224, 224)
        std = np.repeat([0.229, 0.224, 0.225], 224 * 224).reshape(3, 224, 224)
        imag = ((image * std) + mean).permute(1, 2, 0).clip(0, 1).detach().numpy()

        ax[counter].imshow(imag)
        ax[counter].set_yticks([])
        ax[counter].set_xticks([])

        counter += 1


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def return_CAM(feature_conv, weight, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        beforeDot = feature_conv.reshape((nc, h * w))
        cam = np.matmul(weight[idx], beforeDot)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def prepear_heatmap(model: nn, df: pd.DataFrame) -> None:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )

    counter = 0
    for fname in df["path"]:
        images = Image.open(fname)
        tensor = preprocess(images)
        prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

        final_layer = model._modules["base"]._modules["7"]._modules.get("1")
        activated_features = SaveFeatures(final_layer)

        prediction = model(prediction_var)
        pred_probabilities = F.softmax(prediction).data.squeeze()
        activated_features.remove()

        weight_softmax_params = list(model._modules.get("head").parameters())
        weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

        class_idx = topk(pred_probabilities, 1)[1].int()
        CAMs = return_CAM(activated_features.features, weight_softmax, class_idx)

        readImg = fname
        img = cv2.imread(readImg)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(
            cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET
        )
        result = heatmap * 0.5 + img * 0.5
        cv2.imwrite(f"image{counter}.jpg", result)

        counter += 1


def plot_image_heatmap(num: int) -> None:

    fig, ax = plt.subplots(1, num, figsize=(16, 4))
    for i in range(0, num):
        ax[i].imshow(Image.open(f"/content/image{i}.jpg"))
        ax[i].set_yticks([])
        ax[i].set_xticks([])