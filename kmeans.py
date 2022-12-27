from copy import deepcopy
import itertools as itertools
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch
import torchvision
from torchvision import datasets, transforms

from autoencoder import Autoencoder
from trainloaderclient import TrainloaderClient

class KMeans:
    def __init__(
        self,
        use_pca=False,
        use_encoder=True,
        add_noise=True,
        num_workers=4,
        num_components=32,
        n_digits=10,
        max_epochs=20,
        max_images=10,
        noise_fractions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        visual=False
    ):
        self.__use_pca = use_pca
        self.__use_encoder = use_encoder
        self.__add_noise = add_noise
        self.__num_workers = num_workers
        self.__num_components = num_components
        self.__n_digits = n_digits
        self.__max_epochs = max_epochs
        self.__max_images = max_images
        self.__noise_fractions = noise_fractions
        self.__visual = visual
        self.__mnist_data = datasets.MNIST(
            'data', 
            train=True, 
            download=True, 
            transform=transforms.ToTensor()
        )
        self.__train_loader = self.__get_datasets_and_trainloaders()


    def __get_datasets_and_trainloaders(self):
        tl_client = TrainloaderClient()
        if self.__use_encoder:
            train_loader = tl_client.get_trainloader_for_size(
                size=1000, 
                trainset=self.__mnist_data, 
                batch_size=10
            )
        else:
            train_loader = tl_client.get_trainloader_for_size(
                size=100, 
                trainset=self.__mnist_data, 
                batch_size=4
            )
        if self.__visual:
            image, _ = self.__mnist_data[0]
            plt.imshow(image.squeeze().numpy(), cmap="Greys")
        return train_loader


    def run(self) -> None:
        train_loader = self.__get_datasets_and_trainloaders()
        if self.__use_encoder:
            self.__train_autoencoder(train_loader)

        if not self.__add_noise:
        
            data, labels = self.__get_data_and_labels_from(train_loader)
        
            if self.__use_encoder:
                embeddings = self.__get_embeddings(data)
                if self.__use_pca:
                    reduced_data = PCA(n_components=self.__num_components).fit_transform(embeddings)
                    kmeans = KMeans(init="k-means++", n_clusters=self.__n_digits, n_init=1).fit(reduced_data)
                    to_fit = reduced_data
                else:
                    kmeans = KMeans(init="k-means++", n_clusters=self.__n_digits, n_init=4, random_state=0).fit(embeddings)
                    to_fit = embeddings
  
            else:
                data = data.view(1000, 784)
                kmeans = KMeans(init="k-means++", n_clusters=self.__n_digits, n_init=4, random_state=0).fit(data)
                to_fit = data
        
            if self.__visual:
                print(kmeans.labels_)

            if self.__use_encoder:
                for i in range (len(labels)):
                    labels[i] = labels[i].item()
                if self.__visual:
                    print(labels)

        if not self.__add_noise:
            target_names = [str(i) for i in range(10)]
            cm = confusion_matrix(y_true=labels, y_pred=kmeans.labels_)
            self.__plot_confusion_matrix(cm, target_names)


    def __add_noise_to(self, image, fraction):
        img = torch.squeeze(deepcopy(image))
        row, col = img.shape
        total_pixels = row * col
        number_of_pixels = int(total_pixels * fraction)
        
        for i in range(number_of_pixels):
            y_coord=random.randint(0, row - 1)
            x_coord=random.randint(0, col - 1)
            img[y_coord][x_coord] = float(random.randint(0, 100) / 100)
            
        return torch.unsqueeze(img, 0)


    def __train(
        self,
        model,
        num_epochs=5,
        fraction=0.2,
        learning_rate=1e-3
    ):
        torch.manual_seed(42)
        # mean square error loss
        criterion = torch.nn.MSELoss() 
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=1e-5)
        outputs = []
        for epoch in range(num_epochs):
            for data in self.__train_loader:
                img, _ = data
                if self.__add_noise:
                    for i in img:
                        i = self.__add_noise_to(i, fraction)
                recon = model(img)
                loss = criterion(recon, img)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
            outputs.append((epoch, img, recon),)
        return outputs


    def __train_autoencoder(self) -> None:
        model = Autoencoder()

        if not self.__add_noise:
            outputs = self.__train(model, self.__train_loader, num_epochs=self.__max_epochs)

        else:
            for noise_fraction in self.__noise_fractions:
                outputs = self.__train(model, fraction=noise_fraction, num_epochs=self.__max_epochs)
            
            for i in range (self.__max_images):
                image, _ = self.__mnist_data[i]
                noisy_image = self.__add_noise_to(image, noise_fraction)
                repaired_image = model.forward(torch.unsqueeze(noisy_image,0))
                img_grid = torchvision.utils.make_grid([image.squeeze(), noisy_image.squeeze(), repaired_image.squeeze()])
                self.__show(img_grid)
                    
    
    def __show(self, imgs):
        _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = transforms.functional.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img), cmap="Greys")
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


    def __get_data_and_labels_from(self, train_loader):
        data = list()
        labels = list()
        for loaded in iter(train_loader):
            for i in range(len(loaded[0])):
                data.append((loaded[0][i]))
                labels.append(loaded[1][i]) 
        
        print(len(data))
        data = torch.stack(data)
        print("data: ", data.size())
        print("labels: ", len(labels))
        (unique, counts) = np.unique(labels, return_counts=True)
        print(np.asarray((unique, counts)))
        return (data, labels)


    def __get_embeddings(self, model, data):
        embeddings = model.encoder(data)
        embeddings = embeddings.detach()
        embeddings.requires_grad = False
        embeddings = embeddings.view(10000, 64)
        return embeddings

    def __plot_confusion_matrix(
        self,
        cm, 
        target_names, 
        title='Confusion matrix', 
        cmap=None, 
        normalize=True
    ):
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True')
        plt.xlabel('Predicted\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()

KMeans().run() 