from copy import deepcopy
import numpy as np
import torch

from customdataset import CustomDataset


class TrainloaderClient:
    def get_trainloader_for_size(size, trainset, batch_size: int, num_workers: int):
        image_index = 0
        label_index = 1

        counter = np.zeros(size)
        dataset = CustomDataset([], [])

        for data in trainset:
            image = data[image_index]
            label = data[label_index]

            if counter[label] < size:
                dataset.data.append(deepcopy(image))
                dataset.targets.append(deepcopy(label))
                counter[label] += 1

            if sum(counter) == size * 10:
                break

        trainloader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True, 
                                                    num_workers=num_workers)

        return trainloader
    