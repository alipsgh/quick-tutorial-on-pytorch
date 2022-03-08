import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
from torch.utils import data

from models.ann import ANN
from models.ae import AE
from models.rbm import RBM
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(0)
if torch.cuda.is_available():
    print("Setting seed for GPU")
    torch.cuda.manual_seed_all(0)


def show_and_save(file_name, img):
    if torch.cuda.is_available():
        img = img.cpu()

    np_img = np.transpose(img.numpy(), (1, 2, 0))

    f = "./%s.png" % file_name
    plt.imshow(np_img, cmap=plt.cm.binary)
    plt.imsave(f, np_img, cmap=plt.cm.binary)


training_set = datasets.MNIST(root='./data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST(root='./data/mnist/', train=False, transform=transforms.ToTensor())

print(len(training_set), "instances exist for training.")
print(len(test_set), "instances exist for testing.")

batch_size = 64
num_epochs = 64

training_set_loader = data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
test_set_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

input_dim = 28 * 28
output_dim = 10

dae = AE(input_dim=input_dim, encoders_dim=[625, 529, 324, 100], encoder_activations=['relu'] * 4, optimizer='adam', learning_rate=0.001)

print(dae.linear_layers)

'''
compress_dim = 500
rbm = RBM(vis_dim=input_dim, hid_dim=compress_dim, k=15, learning_rate=0.1)
'''

pb_epoch = tqdm(total=num_epochs, ncols=100, desc=f"Epoch {0}, training loss: ?", position=0)
pb_batch = tqdm(total=len(training_set_loader), ncols=100, desc="Batch", position=1)
for epoch in range(num_epochs):

    loss_ = []

    pb_batch.reset()
    for i, (train_X, train_y) in enumerate(training_set_loader):

        '''
        train_X = train_X.view(-1, 28 * 28)
        train_cost, train_loss = rbm.fit(train_X)
        loss_.append(train_cost)
        '''

        train_X = train_X.view(-1, 28 * 28)
        train_loss = dae.fit(train_X)
        loss_.append(train_loss)

        pb_batch.set_description(f"Batch {i + 1}")
        pb_batch.update(1)

    pb_epoch.set_description(f"Epoch {epoch + 1}, training loss: {round(sum(loss_)/len(loss_), 4)}")
    pb_epoch.update(1)

# show_and_save("weights_{:.1f}".format(epoch), make_grid(rbm.W.view(compress_dim, 1, 28, 28).data))

for test_X, test_y in test_set_loader:
    test_X = test_X.view(-1, 28 * 28)

    show_and_save("original", make_grid(test_X.view(batch_size, 1, 28, 28)))

    # test_X_ = rbm.compress(test_X)
    test_X_ = dae.reconstruct(test_X)
    show_and_save("generated", make_grid(test_X_.view(batch_size, 1, 28, 28).data))

    test_X_ = dae.encode(test_X)
    show_and_save("encoded", make_grid(test_X_.view(batch_size, 1, 10, 10).data))

    break

Ws, Wm = dae.get_weight()
show_and_save("weights_dae", make_grid(Wm.view(100, 1, 28, 28).data))

w = Ws[0]
show_and_save("weights_dae_1", make_grid(w.cpu().view(625, 1, 28, 28).data))

w = Ws[1]
show_and_save("weights_dae_2", make_grid(w.cpu().view(529, 1, 25, 25).data))

w = Ws[2]
show_and_save("weights_dae_3", make_grid(w.cpu().view(324, 1, 23, 23).data))

w = Ws[3]
show_and_save("weights_dae_4", make_grid(w.cpu().view(100, 1, 18, 18).data))
