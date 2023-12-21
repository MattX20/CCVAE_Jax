## Config for CCVAE

mnist_config = {
    'num_classes' : 10,
    'latent_dim' : 50,
    'img_shape' : (28, 28, 1),
    'distribution' : "bernoulli",
    'multiclass' : False,
    'scale_factor' : 0.1
}

cifar10_config = {
    'num_classes' : 10,
    'latent_dim' : 128,
    'img_shape' : (32, 32, 3),
    'distribution' : "laplace",
    'multiclass' : False,
    'scale_factor' : 0.3
}

celeba_config = {
    'num_classes' : 40,
    'latent_dim' : 128,
    'img_shape' : (64, 64, 3),
    'distribution' : "laplace",
    'multiclass' : True,
    'scale_factor' : 0.3
}

celeba128_config = {
    'num_classes' : 40,
    'latent_dim' : 128,
    'img_shape' : (128, 128, 3),
    'distribution' : "laplace",
    'multiclass' : True,
    'scale_factor' : 0.3
}

def get_config(dataset_name: str):
    if dataset_name == "MNIST":
        return mnist_config
    elif dataset_name == "CIFAR10":
        return cifar10_config
    elif dataset_name == "CELEBA":
        return celeba_config
    elif dataset_name == "CELEBA128":
        return celeba128_config
    else:
        raise ValueError("Unknown dataset:", str(dataset_name))