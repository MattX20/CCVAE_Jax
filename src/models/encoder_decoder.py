from flax import linen as nn


# MNIST

class MNISTEncoder(nn.Module):
    """
        MNISTEncoder is a simple encoder for the MNIST dataset.
        It returns a 32 vector from a 28x28 grayscale image.
    """
    output_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 28, 28, 1)
        x = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        
        return x


class MNISTDecoder(nn.Module):
    """
        MNISTDecoder is a simple decoder for the MNIST dataset.
        It returns a 28x28 grayscale image from a 32 vector. 
    """
    input_dim: int = 32

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 32)
        x = x.reshape((-1, 1, 1, 32))
        x = nn.relu(x)

        x = nn.ConvTranspose(features=16, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=8, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=4, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=1, kernel_size=(2, 2), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)

        return x


# CIFAR-10

class CIFAR10Encoder(nn.Module):
    """
        CIFAR10Encoder is a simple encoder for the CIFAR-10 dataset.
        It returns a 64 vector from a 32x32 RGB image.
    """
    output_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 32, 32, 1)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        print(x.shape)
        return x
    

class CIFAR10Decoder(nn.Module):
    """
        CIFAR10Decoder is a simple decoder for the CIFAR-10 dataset.
        It returns a 32x32 RGB image from a 64 vector. 
    """
    input_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 64)
        x = x.reshape((-1, 1, 1, 64))
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=16, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)

        return x


# CELEBA 

class CELEBAEncoder(nn.Module):
    """
        CELEBAEncoder is the original encoder proposed by the authors for the CELEBA dataset.
        It returns a 256 vector from a 64x64 RGB image.
    """
    output_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, 64, 64, 3)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(4, 4), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))

        return x
    
    
class CELEBADecoder(nn.Module):
    """
        CELEBADecoder is the original decoder proposed by the authors for the CELEBA dataset.
        It returns a 64x64 RGB image from a 256 vector. 
    """
    input_dim: int = 256

    @nn.compact
    def __call__(self, x):
        # Input latent is expected to be of shape (batch_size, 256)
        x = x.reshape((-1, 1, 1, 256))
        x = nn.relu(x)

        x = nn.ConvTranspose(features=128, kernel_size=(4, 4), strides=(1, 1), padding='VALID')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=3, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.sigmoid(x)

        return x
    
def get_encoder_decoder(dataset_name: str):
    if dataset_name == "MNIST":
        return MNISTEncoder, MNISTDecoder
    elif dataset_name == "CIFAR10":
        return CIFAR10Encoder, CIFAR10Decoder
    elif dataset_name == "CELEBA":
        return CELEBAEncoder, CELEBADecoder
    else:
        raise ValueError("Unknown dataset:", str(dataset_name))