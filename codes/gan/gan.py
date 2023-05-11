from typing import List
from random import sample, randrange
from numpy import array
from torch import nn, randint, tensor, zeros

from torch.optim import Adam



def generate_pseudo_even_data(total_data_size: int, even_data_range: tuple = (0, 10)) -> list:
    # create an empty list to store the even numbers
    even_numbers = []

    # loop 10 times to generate 10 even numbers
    for i in range(total_data_size):
        # generate a random even number between 0 and 100
        num = randrange(even_data_range[0], even_data_range[1] + 1, 2)
        # add the number to the list
        even_numbers.append(num)
    return even_numbers


def convert_integer_to_binary(input_list: list, binary_size: int):
    def _create_binary_list_from_int(number: int) -> List[int]:
        return [int(x) for x in list(bin(number))[2:]]

    data = [_create_binary_list_from_int(x) for x in input_list]
    return [([0] * (binary_size - len(x))) + x for x in data]


def convert_float_matrix_to_int_list(
    float_matrix: array, threshold: float = 0.5
) -> List[int]:
    return [
        int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold
    ]


def generate_noise(noise_range: tuple = (0, 2)):
    return randint(noise_range[0], noise_range[1], size=(batch_size, input_length)).float()


def create_samples(all_data: list, batch_size: int):
    true_data = sample(all_data, batch_size)
    true_labels = [1] * batch_size
    true_data = convert_integer_to_binary(true_data, input_length)

    return true_data, true_labels

class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense = nn.Linear(int(input_length), 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense(x))


class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))
    

def train(all_data: list, batch_size: int = 16, input_length: int = 10, training_steps: int = 500, lr: float = 0.0001):

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr)

    # loss
    loss = nn.BCELoss()

    for i in range(training_steps):
        # print(f"step {i} ...")
        # zero the gradients on each iteration
        generator_optimizer.zero_grad()

        # Create noisy input for generator, need float type instead of int
        noise = generate_noise()
        generated_data = generator(noise)

        # Generate examples of even real data
        true_data, true_labels = create_samples(all_data, batch_size)
        true_labels = tensor(true_labels).float().unsqueeze(1)
        true_data = tensor(true_data).float()

        # Train the generator
        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels)
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels)

        # add .detach() here think about this
        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, zeros(batch_size).unsqueeze(1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()
    
    return generator, discriminator


from torch.backends import mps
use_gpu = False
if mps.is_available() and mps.is_built():
    use_gpu = True

total_pseudo_data_size = 10000
all_even_data = generate_pseudo_even_data(total_pseudo_data_size)

batch_size = 100
input_length = 10
lr = 0.01
training_steps = 3000
generator, discriminator = train(
    all_even_data, 
    input_length=input_length, 
    batch_size = batch_size, 
    training_steps = training_steps, 
    lr = lr)
noise = randint(0, 2, size=(batch_size, input_length)).float()
generated_data = generator(noise)

print(convert_float_matrix_to_int_list(generated_data))
print(f"Use GPU {use_gpu} ...")
