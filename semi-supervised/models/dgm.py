import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .vae import VariationalAutoencoder
from .vae import Encoder, Decoder, LadderEncoder, LadderDecoder, ConvPreEncoder, ConvPostDecoder


class Classifier(nn.Module):
    def __init__(self, dims, activation_fn=nn.ReLU, batch_norm=True):
        """
        Single hidden layer classifier
        with softmax output.
        """
        super(Classifier, self).__init__()
        [x_dim, h_dim, y_dim] = dims

        print(x_dim)
        if isinstance(x_dim, list):
            self.first_dense = nn.ModuleList(
                [nn.Linear(in_dim, h_dim[0]) for in_dim in x_dim])
        else:
            self.first_dense = nn.ModuleList([nn.Linear(x_dim, h_dim[0])])

        # Combine all inputs in the first layer by summation
        # e.g. z = w_1 * x + w_2 * a
        linear_layers = []
        for idx in range(0, len(h_dim) - 1):
            if batch_norm:
                linear_layers += [
                    activation_fn(),
                    nn.BatchNorm1d(h_dim[idx]),
                    nn.Linear(h_dim[idx], h_dim[idx+1])
                ]
            else:
                linear_layers += [
                    activation_fn(),
                    nn.Linear(h_dim[idx], h_dim[idx+1])
                ]

        # linear_layers += [activation_layer(), nn.BatchNorm1d(h_dim[-1])]
        linear_layers += [activation_fn()]
        if batch_norm:
            linear_layers += [nn.BatchNorm1d(h_dim[-1])]
        

        print("Linear layers in classifier", linear_layers)
        self.hidden = nn.ModuleList(linear_layers)
        self.logits = nn.Linear(h_dim[-1], y_dim)

    def forward(self, input_):
        if not isinstance(input_, list):
            input_ = [input_]

        # Combine inputs
        multi_x = []
        for x, dense in zip(input_, self.first_dense):
            multi_x += [dense(x)]
        x = sum(multi_x)
        
        # Forward
        for layer in self.hidden:
            x = layer(x)
        x = F.softmax(self.logits(x), dim=-1)
        return x


class DeepGenerativeModel(VariationalAutoencoder):
    def __init__(
        self, dims, output_activation=nn.Sigmoid,
        activation_fn=nn.ReLU, batch_norm=True):

        """
        M2 code replication from the paper
        'Semi-Supervised Learning with Deep Generative Models'
        (Kingma 2014) in PyTorch.

        The "Generative semi-supervised model" is a probabilistic
        model that incorporates label information in both
        inference and generation.

        Initialise a new generative model
        :param dims: dimensions of x, y, z and hidden layers.
        """
        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__([x_dim, z_dim, h_dim])

        self.encoder = Encoder(
            [[x_dim, self.y_dim], h_dim, z_dim],
            activation_fn=activation_fn,
            batch_norm=batch_norm
        )
        self.decoder = Decoder(
            [[z_dim, self.y_dim], list(reversed(h_dim)), x_dim],
            activation_fn=activation_fn, 
            output_activation=output_activation,
            batch_norm=batch_norm
        )
        self.classifier = Classifier(
            [x_dim, h_dim, self.y_dim],
            activation_fn=activation_fn,
            batch_norm=batch_norm
        )


        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Add label and data and generate latent variable
        z, z_mu, z_log_var = self.encoder([x, y])

        # E_q(z|x, y) [ p(z) - q(z|x,y) ]
        self.kl_divergence = self._kld(z, (z_mu, z_log_var))

        # Reconstruct data point from latent data and label
        x_mu = self.decoder([z, y])

        return x_mu

    def classify(self, x):
        logits = self.classifier(x)
        return logits

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.float()
        x = self.decoder([z, y])
        return x


class StackedDeepGenerativeModel():
    def __init__(self, dims, features, activation_fn=nn.ReLU):
        """
        M1+M2 model as described in [Kingma 2014].

        Initialise a new stacked generative model
        :param dims: dimensions of x, y, z and hidden layers
        :param features: a pretrained M1 model of class `VariationalAutoencoder`
            trained on the same dataset.
        """
        [x_dim, y_dim, z_dim, h_dim] = dims

        self.dgm = DeepGenerativeModel(
            [features.z_dim, y_dim, z_dim, h_dim],
            activation_fn=activation_fn,
            output_activation=None
        )

        # Be sure to reconstruct with the same dimensions
        # in_features = self.decoder.reconstruction.in_features
        # self.decoder.reconstruction = nn.Linear(in_features, x_dim)

        # Make vae feature model untrainable by freezing parameters
        self.features = features
        self.features.train(False)

        for param in self.features.parameters():
            param.requires_grad = False

    def encode(self, x, y=None):
        _, x, _ = self.features.encode(x)
        if y is None:
            logits = self.dgm.classifier(x)
            y = (logits == logits.max(1)[0].reshape(-1, 1)).float()

        z, z_mu, z_log_var = self.dgm.encoder([x, y])
        return z, z_mu, z_log_var


    def forward(self, x, y):
        # Sample a new latent x from the M1 model
        x_sample, _, _ = self.features.encode(x)
        # Use the sample as new input to M2
        x = self.dgm.forward(x_sample, y)
        return self.features.decoder(x)

    def sample(self, z, y):
        y = y.float()
        x = self.dgm.sample(z, y)
        x = self.features.sample(x)
        return x

    def classify(self, x):
        x, _, _ = self.features.encode(x)
        logits = self.dgm.classifier(x)
        return logits

class AuxiliaryDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims, conv=False, batch_norm=True):
        """
        Auxiliary Deep Generative Models [Maaløe 2016]
        code replication. The ADGM introduces an additional
        latent variable 'a', which enables the model to fit
        more complex variational distributions.

        :param dims: dimensions of x, y, z, a and hidden layers.
        """

        [x_dim, y_dim, z_dim, a_dim, h_dim] = dims
        if conv:
            x_dim = 64 * 4 * 4
        super(AuxiliaryDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim, h_dim])
        self.conv = conv

        if conv:
            self.pre_encoder = ConvPreEncoder()
            self.post_decoder = ConvPostDecoder()

        self.aux_encoder = Encoder([x_dim, h_dim, a_dim], batch_norm=batch_norm)
        self.aux_decoder = Encoder([[x_dim, y_dim, z_dim], list(reversed(h_dim)), a_dim], batch_norm=batch_norm)

        self.classifier = Classifier([[x_dim, a_dim], h_dim, y_dim], batch_norm=batch_norm)

        self.encoder = Encoder([[x_dim, y_dim,  a_dim], h_dim, z_dim], batch_norm=batch_norm)
        self.decoder = Decoder([[z_dim, y_dim], list(reversed(h_dim)), x_dim], batch_norm=batch_norm)

    def encode(self, x, y=None):
        if self.conv:
            x = self.pre_encoder(x)

        a, a_mu, a_log_var = self.aux_encoder(x)

        if y is None:
            logits = self.classifier([x, a])
            y = (logits == logits.max(1)[0].reshape(-1, 1)).float()

        z, z_mu, z_log_var = self.encoder([x, y, a])
        return z, z_mu, z_log_var


    def classify(self, x):
        if self.conv:
            x = self.pre_encoder(x)
        # Auxiliary inference q(a|x)
        a, a_mu, a_log_var = self.aux_encoder(x)

        # Classification q(y|a,x)
        logits = self.classifier([x, a])
        return logits

    def forward(self, x, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """
        if self.conv:
            x = self.pre_encoder(x)

        # print(self.conv, x.shape)
        # Auxiliary inference q(a|x)

        q_a, q_a_mu, q_a_log_var = self.aux_encoder(x)

        # Latent inference q(z|a,y,x)
        z, z_mu, z_log_var = self.encoder([x, y, q_a])

        # Generative p(x|z,y)
        x_mu = self.decoder([z, y])
        if self.conv:
            x_mu = self.post_decoder(x_mu)

        # No need to generate p(a|z,y,x)

        # KL done in the same way as in
        # http://github.com/ml-lab/auxiliary-deep-generative-models 

        # E_q(a,z|x, y) [ p(a) - q(a|x) ]
        a_kl = self._kld(q_a, (q_a_mu, q_a_log_var))

        # E_q(a,z|x, y) [ p(z) - q(z|x,y) ]
        z_kl = self._kld(z, (z_mu, z_log_var))

        self.kl_divergence = a_kl + z_kl

        return x_mu

    def sample(self, z, y):
        """
        Forward through the model
        :param x: features
        :param y: labels
        :return: reconstruction
        """

        # Generative p(x|z,y)
        x_mu = self.decoder([z, y])
        if self.conv:
            x_mu = self.post_decoder(x_mu)

        return x_mu


class LadderDeepGenerativeModel(DeepGenerativeModel):
    def __init__(self, dims):
        """
        Ladder version of the Deep Generative Model.
        Uses a hierarchical representation that is
        trained end-to-end to give very nice disentangled
        representations.

        :param dims: dimensions of x, y, z layers and h layers
            note that len(z) == len(h).
        """
        [x_dim, y_dim, z_dim, h_dim] = dims
        super(LadderDeepGenerativeModel, self).__init__([x_dim, y_dim, z_dim[0], h_dim])

        neurons = [x_dim, *h_dim]
        encoder_layers = [LadderEncoder([neurons[i - 1], neurons[i], z_dim[i - 1]]) for i in range(1, len(neurons))]

        e = encoder_layers[-1]
        encoder_layers[-1] = LadderEncoder([e.in_features + y_dim, e.out_features, e.z_dim])

        decoder_layers = [LadderDecoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]]) for i in range(1, len(h_dim))][::-1]

        self.classifier = Classifier([x_dim, h_dim[0], y_dim])

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = Decoder([z_dim[0]+y_dim, h_dim, x_dim])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        # Gather latent representation
        # from encoders along with final z.
        latents = []
        for i, encoder in enumerate(self.encoder):
            if i == len(self.encoder)-1:
                x, (z, mu, log_var) = encoder(torch.cat([x, y], dim=1))
            else:
                x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))

        latents = list(reversed(latents))

        self.kl_divergence = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            # If at top, encoder == decoder,
            # use prior for KL.
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kl_divergence += self._kld(z, (l_mu, l_log_var))

            # Perform downword merge of information.
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kl_divergence += self._kld(*kl)

        x_mu = self.reconstruction(torch.cat([z, y], dim=1))
        return x_mu

    def sample(self, z, y):
        for i, decoder in enumerate(self.decoder):
            z = decoder(z)
        return self.reconstruction(torch.cat([z, y], dim=1))
