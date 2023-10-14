import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class DNNCDiscover(nn.Module):
    def __init__(self, input_shape, head_shapes, body_shapes, dropout_rate=0.1, combine_mode="subtract"):
        super(DNNCDiscover, self).__init__()
        self.input_parameters = {
            'input_shape': input_shape,
            'head_shapes': head_shapes,
            'body_shapes': body_shapes,
            'dropout_rate': dropout_rate,
            'combine_mode': combine_mode
        }
        self.combine_mode = combine_mode
        self.heads = nn.ModuleList()
        self.head_dropouts = nn.ModuleList()
        self.bodies = nn.ModuleList()
        self.body_dropouts = nn.ModuleList()

        # Initialize the head layers
        in_shape = input_shape
        for head_shape in head_shapes:
            self.heads.append(nn.Linear(in_shape, head_shape))
            self.head_dropouts.append(nn.Dropout(dropout_rate))
            in_shape = head_shape

        # Determine the output shape after head layers based on combine_mode
        if combine_mode in ['subtract', 'add']:
            in_shape = in_shape
        elif combine_mode == "cos":
            in_shape = 1
        elif combine_mode == 'concat':
            in_shape = in_shape * 2

        # Initialize the body layers
        for body_shape in body_shapes:
            self.bodies.append(nn.Linear(in_shape, body_shape))
            self.body_dropouts.append(nn.Dropout(dropout_rate))
            in_shape = body_shape

        # Output layer
        self.output_layer = nn.Linear(in_shape, 1)
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.support_pos = None
        self.support_neg = None

    def network(self, left, right):
        # Forward pass through the model architecture
        for i in range(len(self.heads)):
            layer = self.heads[i]
            dropout = self.head_dropouts[i]
            left = dropout(torch.relu(layer(left)))
            right = dropout(torch.relu(layer(right)))

        if self.combine_mode == 'subtract':
            result = left - right
        elif self.combine_mode == 'add':
            result = left + right
        elif self.combine_mode == 'cos':
            result = torch.clamp(self.cosine_similarity(left, right).reshape([-1, 1]), 0, 1)
            return result
        elif self.combine_mode == 'concat':
            result = torch.cat([left, right], dim=1)

        for i in range(len(self.bodies)):
            layer = self.bodies[i]
            dropout = self.body_dropouts[i]
            result = dropout(torch.relu(layer(result)))

        result = torch.sigmoid(self.output_layer(result))
        return result

    def forward(self, x):
        # Split input tensor into left and right parts
        inputs = torch.split(x, x.shape[1] // 2, dim=1)
        left = inputs[0]
        right = inputs[1]
        return self.network(left, right)

    def fit(self, x, y, val_set=None, support=None, batch_size=128, epochs=40, print_interval=25, use_gpu=False,
            optimizer=None, learning_rate=0.00005, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False,
            loss_fn=nn.BCELoss(), early_stop=True, es_threshold=20):
        if optimizer is None:
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate, betas=betas, eps=eps,
                                     weight_decay=weight_decay, amsgrad=amsgrad)
        x_train = t.from_numpy(x).float()
        y_train = t.from_numpy(y).float()
        
        if support is not None:
            if isinstance(support, list):
                self.support_pos = support[0]
                self.support_neg = support[1]
            elif isinstance(support, dict):
                self.support_pos = support

        if val_set is not None:
            x_val, y_val = val_set
            x_val = t.from_numpy(x_val).float()
            y_val = t.from_numpy(y_val).float()
            if use_gpu:
                x_val = x_val.cuda()
                y_val = y_val.cuda()

        if use_gpu:
            self.cuda()

        min_val_loss = None
        dataset = TensorDataset(x_train, y_train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
        losses = pd.DataFrame(columns=['train_loss', 'validation_loss'])
        self.train()

        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(data_loader):
                if use_gpu:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                optimizer.zero_grad()
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            with t.no_grad():
                self.eval()
                if use_gpu:
                    train_loss = loss_fn(self(x_train.cuda()), y_train.cuda())
                else:
                    train_loss = loss_fn(self(x_train), y_train)

                if use_gpu:
                    train_loss = train_loss.cpu()
                train_loss = train_loss.detach().numpy()
                text = 'Epoch: %d\tTrain Loss: %f' % (epoch + 1, train_loss)
                losses.loc[epoch + 1, 'train_loss'] = float(train_loss)

                if val_set is not None:
                    val_loss = loss_fn(self(x_val), y_val)
                    if use_gpu:
                        val_loss = val_loss.cpu()
                    val_loss = val_loss.detach().numpy()
                    losses.loc[epoch + 1, 'validation_loss'] = float(val_loss)
                    text += '\tValidation Loss: %f' % val_loss

                    if early_stop and epoch > 0:
                        if min_val_loss is None or val_loss <= min_val_loss:
                            best_epoch = epoch
                            min_val_loss = val_loss
                            best_state = self.state_dict()
                        elif epoch > best_epoch + es_threshold:
                            break

                logging.info(text)
                self.train()

        if early_stop and min_val_loss is not None:
            logging.info('Early stopping at epoch %d' % (best_epoch + 1))
            self.load_state_dict(best_state)

        if use_gpu:
            self.cpu()

        return losses, optimizer

    def predict(self, x):
        self.eval()
        return self(t.from_numpy(x).float()).detach().numpy()

    def predict_pairs(self, left, right):
        self.eval()
        return self.network(t.from_numpy(left).float(), t.from_numpy(right).float()).detach().numpy()

    def predict_support(self, x, n_support=None):
        if n_support is not None:
            positive_support = self.support_pos.sample(n=min(len(self.support_pos), n_support)).reset_index(drop
