import math

import torch
import torch.nn as nn

from src import utils


class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.epoch = 0
        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_layers + [10]
        self.act_fn = nn.ReLU() if self.opt.training.backpropagation else ReLU_full_grad()

        input_layer_size = utils.get_input_layer_size(opt)

        # Initialize the model.
        self.model = nn.ModuleList([nn.Linear(input_layer_size + (self.num_channels[1] if self.opt.model.top_down else 0), self.num_channels[0])])
        for i in range(1, len(self.num_channels) - 1):
            self.model.append(nn.Linear(self.num_channels[i - 1] + (self.num_channels[i + 1] if self.opt.model.top_down else 0), self.num_channels[i]))

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize downstream classification loss.
        self.linear_classifier = nn.Sequential(
            nn.Linear(self.num_channels[-2], self.num_channels[-1], bias=True)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()
        self.chose_functions()

    def chose_functions(self):
        if self.opt.model.top_down:
            if self.opt.training.backpropagation:
                self.forward = self.forward_backpropagation_top_down
                self.forward_downstream_classification_model = self.forward_backpropagation_top_down
            else:
                self.forward = self.forward_top_down
                self.forward_downstream_classification_model = self.forward_downstream_classification_model_top_down
                self.forward_downstream_multi_pass = self.forward_downstream_multi_pass_top_down
        elif self.opt.training.backpropagation:
            self.forward = self.forward_backpropagation
            self.forward_downstream_classification_model = self.forward_backpropagation

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(
                    m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                )
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1) # sum of squares of each activation. bs*2

        # print("sum of squares shape: ", sum_of_squares.shape)
        # exit()
        # s - thresh    --> sigmoid --> cross entropy

        logits = sum_of_squares - z.shape[1] # if the average value of each activation is >1, logit is +ve, else -ve.
        ff_loss = self.ff_loss(logits, labels.float()) # labels are 0 or 1, so convert to float. logits->sigmoid->normal cross entropy

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels) # threshold is logits=0, so sum of squares = 784 
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def _calc_loss(self, idx, z, posneg_labels, scalar_outputs, number_of_images=1):
        ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
        # FOR SPEED Incase you want to use you need to uncomment line 228-230 and 94-97
        # scalar_outputs[f"loss_layer_{idx}"] += ff_loss
        # scalar_outputs[f"ff_accuracy_layer_{idx}"] += ff_accuracy
        scalar_outputs["Loss"] += ff_loss  / number_of_images

    def forward(self, inputs, labels): # For Forward-forward
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "classification_loss" : torch.zeros(1, device=self.opt.device)
        }
        # layer_entries = {f"{metric}_layer_{idx}": torch.zeros(1, device=self.opt.device)
        #                 for idx in range(len(self.model)) for metric in ["loss", "ff_accuracy"]}
        # scalar_outputs.update(layer_entries)

        z_all = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z_all.shape[0], device=self.opt.device) 
        posneg_labels[: self.opt.input.batch_size] = 1 
        
        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)

        cum_output = torch.zeros([self.opt.input.batch_size, 10], dtype=torch.float32, device=self.opt.device)
        for f in range(z_all.shape[1]):
            z = z_all[:,f]
            z = self._layer_norm(z)

            for idx, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn.apply(z)
                self._calc_loss(idx, z, posneg_labels, scalar_outputs, z_all.shape[1])
                z = z.detach()
                z = self._layer_norm(z)

            if self.opt.training.unsupervised:
                output = self.linear_classifier(z[: self.opt.input.batch_size])
                classification_loss = self.classification_loss(output, labels["class_labels"]) / z_all.shape[1]
                scalar_outputs["Loss"] += classification_loss
                scalar_outputs["classification_loss"] += classification_loss
                cum_output += output

        if not self.opt.training.unsupervised:
            scalar_outputs = self.forward_downstream_classification_model(
                inputs, labels, scalar_outputs=scalar_outputs
            )

            scalar_outputs = self.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs
            )
            
        else:
            classification_accuracy = utils.get_accuracy(
                    self.opt, cum_output.data, labels["class_labels"]
            )
            scalar_outputs["classification_accuracy"] = classification_accuracy
        
        return scalar_outputs
    
    def forward_downstream_multi_pass(  # For Forward-forward
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z_all = inputs["all_sample"] # bs, num_classes, C, H, W

        if len(z_all.shape) == 5:
            z_all = z_all.unsqueeze(dim=2) # bs, num_classes, FRAMES ,C, H, W

        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], z_all.shape[2], -1) # bs, num_classes,FRAMES, C*H*W

        ssq_all = []
        for class_num in range(z_all.shape[1]):
            ssq = torch.zeros(z_all.shape[0], device=self.opt.device)
            for f in range(z_all.shape[2]):
                z = z_all[:, class_num, f] # bs, C*H*W

                for _, layer in enumerate(self.model):
                    z = self._layer_norm(z)
                    z = layer(z)
                    z = self.act_fn.apply(z)

                ssq += torch.sum(z ** 2, dim=-1) # bs # sum of squares of each activation
            ssq_all.append(ssq)
        ssq_all = torch.stack(ssq_all, dim=-1) # bs x num_classes # sum of squares of each activation for each class
        
        classification_accuracy = utils.get_accuracy(
            self.opt, ssq_all.data, labels["class_labels"]
        )

        scalar_outputs["multi_pass_classification_accuracy"] = classification_accuracy
        return scalar_outputs

    def forward_downstream_classification_model(  # For Forward-forward
        self, inputs, labels, scalar_outputs=None,
    ):        
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
                "classification_loss" : torch.zeros(1, device=self.opt.device)
            }

        if self.opt.training.unsupervised:
            z_all = inputs["pos_images"]
        else:
            z_all = inputs["neutral_sample"]


        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1) 

        cum_output = torch.zeros([self.opt.input.batch_size, 10], dtype=torch.float32, device=self.opt.device)
        for f in range(z_all.shape[1]):
            z = z_all[:,f]
            z = self._layer_norm(z)

            with torch.no_grad():
                for _, layer in enumerate(self.model):
                    z = layer(z)
                    z = self.act_fn.apply(z)
                    z = self._layer_norm(z)

            output = self.linear_classifier(z)
            cum_output += output
            classification_loss = self.classification_loss(output, labels["class_labels"]) / z_all.shape[1]

            scalar_outputs["Loss"] += classification_loss
            scalar_outputs["classification_loss"] += classification_loss
 
        classification_accuracy = utils.get_accuracy(
                self.opt, cum_output.data, labels["class_labels"]
        )
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

    def forward_top_down(self, inputs, labels): # For Forward-forward recurrent net
        frame = self.epoch // self.opt.training.frame_switch if self.opt.training.frame_switch != -1 else 10000
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "classification_loss" : torch.zeros(1, device=self.opt.device)
        }
        # layer_entries = {f"{metric}_layer_{idx}": torch.zeros(1, device=self.opt.device)
        #                 for idx in range(len(self.model)) for metric in ["loss", "ff_accuracy"]}
        # scalar_outputs.update(layer_entries)

        z_all = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z_all.shape[0], device=self.opt.device) 
        posneg_labels[: self.opt.input.batch_size] = 1 

        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)

        loop = z_all.shape[1] if self.opt.training.randomized else 1
        for _ in range(loop):

            cum_output = []
            state = [torch.zeros(z_all.shape[0], channels, device=self.opt.device) for channels in self.num_channels]
            if not self.opt.training.unsupervised:
                state[-1] = z_all[:,0,0:10].clone()
            for f in range(9):
                z = z_all[:,torch.randint(0, z_all.size(1), (1,))[0] if self.opt.training.randomized else min(max(f - 1, 0), z_all.size(1) - 1)]
                z = self._layer_norm(z)

                for idx, layer in enumerate(self.model):
                    z = layer(torch.cat([z, self._layer_norm(state[idx + 1])], dim=1))
                    z = self.act_fn.apply(z)
                    if f < frame + 1:
                        self._calc_loss(idx, z, posneg_labels, scalar_outputs, loop * (frame + 1))
                    state[idx] = (z * 0.7 + state[idx] * 0.3).detach()
                    z = self._layer_norm(state[idx]).detach()

                if self.opt.training.unsupervised:
                    output = self.linear_classifier(z)
                    state[-1] = output.detach()
                    classification_loss = self.classification_loss(output[: self.opt.input.batch_size], labels["class_labels"]) / loop
                    scalar_outputs["Loss"] += classification_loss 
                    scalar_outputs["classification_loss"] += classification_loss
                    cum_output.append(output[: self.opt.input.batch_size])

        if not self.opt.training.unsupervised:
            scalar_outputs = self.forward_downstream_multi_pass(
                inputs, labels, scalar_outputs=scalar_outputs
            )    
        else:
            self.calculate_accurcies(scalar_outputs, cum_output, labels)
        return scalar_outputs

    def forward_downstream_multi_pass_top_down( # For Forward-forward recurrent net
        self, inputs, labels, scalar_outputs=None,
    ):
        z_all = inputs["all_sample"] # bs, num_classes, C, H, W

        if len(z_all.shape) == 5:
            z_all = z_all.unsqueeze(dim=2) # bs, num_classes, FRAMES ,C, H, W

        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], z_all.shape[2], -1) # bs, num_classes,FRAMES, C*H*W

        ssq_all = []
        for class_num in range(z_all.shape[1]):
            ssq = []

            state = []
            z = z_all[:, class_num, 0]
            z = self._layer_norm(z)

            for idx, layer in enumerate(self.model):
                if len(self.model) - 1 == idx:
                    z = layer(torch.cat([z, self._layer_norm(z_all[:, class_num, 0, :10])], dim=1))
                else:
                    z = layer(torch.cat([z, torch.zeros(z.shape[0], self.num_channels[idx + 1], device=self.opt.device)], dim=1))
                z = self.act_fn.apply(z)
                state.append(z * 0.7)
                z = self._layer_norm(z)

            # state.append(self.linear_classifier(z))
            state.append(z_all[:, class_num, 0, :10].clone())
            ssq.append(torch.sum(state[-2] ** 2, dim=-1))

            for f in range(8): 
                z = z_all[:, class_num, min(f, z_all.shape[2] - 1)]
                z = self._layer_norm(z)

                for idx, layer in enumerate(self.model):
                    z = layer(torch.cat([z, self._layer_norm(state[idx + 1])], dim=1)) 
                    z = self.act_fn.apply(z)
                    state[idx] = (z * 0.7 + state[idx] * 0.3)
                    z = self._layer_norm(state[idx])

                # output = self.linear_classifier(z)
                # state[-1] = output

                ssq.append(torch.sum(state[-2] ** 2, dim=-1)) # bs # sum of squares of each activation
            ssq_all.append(ssq)

        self.calculate_accurcies(scalar_outputs, ssq_all, labels)

        return scalar_outputs
    
    def forward_downstream_classification_model_top_down( # For Forward-forward recurrent net
        self, inputs, labels, scalar_outputs=None,
    ):        
        if scalar_outputs is None:
            scalar_outputs = {}

        if self.opt.training.unsupervised:
            z_all = inputs["pos_images"]
        else:
            z_all = inputs["neutral_sample"]

        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)

        cum_output = []
        state = []
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)
        z = z_all[:,0]
        z = self._layer_norm(z)

        for idx, layer in enumerate(self.model):
            z = layer(torch.cat([z, torch.zeros(z.shape[0], self.num_channels[idx + 1], device=self.opt.device)], dim=1))
            z = self.act_fn.apply(z)
            state.append(z * 0.7)
            z = self._layer_norm(z)

        z = self.linear_classifier(z)
        state.append(z)
        cum_output.append(z)

        for f in range(8): 
            z = z_all[:,min(f, z_all.shape[1] - 1)]
            z = self._layer_norm(z)

            for idx, layer in enumerate(self.model):
                z = layer(torch.cat([z, self._layer_norm(state[idx + 1])], dim=1))
                z = self.act_fn.apply(z)
                state[idx] = (z * 0.7 + state[idx] * 0.3)
                z = self._layer_norm(state[idx])

            output = self.linear_classifier(z)
            state[-1] = output
            cum_output.append(output)

        self.calculate_accurcies(scalar_outputs, cum_output, labels)

        return scalar_outputs

    def forward_backpropagation_top_down(self, inputs, labels, scalar_outputs=None): # For For Backpropagation recurrent net
        frame = self.epoch // self.opt.training.frame_switch if self.opt.training.frame_switch != -1 else 10000

        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z_all = inputs["pos_images"]
        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)

        loop = z_all.shape[1] if self.opt.training.randomized and self.training else 1
        for _ in range(loop):

            cum_output = []
            state = [torch.zeros(z_all.shape[0], channels, device=self.opt.device) for channels in self.num_channels]
            for f in range(9):
                z = z_all[:,torch.randint(0, z_all.size(1), (1,))[0] if self.opt.training.randomized and self.training else min(max(f - 1, 0), z_all.size(1) - 1)]

                for idx, layer in enumerate(self.model):
                    z = layer(torch.cat([z, state[idx + 1]], dim=1))
                    z = self.act_fn(z)
                    state[idx] = (z * 0.7 + state[idx] * 0.3)
                    z = state[idx].clone()

                output = self.linear_classifier(z)
                state[-1] = output
                if f < frame + 1:
                    classification_loss = self.classification_loss(output, labels["class_labels"]) / loop
                    scalar_outputs["Loss"] += classification_loss 

                cum_output.append(output)

        self.calculate_accurcies(scalar_outputs, cum_output, labels)
        return scalar_outputs
    
    def forward_backpropagation( # For Backpropagation
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        cum_output = torch.zeros([self.opt.input.batch_size, 10], dtype=torch.float32, device=self.opt.device)
        
        z_all = inputs["pos_images"]
        if len(z_all.shape) == 4:
            z_all = z_all.unsqueeze(dim=1)
        z_all = z_all.reshape(z_all.shape[0], z_all.shape[1], -1)

        for f in range(z_all.shape[1]):
            z = z_all[:,f]
            for _, layer in enumerate(self.model):
                z = layer(z)
                z = self.act_fn(z)

            output = self.linear_classifier(z)
            cum_output += output

            classification_loss = self.classification_loss(output, labels["class_labels"]) / z_all.shape[1]

            scalar_outputs["Loss"] += classification_loss
        
        classification_accuracy = utils.get_accuracy(self.opt, cum_output.data, labels["class_labels"])
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs

    def calculate_accurcies(self, scalar_outputs, output, labels):
        accuracy_ranges = {
                "classification_accuracy": slice(1, None),
                "classification_accuracy_3_5": slice(3, 6),
                "classification_accuracy_last_frame": slice(-1, None),
        }

        for key, range_value in accuracy_ranges.items():
            if len(output) == 10: # 10 class
                data_to_use = torch.stack([sum(output[cls][range_value]).data for cls in range(10)]).transpose(0, 1)
            elif len(output) == 9: # 9 Frame 
                data_to_use = sum(output[range_value]).data
            else:  # it's a single index
                raise ValueError("Unknown value.")
            scalar_outputs[key] = utils.get_accuracy(self.opt, data_to_use, labels["class_labels"])


# unclear as to why normal relu doesn't work # TODO check this
class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
