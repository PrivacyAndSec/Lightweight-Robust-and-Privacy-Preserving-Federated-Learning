import torch
from torch import nn
import poison
from torch.optim import SGD
from functools import reduce
import numpy as np
from torch.functional import F
import torchvision as tv

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # cifar-10 3072
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def setup_model(model_architecture, num_classes=None):
    available_models = {
        "CNNMNIST": NeuralNetwork,
        "VGG16": tv.models.vgg16,

    }
    print('--> Creating {} model.....'.format(model_architecture))
    # variables in pre-trained ImageNet models are model-specific.
    if "VGG16" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.classifier[0].in_features
        model.classifier = nn.Linear(n_features, num_classes)

    else:
        model = available_models[model_architecture]()

    if model is None:
        print("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)
    print('--> Model has been created!')
    return model


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class Trianer():

    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = setup_model(model_name, 10).to(self.device)
        if model_name == 'VGG16':
            self.model = WrappedModel(self.model)
        self.weight = None

    def set_weight(self, round_id):
        if round_id == 0:
            self.weight = np.array(self.flatten(self.model.parameters())) - np.array(self.flatten(self.model.parameters()))
        else:
            self.weight = np.array(self.flatten(self.model.parameters()))
        self.set_gradient(0)

    def local_update(self, dataloader, edge_id, round_id, arg, malicious,
                     lr=0.01, momentum=0.9, local_eps=1,):
        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=lr, momentum=momentum)

        epoch_loss = []
        grads_local = [para.detach().clone() for para in self.model.parameters()]
        self.set_weight(round_id)
        for epoch in range(local_eps):
            total_batch_loss = 0
            correct = 0
            for images, labels in dataloader:

                new_images, new_labels, poison_num = poison.get_poison(images, labels, arg=arg, malicious=malicious,adversarial_index=edge_id, edge_id =edge_id, round_id = round_id)
                new_images, new_labels = new_images.to(self.device), new_labels.to(self.device)

                self.model.zero_grad()
                preds = self.model(new_images)
                correct += (preds.argmax(1) == new_labels).type(torch.float).sum().item()
                loss = self.loss_fn(preds, labels)
                total_batch_loss += loss.item()
                loss.backward()
                optimizer.step()

            total_batch_loss /= len(dataloader)
            epoch_loss.append(total_batch_loss)
            acc = float(correct) / len(dataloader.dataset)
            #file = open(arg.result_path, "a+")
            #file.write('round = {:>2d} edge = {} poison = {} Average loss: {:.4f}  Accuracy: {:.2%}\n'.format(
                #round_id, edge_id, malicious,  total_batch_loss, acc))
            #fil#e.close()
            #print('round = {:>2d} edge = {} poison = {} Average loss: {:.4f}  Accuracy: {:.2%}'.format(
                #round_id, edge_id, malicious,  total_batch_loss, acc))

        # 计算梯度
        for idx, para in enumerate(self.model.parameters()):
            grads_local[idx] = (grads_local[idx] - para) / lr

        return grads_local, sum(epoch_loss) / len(epoch_loss)

    def recall(self, test_dataloader, arg, source=None ):
        self.model.eval()
        if source == None:
            recall_total = np.zeros(arg.label_number)
            recall_correct = np.zeros(arg.label_number)
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    preds = self.model(images)
                    preds = preds.argmax(1)
                    for i in range(0, len(preds)):
                        recall_total[labels[i]] = recall_total[labels[i]] + 1
                        if preds[i] == labels[i]:
                            recall_correct[labels[i]] = recall_correct[labels[i]] + 1
            total_recall = recall_correct / recall_total

            file = open(arg.result_path, "a+")
            for i in range(0, arg.label_number):
                if arg.attack == 'target':
                    if arg.source is i:
                        file.write('label : {} source,Recall: {}/{} ({:.2%}))\n'
                              .format(i ,recall_correct[i],
                                      recall_total[i], total_recall[i]))
                        print('label : {} source,Recall: {}/{} ({:.2%}))'
                              .format(i ,recall_correct[i],
                                      recall_total[i], total_recall[i]))
                    elif arg.target is i:
                        file.write('label : {} target,Recall: {}/{} ({:.2%}))\n'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                        print('label : {} target,Recall: {}/{} ({:.2%}))'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                    else:
                        file.write('label : {} ,Recall: {}/{} ({:.2%}))\n'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                        print('label : {} ,Recall: {}/{} ({:.2%}))'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                elif arg.attack == 'backdoor':
                    if arg.target is i :
                        file.write('label : {} target,Recall: {}/{} ({:.2%}))\n'
                                   .format(i, recall_correct[i],
                                           recall_total[i], total_recall[i]))
                        print('label : {} target,Recall: {}/{} ({:.2%}))'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                    else:
                        file.write('label : {} ,Recall: {}/{} ({:.2%}))\n'
                                   .format(i, recall_correct[i],
                                           recall_total[i], total_recall[i]))
                        print('label : {} ,Recall: {}/{} ({:.2%}))'
                              .format(i, recall_correct[i],
                                      recall_total[i], total_recall[i]))
                else:
                    file.write('label : {} ,Recall: {}/{} ({:.2%}))\n'
                               .format(i, recall_correct[i],
                                       recall_total[i], total_recall[i]))
                    print('label : {} ,Recall: {}/{} ({:.2%}))'
                          .format(i, recall_correct[i],
                                  recall_total[i], total_recall[i]))

            file.close()
        else:
            recall_total = 0
            recall_correct = 0
            for images, labels in test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    preds = self.model(images)
                    preds = preds.argmax(1)
                    for i in range(0, len(preds)):
                        if source == labels[i]:
                            recall_total = recall_total + 1
                            if preds[i] == labels[i]:
                                recall_correct = recall_correct + 1
            total_recall = recall_correct / recall_total
            file = open(arg.result_path, "a+")
            if arg.attack == "target":
                file.write('label : {} source,Recall: {}/{} ({:.2%}))\n'
                      .format(source, recall_correct, recall_total, total_recall))
                file.close()
                print('label : {} source,Recall: {}/{} ({:.2%}))'
                      .format(source, recall_correct, recall_total, total_recall))
            elif arg.attack == "backdoor":
                file.write('label : {} target,Recall: {}/{} ({:.2%}))\n'
                           .format(source, recall_correct, recall_total, total_recall))
                file.close()
                print('label : {} target,Recall: {}/{} ({:.2%}))'
                      .format(source, recall_correct, recall_total, total_recall))


    def test_model(self, test_dataloader,epoch, arg):
        self.model.eval()
        total_loss = 0
        correct = 0
        for images, labels in test_dataloader:

            images, labels = images.to(self.device), labels.to(self.device)
            with torch.no_grad():
                preds = self.model(images)
                total_loss += self.loss_fn(preds, labels).item()
                preds = preds.argmax(1)
                correct += (preds == labels).type(torch.float).sum().item()

        total_loss /= len(test_dataloader)
        total_acc = correct / len(test_dataloader.dataset)
        file = open(arg.result_path, "a+")
        file.write('poisoned: {}, round: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2%})\n'
              .format(arg.attack, epoch, total_loss, correct, len(test_dataloader.dataset), total_acc))
        file.close()
        print('poisoned: {}, round: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.2%})'
              .format(arg.attack, epoch, total_loss, correct, len(test_dataloader.dataset), total_acc))


        return total_loss, total_acc

    def yield_accumulated_grads(self, accumulated_grads):
        for i in accumulated_grads:
            yield i

    def flatten(self,generator) -> [float]:
        """
        Get a vector from generator whose element is a matrix of different shapes for different layers of model.
        eg. model.parameters() and yield_accumulated_grads() will create a generator
        for parameters and accumulated_grads
        :param generator:
        :return:
        """

        vector = []
        for para in generator:
            para_list = [i.item() for i in para.flatten()]
            vector.extend(para_list)
        return vector

    def get_gradient(self, generator):
        generator = self.yield_accumulated_grads(generator)
        vector = np.array(self.flatten(generator))
        return vector

    def deflatten(self, vector: [float]) -> None:
        with torch.no_grad():
            index = 0
            for para in self.model.parameters():
                shape = para.shape
                prod_shape = reduce(lambda x, y: x * y, shape)
                para.copy_(torch.tensor(vector[index: (index + prod_shape)]).reshape(shape).requires_grad_())
                index += prod_shape
            assert index == len(vector)

    def set_gradient(self, g):
        self.weight = self.weight - g * 0.01
        self.deflatten(self.weight)