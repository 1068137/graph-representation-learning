import torch
import torch.nn as nn

from utils.evaluation import validate


class EarlyStopping:
    def __init__(self, patience: int):
        self.best_acc = -1
        self.patience = patience
        self.k = 0

    def register(self, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.k = 0
            return True
        else:
            self.k += 1
            return False

    def stop(self):
        return self.k == self.patience


def step(model, data, mask, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()

    y_ = model(data.x, data.edge_index)
    loss = loss_fn(y_[mask], data.y[mask])
    loss.backward()
    optimizer.step()

    y_idx = torch.argmax(y_, dim=1)
    acc = (y_idx[mask] == data.y[mask]).float().mean()

    return loss.item(), acc.item()


def train(model, data, args):
    earlyStopping = EarlyStopping(5)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs + 1):
        train_loss, train_acc = step(model, data, data.train_mask, optimizer, loss_fn)
        val_loss, val_acc = validate(model, data, data.val_mask, loss_fn)

        if epoch % 10 == 0:
            pass
            # print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}, Val Acc: {:.3f}".format(epoch + 1,
            #                                                                            args.epochs,
            #                                                                            train_loss,
            #                                                                            train_acc,
            #                                                                            val_acc))

    return model
