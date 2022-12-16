import torch


def validate(model, data, mask, loss_fn):
    model.eval()
    y_ = model(data.x, data.edge_index)
    loss = loss_fn(y_[mask], data.y[mask])
    y_idx = torch.argmax(y_, dim=1)
    acc = (y_idx[mask] == data.y[mask]).float().mean()

    return loss.item(), acc.item()


def test(model, data, mask):
    model.eval()
    y_ = model(data.x, data.edge_index)
    y_idx = torch.argmax(y_, dim=1)
    acc = (y_idx[mask] == data.y[mask]).float().mean()

    return acc.item()
