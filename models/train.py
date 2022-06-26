from os import path
from pathlib import Path
from typing import Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tb
from tqdm.auto import trange, tqdm

from game.simulator import load_simulator_yarn, GraphSimulator
from helper.helper import save_dict, save_pickle
from .models import StateActionModel, save_model, load_model
from .utils import ConfusionMatrix, load_data


def train(
        model: StateActionModel,
        dict_model: Dict,
        graph: nx.DiGraph,
        log_dir: str = './models/logs',
        save_path: str = './models/saved',
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        n_epochs: int = 100,
        batch_size: int = 8,
        num_workers: int = 0,
        scheduler_mode: str = 'max_val_acc',
        debug_mode: bool = False,
        device=None,
        steps_save: int = 1,
        use_cpu: bool = False,
        freeze_bert: bool = True,
        balanced_actions: bool = False,
        augment_negative:bool = False,
        scheduler_patience:int = 10,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param log_dir: directory where the tensorboard log should be saved
    :param graph: graph for the game
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be `adam, adamw, sgd`
    :param n_epochs: number of epochs of training
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be `min_loss, max_acc, max_val_acc, max_val_mcc`
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param device: if not none, device to use ignoring other parameters. If none, the device will be used depending on `use_cpu` and `debug_mode` parameters
    :param steps_save: number of epoch after which to validate and save model (if conditions met)
    :param freeze_bert: whether to freeze BERT during training
    :param balanced_actions: if true, it will ensure that the actions returned are balanced
    :param augment_negative: if true, it will add a negated version of all the samples by adding "No" in front of all actions
    :param scheduler_patience: patience for the learning rate scheduler
    """

    # cpu or gpu used for training if available (gpu much faster)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    # print(device)

    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # Tensorboard
    global_step = 0
    # dictionary of training parameters
    dict_param = {f"train_{k}": v for k, v in locals().items() if k in [
        'lr',
        'optimizer_name',
        'batch_size',
        'scheduler_mode',
        'balanced_actions',
        'augment_negative',
        'scheduler_patience',
    ]}
    # dictionary to set model name
    name_dict = dict_model.copy()
    name_dict.update(dict_param)
    # model name
    name_model = '/'.join([
        str(name_dict)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
    ])

    # train_logger = tb.SummaryWriter(path.join(log_dir, 'train', name_model), flush_secs=1)
    # valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid', name_model), flush_secs=1)
    train_logger = tb.SummaryWriter(path.join(log_dir, name_model), flush_secs=1)
    valid_logger = train_logger

    # Model
    dict_model.update(dict_param)
    # dict_model.update(dict(
    #     # metrics
    #     train_loss=None,
    #     train_acc=0,
    #     val_acc=0,
    #     epoch=0,
    # ))
    model = model.to(device)

    # Loss
    loss = torch.nn.BCEWithLogitsLoss().to(device)  # sigmoid + BCELoss (good for 2 classes classification)

    # load train and test data
    loader_train, loader_valid, _ = load_data(
        graph=graph,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        random_seed=4444,
        tokenizer=model.tokenizer,
        device=device,
        balanced_actions=balanced_actions,
        augment_negative=augment_negative,
    )

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not configured")

    # :param scheduler_patience: value used as patience for the learning rate scheduler
    if scheduler_mode == "min_loss":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               patience=scheduler_patience)
    elif scheduler_mode in ["max_acc", "max_val_acc", "max_val_mcc"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                               patience=scheduler_patience)
    else:
        raise Exception("Optimizer not configured")

    # print(f"{log_dir}/{name_model}")
    for epoch in (p_bar := trange(n_epochs)):
        p_bar.set_description(f"{name_model} -> {dict_model.get('val_acc', 0)}")
        # print(epoch)
        train_loss = []
        train_cm = ConfusionMatrix(size=2, name='train')

        # Start training: train mode and freeze bert
        model.train()
        model.freeze_bert(freeze_bert)
        for state, action, reward in loader_train:
            # Compute loss and update parameters
            pred = model(state, action)[:, 0]
            loss_val = loss(pred, reward)

            # Do back propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Add train loss and accuracy
            train_loss.append(loss_val.cpu().detach().numpy())
            train_cm.add(preds=(pred > 0).float(), labels=reward)

        # Evaluate the model
        val_cm = ConfusionMatrix(size=2, name='val')
        model.eval()
        with torch.no_grad():
            for state, action, reward in loader_valid:
                pred = model(state, action)[:, 0]
                val_cm.add((pred > 0).float(), reward)

        train_loss = np.mean(train_loss)
        # train_acc = train_cm.global_accuracy
        # val_acc = val_cm.global_accuracy
        # val_mcc = val_cm.matthews_corrcoef

        # Step the scheduler to change the learning rate
        is_better = False
        if scheduler_mode == "min_loss":
            met = train_loss
            if (best_met := dict_model.get('train_loss', None)) is not None:
                is_better = met <= best_met
            else:
                dict_model['train_loss'] = met
                is_better = True
        elif scheduler_mode == "max_acc":
            met = train_cm.global_accuracy
            if (best_met := dict_model.get('train_acc', None)) is not None:
                is_better = met >= best_met
            else:
                dict_model['train_acc'] = met
                is_better = True
        elif scheduler_mode == "max_val_acc":
            met = val_cm.global_accuracy
            if (best_met := dict_model.get('val_acc', None)) is not None:
                is_better = met >= best_met
            else:
                dict_model['val_acc'] = met
                is_better = True
        elif scheduler_mode == 'max_val_mcc':
            met = val_cm.matthews_corrcoef
            if (best_met := dict_model.get('val_mcc', None)) is not None:
                is_better = met >= best_met
            else:
                dict_model['val_mcc'] = met
                is_better = True
        else:
            met = None

        if met is not None:
            scheduler.step(met)

        global_step += 1
        if train_logger is not None:
            suffix = 'train'
            train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
            log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)
            # validation log
            suffix = 'val'
            log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)
            # lr
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (epoch % steps_save == steps_save - 1) or is_better:
            d = dict_model if is_better else dict_model.copy()

            # print(f"Best val acc {epoch}: {val_acc}")
            d["epoch"] = epoch + 1
            # metrics
            d["train_loss"] = train_loss
            d["train_acc"] = train_cm.global_accuracy
            d["val_acc"] = val_cm.global_accuracy
            d["val_mcc"] = val_cm.matthews_corrcoef

            name_path = str(list(name_dict.values()))[1:-1].replace(',', '_').replace("'", '').replace(' ', '')
            # name_path = f"{d['val_acc']:.2f}_{name_path}"
            # if periodic save, then include epoch
            if not is_better:
                name_path = f"{name_path}_{epoch + 1}"
            save_model(model, save_path, name_path, param_dicts=d)


def log_confussion_matrix(logger, confussion_matrix: ConfusionMatrix, global_step: int, suffix=''):
    """
    Logs the data in the confussion matrix to a logger
    :param logger: tensorboard logger to use for logging
    :param confussion_matrix: confussion matrix from where the metrics are obtained
    :param global_step: global step for the logger
    """
    logger.add_scalar(f'acc_global_{suffix}', confussion_matrix.global_accuracy, global_step=global_step)
    logger.add_scalar(f'acc_avg_{suffix}', confussion_matrix.average_accuracy, global_step=global_step)
    logger.add_scalar(f'mcc_{suffix}', confussion_matrix.matthews_corrcoef, global_step=global_step)
    for idx, k in enumerate(confussion_matrix.class_accuracy):
        logger.add_scalar(f'acc_class_{idx}_{suffix}', k, global_step=global_step)


def test(
        graph: nx.DiGraph,
        save_path: str = './models/saved',
        n_runs: int = 1,
        batch_size: int = 8,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
        save: bool = True,
        verbose: bool = False,
        balanced_actions:bool = False,
) -> list[dict[str, Union[dict, list[ConfusionMatrix]]]]:
    """
    Calculates the metric on the test set of the model given in args.
    Prints the result and saves it in the dictionary files.

    :param graph: graph for the game
    :param save_path: directory where the model will be saved
    :param n_runs: number of runs from which to take the mean
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param save: whether to save the results in the model dict
    :param verbose: whether to print results
    :param balanced_actions: if true, it will ensure that the actions returned are balanced
    """

    def print_v(s):
        if verbose:
            print(s)

    from pathlib import Path
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print_v(device)
    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # get model names from folder
    model = None
    list_all = []
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
        print_v(f"Testing {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        loader_train, loader_valid, loader_test = load_data(
            graph=graph,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False,
            random_seed=123,
            tokenizer=model.tokenizer,
            device=device,
            balanced_actions=balanced_actions,
            balanced_actions_test=balanced_actions,
        )

        # start testing
        train_cm = []
        train_values_list = []
        val_cm = []
        val_values_list = []
        test_cm = []
        test_values_list = []
        for k in range(n_runs):
            train_run_cm = ConfusionMatrix(size=2, name='train')
            train_values = [[], []]
            val_run_cm = ConfusionMatrix(size=2, name='val')
            val_values = [[], []]
            test_run_cm = ConfusionMatrix(size=2, name='test')
            test_values = [[], []]

            with torch.no_grad():
                # train
                for state, action, reward in loader_train:
                    pred = model(state, action)[:, 0]
                    train_run_cm.add(preds=(pred > 0).float(), labels=reward)
                    train_values[0].extend(reward.cpu().detach().numpy().tolist())
                    train_values[1].extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

                # valid
                for state, action, reward in loader_valid:
                    pred = model(state, action)[:, 0]
                    val_run_cm.add(preds=(pred > 0).float(), labels=reward)
                    val_values[0].extend(reward.cpu().detach().numpy().tolist())
                    val_values[1].extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

                # test
                for state, action, reward in loader_test:
                    pred = model(state, action)[:, 0]
                    test_run_cm.add(preds=(pred > 0).float(), labels=reward)
                    test_values[0].extend(reward.cpu().detach().numpy().tolist())
                    test_values[1].extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

            print_v(f"Run {k}: {test_run_cm.global_accuracy}")

            # Make dataframe calculations for chosen and certainty
            train_df = pd.DataFrame(data=list(zip(*train_values)), columns=['true', 'pred'])
            train_df['chosen'] = (train_df.pred > 0.5).astype(float)
            train_df['cert'] = (train_df.chosen-1+train_df.pred).abs() * 100
            val_df = pd.DataFrame(data=list(zip(*val_values)), columns=['true', 'pred'])
            val_df['chosen'] = (val_df.pred > 0.5).astype(float)
            val_df['cert'] = (val_df.chosen-1+val_df.pred).abs() * 100
            test_df = pd.DataFrame(data=list(zip(*test_values)), columns=['true', 'pred'])
            test_df['chosen'] = (test_df.pred > 0.5).astype(float)
            test_df['cert'] = (test_df.chosen-1+test_df.pred).abs() * 100

            train_cm.append(train_run_cm)
            train_values_list.append(train_df)
            val_cm.append(val_run_cm)
            val_values_list.append(val_df)
            test_cm.append(test_run_cm)
            test_values_list.append(test_df)

        dict_result = {
            "train_mcc": np.mean([k.matthews_corrcoef for k in train_cm]),
            "val_mcc": np.mean([k.matthews_corrcoef for k in val_cm]),
            "test_mcc": np.mean([k.matthews_corrcoef for k in test_cm]),

            "train_acc": np.mean([k.global_accuracy for k in train_cm]),
            "val_acc": np.mean([k.global_accuracy for k in val_cm]),
            "test_acc": np.mean([k.global_accuracy for k in test_cm]),
        }

        print_v(f"RESULT: {dict_result}")

        dict_model.update(dict_result)
        if save:
            save_dict(dict_model, f"{folder_path}/{folder_path.name}.dict")

        list_all.append(dict(
            dict=dict_model,
            train_cm=train_cm,
            val_cm=val_cm,
            test_cm=test_cm,
            train_values = train_values_list,
            val_values = val_values_list,
            test_values = test_values_list,
        ))

    return list_all


def show_examples(
        graph: nx.DiGraph,
        save_path: str = './models/saved',
        num_examples: int = 20,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
) -> None:
    """
    Show examples from the test set

    :param graph: graph for the game
    :param save_path: directory where the model will be saved
    :param num_examples: number of examples to show
    :param num_workers: number of workers (processes) to use for data loading
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    """
    from pathlib import Path
    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print(device)
    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # get model names from folder
    model = None
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
        print(f"Show examples {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        _, _, loader_test = load_data(
            graph=graph,
            num_workers=num_workers,
            batch_size=1,
            drop_last=False,
            random_seed=123,
            tokenize=False,
        )

        # start with examples
        examples = []
        with torch.no_grad():
            for state, action, reward in loader_test:
                # if number of examples reached, break
                if len(examples) >= num_examples:
                    break
                # get predicted percentages
                pred = model.run(state, action, return_percentages=True)[:, 0]
                examples.append((state, action, reward, pred.cpu().detach().numpy()))

        # write in file
        with open(f"{folder_path}/{folder_path.name}.md", "w", encoding="utf-8") as file:
            k = 1
            for state, action, reward, pred in examples:
                file.write(f"# Example {k}\n\n")
                file.write(f"## State\n\n{state}\n\n")
                file.write(f"## Action\n\n{action}\n\n")
                file.write(f"## Response\n\n")
                file.write(f"True value: {reward}\n\nPredicted: {pred}\n\n")
                k += 1


def transform_graph_model(
    graph: nx.DiGraph, 
    model: StateActionModel, 
    save_path: str = "./graph_transformed.pickle",
    use_cpu: bool = False
):
    """
    Runs the given model through the given graph and stores the predictions in it. Then it saves it in a file.
    :param graph: graph over which the model will be run
    :param model: model to use for predictions
    :param save_path: path where graph with the predictions will be saved
    :param use_cpu: whether to use cpu for predictions or cuda
    :return: the graph with the predictions
    """
    # runs the model over all the network and save the corresponding extra
    model = model.eval().to(torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu'))
    for p, n, attr in tqdm(graph.edges(data=True)):
        pred = model.run(
            states=[graph.nodes[p]['text'].strip()],
            actions=[attr['action'].strip()],
            return_percentages=True,
        )[0,...].cpu().detach().tolist()
        graph.edges[(p, n)][GraphSimulator.ATTR_EXTRAS][GraphSimulator.ATTR_PRED] = pred

    # save picle
    save_pickle(graph, save_path)
    return graph


if __name__ == '__main__':
    simulator = load_simulator_yarn()
    model = load_model('./notebooks/saved_bert/125_[30]_[20]_1_False_bert-base-multilingual-cased_0.001_adamw_8_max_val_mcc_False_False_100')[0]
    transform_graph_model(simulator.graph, model, use_cpu=False)
