from os import path
from typing import Dict

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import StateActionModel, save_model, load_model
from .utils import load_data, accuracy, save_dict


def train(
        model: StateActionModel,
        dict_model: Dict,
        log_dir: str = './models/logs',
        data_path: str = './yarnScripts',
        save_path: str = './models/saved',
        lr: float = 1e-3,
        optimizer_name: str = "adamw",
        n_epochs: int = 100,
        batch_size: int = 8,
        num_workers: int = 0,
        scheduler_mode: str = 'max_val_acc',
        debug_mode: bool = False,
        steps_validate: int = 1,
        use_cpu: bool = False,
        freeze_bert: bool = True,

        # # Parameters of the model to use (look at model for more detail)
        # shared_out_dim: int = 50,
        # state_layers: List[int] = [25],
        # action_layers: List[int] = [25],
        # lstm_model = True,
        # out_features: int = 1,
        # bert_name = "bert-base-multilingual-cased",
        # freeze_bert: bool = False,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param log_dir: directory where the tensorboard log should be saved
    :param data_path: directory where the data can be found
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be adam, adamw, sgd
    :param n_epochs: number of epochs of training
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be min_loss, max_acc, max_val_acc
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param steps_validate: number of epoch after which to validate and save model (if conditions met)
    :param freeze_bert: whether to freeze BERT during training
    """

    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print(device)

    # num_workers 0 if debug_mode
    if debug_mode:
        num_workers = 0

    # Number of epoch after which to validate and save model
    # steps_validate = 1

    # Hyperparameters

    # # learning rates
    # lr: int = args.lr
    # # lr: int = 1e-2
    # # optimizer to use for training
    # optimizer_name: str = "adamw"  # adam, adamw, sgd
    # # number of epochs to train on
    # n_epochs: int = args.n_epochs
    # # size of batches to use
    # # batch_size: int = args.batch_size
    # batch_size: int = 8
    # # number of workers (processes) to use for data loading
    # num_workers: int = 0 if args.debug else args.num_workers
    # # dimensions of the model to use (look at model for more detail)
    # # shared_out_dim:int = 125
    # # state_layers:List[int] = [20]
    # # action_layers:List[int] = [20]
    # shared_out_dim: int = 50
    # state_layers: List[int] = [25]
    # action_layers: List[int] = [25]
    # lstm_model = True
    # # output features
    # out_features: int = 1
    # # scheduler mode to use for the learning rate scheduler
    # scheduler_mode: str = 'max_val_acc'  # min_loss, max_acc, max_val_acc
    # # Name of the BERT model to use
    # bert_name = "bert-base-multilingual-cased"
    # freeze_bert: bool = False

    # # Tensorboard
    # global_step = 0
    # name_model = f"{optimizer_name}/{scheduler_mode}/{batch_size}/{lstm_model}/" \
    #              f"{shared_out_dim},{state_layers},{action_layers}/{lr}/{bert_name}/{freeze_bert}"
    # train_logger = tb.SummaryWriter(path.join(log_dir, 'train', name_model), flush_secs=1)
    # valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid', name_model), flush_secs=1)

    # Tensorboard
    global_step = 0
    name_model = '/'.join([
        str(dict_model)[1:-1].replace(',', '/').replace("'", '').replace(' ', '').replace(':', '='),
        str(lr),
        optimizer_name,
        str(batch_size),
        scheduler_mode,
    ])
    train_logger = tb.SummaryWriter(path.join(log_dir, 'train', name_model), flush_secs=1)
    valid_logger = tb.SummaryWriter(path.join(log_dir, 'valid', name_model), flush_secs=1)

    # Model
    dict_model.update(dict(
        name=name_model,
        # metrics
        train_loss=None,
        train_acc=0,
        val_acc=0,
        epoch=0,
    ))
    model = model.to(device)
    # dict_model = dict(
    #     # dictionary with model information
    #     name=name_model,
    #     shared_out_dim=shared_out_dim,
    #     state_layers=state_layers,
    #     action_layers=action_layers,
    #     out_features=out_features,
    #     lstm_model=lstm_model,
    #     bert_name=bert_name,
    #     # metrics
    #     train_loss=None,
    #     train_acc=0,
    #     val_acc=0,
    #     epoch=0,
    # )
    # model = StateActionModel(**dict_model).to(device)

    # Loss
    loss = torch.nn.BCEWithLogitsLoss().to(device)  # sigmoid + BCELoss (good for 2 classes classification)

    # load train and test data
    loader_train, loader_valid, _ = load_data(
        dataset_path=data_path,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=False,
        random_seed=123,
        tokenizer=model.tokenizer,
        device=device,
    )

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise Exception("Optimizer not configured")

    if scheduler_mode == "min_loss":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    elif scheduler_mode in ["max_acc", "max_val_acc"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    else:
        raise Exception("Optimizer not configured")

    print(f"{log_dir}/{name_model}")

    for epoch in range(n_epochs):
        print(epoch)
        train_loss = []
        train_acc = []

        # Start training: train mode and freeze bert
        model.train()
        model.freeze_bert(freeze_bert)
        for state, action, reward in loader_train:
            # To device
            # state, action, reward = state.to(device), action.to(device), reward.to(device)

            # Compute loss and update parameters
            pred = model(state, action)[:, 0]
            loss_val = loss(pred, reward)

            # Do back propagation
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Add train loss and accuracy
            train_loss.append(loss_val.cpu().detach().numpy())
            train_acc.append(accuracy(pred, reward))

        # Evaluate the model
        val_acc = []
        model.eval()
        with torch.no_grad():
            for state, action, reward in loader_valid:
                # To device
                # state, action, reward = state.to(device), action.to(device), reward.to(device)
                pred = model(state, action)[:, 0]
                val_acc.append(accuracy(pred, reward))

        train_loss = np.mean(train_loss)
        train_acc = np.mean(train_acc)
        val_acc = np.mean(val_acc)

        # Step the scheduler to change the learning rate
        if scheduler_mode == "min_loss":
            scheduler.step(train_loss)
        elif scheduler_mode == "max_acc":
            scheduler.step(train_acc)
        elif scheduler_mode == "max_val_acc":
            scheduler.step(val_acc)

        global_step += 1
        if train_logger is not None:
            # train_logger.add_text(model, img)
            train_logger.add_scalar('loss', train_loss, global_step=global_step)
            train_logger.add_scalar('acc', train_acc, global_step=global_step)
            valid_logger.add_scalar('acc', val_acc, global_step=global_step)
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (epoch % steps_validate == steps_validate - 1) and (val_acc >= dict_model["val_acc"]):
            print(f"Best val acc {epoch}: {val_acc}")
            dict_model["train_loss"] = train_loss
            dict_model["train_acc"] = train_acc
            dict_model["val_acc"] = val_acc
            dict_model["epoch"] = epoch
            name_path = name_model.replace('/', '_')
            save_model(model, save_path, name_path, param_dicts=dict_model)


def test(
        data_path: str = './yarnScripts',
        save_path: str = './models/saved',
        n_runs: int = 3,
        batch_size: int = 8,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
        save: bool = True,
) -> None:
    """
    Calculates the metric on the test set of the model given in args.
    Prints the result and saves it in the dictionary files.

    :param data_path: directory where the data can be found
    :param save_path: directory where the model will be saved
    :param n_runs: number of runs from which to take the mean
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param save: whether to save the results in the model dict
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
    for folder_path in Path(save_path).glob('*'):
        print(f"Testing {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        _, _, loader_test = load_data(
            dataset_path=data_path,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False,
            random_seed=123,
            tokenizer=model.tokenizer,
            device=device,
        )

        # start testing
        test_acc = []
        for k in range(n_runs):
            run_acc = []

            with torch.no_grad():
                for state, action, reward in loader_test:
                    pred = model(state, action)[:, 0]
                    run_acc.append(accuracy(pred, reward))

            run_acc = np.mean(run_acc)
            print(f"Run {k}: {run_acc}")
            test_acc.append(run_acc)

        test_acc = np.mean(test_acc)
        dict_result = {"test_acc": test_acc}

        print(f"{folder_path.name}: {dict_result}")
        dict_model.update(dict_result)
        if save:
            save_dict(dict_model, f"{folder_path}/{folder_path.name}.dict")


def show_examples(
        data_path: str = './yarnScripts',
        save_path: str = './models/saved',
        num_examples: int = 20,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
) -> None:
    """
    Show examples from the test set

    :param data_path: directory where the data can be found
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
    for folder_path in Path(save_path).glob('*'):
        print(f"Show examples {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        _, _, loader_test = load_data(
            dataset_path=data_path,
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
                pred = model.run(state, action, return_percentages=True, device=device)[:, 0]
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


if __name__ == '__main__':
    from argparse import ArgumentParser

    args_parser = ArgumentParser()

    # args_parser.add_argument('--log_dir', default="./models/logs")
    # args_parser.add_argument('--data_path', default="./yarnScripts")
    # args_parser.add_argument('--save_path', default="./models/saved")
    args_parser.add_argument('-t', '--test', type=int, default=None,
                             help='the number of test runs that will be averaged to give the test result,'
                                  'if None, training mode')
    args_parser.add_argument('-ex', '--show_examples', type=int, default=None)

    # Hyper-parameters
    # args_parser.add_argument('-lr', type=float, default=1e-3, help='learning rates')
    # args_parser.add_argument('-law', '--loss_age_weight', nargs='+', type=float, default=[1e-2],
    #                          help='weight for the age loss')
    # args_parser.add_argument('-opt', '--optimizers', type=str, nargs='+', default=["adam"], help='optimizer to use')
    # args_parser.add_argument('-n', '--n_epochs', default=100, type=int, help='number of epochs to train on')
    # args_parser.add_argument('-b', '--batch_size', default=8, type=int, help='size of batches to use')
    # args_parser.add_argument('-w', '--num_workers', default=2, type=int,
    #                          help='number of workers to use for data loading')

    # args_parser.add_argument('--cpu', action='store_true')
    # args_parser.add_argument('-d', '--debug', action='store_true')

    args = args_parser.parse_args()

    if args.test is not None:
        test()
    elif args.show_examples is not None:
        show_examples()
    else:
        # Model
        bert_dict_model = dict(
            shared_out_dim=125,
            state_layers=[20],
            action_layers=[20],
            out_features=1,
            lstm_model=False,
            bert_name="bert-base-multilingual-cased",
        )
        # lstm_dict_model = dict(
        #     shared_out_dim=50,
        #     state_layers=[30],
        #     action_layers=[30],
        #     out_features=1,
        #     lstm_model=True,
        #     bert_name="bert-base-multilingual-cased",
        # )
        dict_model = bert_dict_model
        model = StateActionModel(**dict_model)

        # Training hyperparameters
        train(
            model=model,
            dict_model=dict_model,
            log_dir='./models/logs',
            data_path='./yarnScripts',
            save_path='./models/saved',
            lr=1e-3,
            optimizer_name="adamw",
            n_epochs=100,
            batch_size=8,
            num_workers=0,
            scheduler_mode='max_val_acc',
            debug_mode=False,
            steps_validate=1,
            use_cpu=False,
            freeze_bert=True,
        )
