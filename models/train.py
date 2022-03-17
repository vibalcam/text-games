from os import path
from typing import Dict

import numpy as np
import torch
import torch.utils.tensorboard as tb
from tqdm.auto import trange

from .models import StateActionModel, save_model, load_model
from .utils import ConfusionMatrix, load_data, save_dict


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
        balanced_actions: bool = False,
):
    """
    Method that trains a given model

    :param model: model that will be trained
    :param dict_model: dictionary of model parameters
    :param log_dir: directory where the tensorboard log should be saved
    :param data_path: directory where the data can be found
    :param save_path: directory where the model will be saved
    :param lr: learning rate for the training
    :param optimizer_name: optimizer used for training. Can be `adam, adamw, sgd`
    :param n_epochs: number of epochs of training
    :param batch_size: size of batches to use
    :param num_workers: number of workers (processes) to use for data loading
    :param scheduler_mode: scheduler mode to use for the learning rate scheduler. Can be `min_loss, max_acc, max_val_acc, max_val_mcc`
    :param use_cpu: whether to use the CPU for training
    :param debug_mode: whether to use debug mode (cpu and 0 workers)
    :param steps_validate: number of epoch after which to validate and save model (if conditions met)
    :param freeze_bert: whether to freeze BERT during training
    :param balanced_actions: parameter for `utils.StateActionDataset`
    """

    # cpu or gpu used for training if available (gpu much faster)
    device = torch.device('cuda' if torch.cuda.is_available() and not (use_cpu or debug_mode) else 'cpu')
    print(device)

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
    ]}
    # dict_param.update(dict(
    #     train_self_loop=dataset.add_self_loop,
    #     train_drop_edges=dataset.drop_edges,
    # ))
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
    dict_model.update(dict(
        # metrics
        train_loss=None,
        train_acc=0,
        val_acc=0,
        epoch=0,
    ))
    model = model.to(device)

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
        balanced_actions=balanced_actions,
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10 if not balanced_actions else 480)
    elif scheduler_mode in ["max_acc", "max_val_acc", "max_val_mcc"]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10 if not balanced_actions else 480)
    else:
        raise Exception("Optimizer not configured")

    # print(f"{log_dir}/{name_model}")
    for epoch in (p_bar := trange(n_epochs)):
        p_bar.set_description(f"{name_model} -> {dict_model['val_acc']}")
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
            train_cm.add(preds=pred, labels=reward)

        # Evaluate the model
        val_cm = ConfusionMatrix(size=2, name='val')
        model.eval()
        with torch.no_grad():
            for state, action, reward in loader_valid:
                pred = model(state, action)[:, 0]
                val_cm.add(pred, reward)

        train_loss = np.mean(train_loss)
        train_acc = train_cm.global_accuracy
        val_acc = val_cm.global_accuracy

        # Step the scheduler to change the learning rate
        if scheduler_mode == "min_loss":
            scheduler.step(train_loss)
        elif scheduler_mode == "max_acc":
            scheduler.step(train_acc)
        elif scheduler_mode == "max_val_acc":
            scheduler.step(val_acc)
        elif scheduler_mode == 'max_val_mcc':
            scheduler.step(val_cm.matthews_corrcoef)

        global_step += 1
        if train_logger is not None:
            suffix = 'train'
            train_logger.add_scalar(f'loss_{suffix}', train_loss, global_step=global_step)
            log_confussion_matrix(train_logger, train_cm, global_step, suffix=suffix)
            # validation log
            suffix = 'val'
            log_confussion_matrix(valid_logger, val_cm, global_step, suffix=suffix)
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        # Save the model
        if (epoch % steps_validate == steps_validate - 1) and (True or (val_acc >= dict_model["val_acc"])):
            # print(f"Best val acc {epoch}: {val_acc}")
            dict_model["train_loss"] = train_loss
            dict_model["train_acc"] = train_acc
            dict_model["val_acc"] = val_acc
            dict_model["epoch"] = epoch
            name_path = name_model.replace('/', '_')
            save_model(model, save_path, name_path, param_dicts=dict_model)


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
        data_path: str = './yarnScripts',
        save_path: str = './models/saved',
        n_runs: int = 1,
        batch_size: int = 8,
        num_workers: int = 0,
        debug_mode: bool = False,
        use_cpu: bool = False,
        save: bool = True,
        verbose: bool = False,
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
    :param verbose: whether to print results
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
    best_dict = None
    best_acc = 0.0
    list_all = []
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
        print_v(f"Testing {folder_path.name}")

        # load model and data loader
        del model
        model, dict_model = load_model(folder_path)
        model = model.to(device).eval()
        loader_train, loader_valid, loader_test = load_data(
            dataset_path=data_path,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False,
            random_seed=123,
            tokenizer=model.tokenizer,
            device=device,
        )

        # start testing
        train_cm = []
        val_cm = []
        test_cm = []
        for k in range(n_runs):
            train_run_cm = ConfusionMatrix(size=2, name='train')
            val_run_cm = ConfusionMatrix(size=2, name='val')
            test_run_cm = ConfusionMatrix(size=2, name='test')

            with torch.no_grad():
                # train
                for state, action, reward in loader_train:
                    pred = model(state, action)[:, 0]
                    train_run_cm.add(preds=pred, labels=reward)

                # valid
                for state, action, reward in loader_valid:
                    pred = model(state, action)[:, 0]
                    val_run_cm.add(preds=pred, labels=reward)

                # test
                for state, action, reward in loader_test:
                    pred = model(state, action)[:, 0]
                    test_run_cm.add(preds=pred, labels=reward)

            print_v(f"Run {k}: {test_run_cm.global_accuracy}")
            
            train_cm.append(train_run_cm)
            val_cm.append(val_run_cm)
            test_cm.append(test_run_cm)

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
        ))

        # save if best
        if best_acc < (test_acc := dict_model['test_acc']):
            best_acc = test_acc
            best_dict = dict_model

    return best_dict, best_acc, list_all


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
    paths = list(Path(save_path).glob('*'))
    for folder_path in tqdm(paths):
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

    args_parser.add_argument('-t', '--test', type=int, default=None,
                             help='the number of test runs that will be averaged to give the test result,'
                                  'if None, training mode')
    args_parser.add_argument('-ex', '--show_examples', type=int, default=None)

    args = args_parser.parse_args()

    if args.test is not None:
        test(
            n_runs=args.test,
            save_path="./models/saved_good"
        )
    elif args.show_examples is not None:
        show_examples(num_examples=args.show_examples)
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
            n_epochs=30,
            batch_size=8,
            num_workers=0,
            scheduler_mode='max_val_acc',
            debug_mode=False,
            steps_validate=1,
            use_cpu=False,
            freeze_bert=True,
            balanced_actions=True,
        )





        # # Model
        # bert_dict_model = dict(
        #     shared_out_dim=125,
        #     state_layers=[30],
        #     action_layers=[30],
        #     out_features=1,
        #     lstm_model=False,
        #     bert_name="bert-base-multilingual-cased",
        # )
        # # lstm_dict_model = dict(
        # #     shared_out_dim=50,
        # #     state_layers=[30],
        # #     action_layers=[30],
        # #     out_features=1,
        # #     lstm_model=True,
        # #     bert_name="bert-base-multilingual-cased",
        # # )
        # dict_model = bert_dict_model
        # model = StateActionModel(**dict_model)

        # # Training hyperparameters
        # train(
        #     model=model,
        #     dict_model=dict_model,
        #     log_dir='./models/logs',
        #     data_path='./yarnScripts',
        #     save_path='./models/saved',
        #     lr=1e-3,
        #     optimizer_name="adamw",
        #     n_epochs=30,
        #     batch_size=8,
        #     num_workers=0,
        #     scheduler_mode='max_val_acc',
        #     debug_mode=False,
        #     steps_validate=1,
        #     use_cpu=False,
        #     freeze_bert=True,
        #     balanced_actions=True,
        # )


        # # other
        # # Model
        # bert_dict_model = dict(
        #     shared_out_dim=125,
        #     state_layers=[20],
        #     action_layers=[20],
        #     out_features=1,
        #     lstm_model=False,
        #     bert_name="bert-base-multilingual-cased",
        # )
        # # lstm_dict_model = dict(
        # #     shared_out_dim=50,
        # #     state_layers=[30],
        # #     action_layers=[30],
        # #     out_features=1,
        # #     lstm_model=True,
        # #     bert_name="bert-base-multilingual-cased",
        # # )
        # dict_model = bert_dict_model
        # model = StateActionModel(**dict_model)

        # # Training hyperparameters
        # train(
        #     model=model,
        #     dict_model=dict_model,
        #     log_dir='./models/logs2',
        #     data_path='./yarnScripts',
        #     save_path='./models/saved2',
        #     lr=1e-3,
        #     optimizer_name="adamw",
        #     n_epochs=30,
        #     batch_size=8,
        #     num_workers=0,
        #     scheduler_mode='max_val_acc',
        #     debug_mode=False,
        #     steps_validate=1,
        #     use_cpu=False,
        #     freeze_bert=True,
        #     balanced_actions=True,
        # )