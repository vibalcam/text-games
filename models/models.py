import pathlib
from typing import List, Dict, Tuple, Optional, Union, overload

import torch
from overrides import overrides
from torch import device, dtype, Tensor
from torch.nn.modules.module import T
from transformers import BertConfig, BertTokenizerFast, BertModel

from helper.helper import load_dict, save_dict


class StateActionModel(torch.nn.Module):
    """
    Model to be used to extract the personality traits labels given a state and action
    """

    class LSTMSharedBlock(torch.nn.Module):
        """
        Shared block of the neural net
        """

        class BertLSTM(torch.nn.LSTM):
            """
            LSTM wrapper to use in Bert Model
            """

            def forward(self, x, *args, **kwargs):
                return super().forward(x)

        def __init__(
                self,
                out_features: int,
                sequence_length: int,
                bert_name: str = "bert-base-multilingual-cased",
                num_layers: int = 2,
                dropout: float = 0,
                bidirectional: bool = True,
                hidden_size: int = 768,
                download_bert:bool = True,
        ):
            """
            Initialization of a block that composes the classifier

            :param out_features: number of output features
            :param sequence_length: length of the input sequence
            :param bert_name: name of bert model from HuggingFace
            :param num_layers: number of layers of LSTM
            :param dropout: dropout for LSTM
            :param bidirectional: whether the LSTM should be bidirectional
            :param hidden_size: hidden size of LSTM
            """
            super().__init__()

            self.net = BertModel.from_pretrained(bert_name) if download_bert else BertModel(BertConfig.from_pretrained(bert_name))

            self.net.encoder = self.BertLSTM(
                input_size=self.net.config.hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )

            self.net.pooler = torch.nn.Sequential(
                # todo average pooling instead of flatten, this also allows any input length
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=(sequence_length * hidden_size * (2 if bidirectional else 1)),
                    out_features=out_features,
                    bias=True,
                ),
                torch.nn.Tanh(),
            )

        def forward(self, x):
            """
            Method which runs the given input through the block
            :param x: dict of torch.Tensor(B,sequence_length), output of the tokenizer
            :return: torch.Tensor(B,out_features)
            """
            return self.net(**x, return_dict=False)[1]

    class BertSharedBlock(torch.nn.Module):
        """
        Shared block of the neural net
        """

        def __init__(self, out_features: int, bert_name="bert-base-multilingual-cased",
                     hidden_post_bert: Optional[int] = None, download_bert:bool = True):
            """
            Initialization of a block that composes the classifier
            :param out_features: number of output features
            :param bert_name: name of bert model from HuggingFace
            :param hidden_post_bert: hidden size of linear layer after BERT encoder model
            """
            super().__init__()

            self.bert = BertModel.from_pretrained(bert_name) if download_bert else BertModel(BertConfig.from_pretrained(bert_name))

            # Can also use dropout
            # num_extra_layers = max(num_layers - 2, 0)
            bert_hidden = self.bert.config.hidden_size
            layers = [
                torch.nn.Linear(bert_hidden, bert_hidden if hidden_post_bert is None else hidden_post_bert),
                torch.nn.ReLU(),
                torch.nn.Linear(bert_hidden if hidden_post_bert is None else hidden_post_bert, out_features)
            ]

            self.net = torch.nn.Sequential(*layers)

        def freeze_bert(self, freeze):
            """
            Used to freeze the BERT model in order to fine-tune it
            :param freeze: whether to freeze the BERT encoder
            """
            for param in self.bert.parameters():
                param.requires_grad = not freeze

        def forward(self, x):
            """
            Method which runs the given input through the block
            :param x: dict of torch.Tensor(B,sequence_length), output of the tokenizer
            :return: torch.Tensor(B,out_features)
            """
            # z = self.bert(**x)
            z = self.bert(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'],
            attention_mask=x['attention_mask'])
            return self.net(z['pooler_output'])

    class IndividualBlock(torch.nn.Module):
        """
        Block of the neural net for individual layers
        """

        def __init__(self, in_size: int, dim_layers: List[int]):
            """
            Initialization of a block that composes the classifier
            :param in_size: input size
            :param dim_layers: output size of linear layers
            """
            super().__init__()

            # Can also use dropout
            k = dim_layers[0]
            layers = [
                torch.nn.Linear(in_size, k),
            ]
            for out_size in dim_layers[1:]:
                layers.extend([
                    torch.nn.ReLU(),
                    # torch.nn.Dropout(0.1),
                    torch.nn.Linear(k, out_size),
                ])
                k = out_size

            self.net = torch.nn.Sequential(*layers)

        def forward(self, x: torch.Tensor):
            """
            Method which runs the given input through the block
            :param x: torch.Tensor(B,in_size)
            :return: torch.Tensor(B,dim_layers[-1])
            """
            return self.net(x)

    def __init__(
            self,
            shared_out_dim: int = 512,
            state_layers: List[int] = [255, 125],
            action_layers: List[int] = [255, 125],
            out_features: int = 1,
            lstm_model: bool = False,
            bert_name: str = "bert-base-multilingual-cased",
            max_sequence_length: Optional[int] = None,
            download_bert:bool = True,
            # agg_method:str='sum',
            **kwargs
    ):
        """
        Initialization of the classifier
        :param shared_out_dim: out dimension of shared layer
        :param state_layers: layer dimensions of state individual block
        :param action_layers: layer dimensions of action individual block
        :param out_features: output features
        :param lstm_model: whether to use LSTM model
        :param bert_name: name of bert model from HuggingFace
        :param max_sequence_length: maximum sequence length
        """
        super().__init__()
        self.device = torch.device('cpu')

        # tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_name, padding_side='left')

        # max_sequence_length
        self.max_sequence_length = max_sequence_length
        if max_sequence_length is None or max_sequence_length > self.tokenizer.model_max_length:
            self.max_sequence_length = self.tokenizer.model_max_length

        # shared block
        self.lstm_model = lstm_model
        if not lstm_model:
            self.shared = self.BertSharedBlock(shared_out_dim, bert_name=bert_name, download_bert=download_bert)
        else:
            self.shared = self.LSTMSharedBlock(
                out_features=shared_out_dim,
                sequence_length=self.max_sequence_length,
                bert_name=bert_name,
                num_layers=2,
                dropout=0.1,
                bidirectional=False,
                hidden_size=512,
                download_bert=download_bert,
            )
        # block for state
        self.state = self.IndividualBlock(shared_out_dim, state_layers)
        # block for action
        self.action = self.IndividualBlock(shared_out_dim, action_layers)
        # combine state and action, and output
        # todo allow for sum, not only concatenate
        self.output = torch.nn.Linear(state_layers[-1] + action_layers[-1], out_features)

    def freeze_bert(self, freeze):
        """
        Used to freeze the BERT model in order to fine-tune it.
        Only freeze for bert model, not lstm model
        :param freeze: whether to freeze the BERT encoder
        """
        if not self.lstm_model:
            self.shared.freeze_bert(freeze)

    @overrides(check_signature=False)
    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def predict(self, states: List[str], actions: List[str], threshold: float = 0.5) -> torch.Tensor:
        """
        Predicts the output given the states and actions
        :param states: list of states
        :param actions: list of actions
        :param device: torch.device to use
        :param threshold: level of certainty to output a 1 (if result > threshold -> 1, otherwise 0)
        :return: torch tensor with the predictions
        """
        return (self.run(states, actions, return_percentages=True) > threshold).float()

    def run(self, states: List[str], actions: List[str], return_percentages: bool = False) -> torch.Tensor:
        """
        Runs the model given a list of states and actions and returns the output of the model
        (before sigmoid, THESE VALUES ARE NOT PERCENTAGES).
        :param states: list of states
        :param actions: list of actions
        :param device: torch.device to use
        :param return_percentages: whether it should return percentages (using a sigmoid) or raw outputs
        :return: torch tensor with the output of the model
        """
        states = self.tokenizer([k[-self.max_sequence_length:].split('\n', 1)[-1] for k in states],
                                return_tensors='pt', padding='max_length', max_length=self.max_sequence_length)
        actions = self.tokenizer([k[-self.max_sequence_length:].split('\n', 1)[-1] for k in actions],
                                 return_tensors='pt', padding='max_length', max_length=self.max_sequence_length)
        states = {k: v.to(self.device) for k, v in states.items()}
        actions = {k: v.to(self.device) for k, v in actions.items()}
        return self(states, actions) if not return_percentages else torch.sigmoid(self(states, actions))

    def forward(self, state: Dict[str, torch.Tensor], action: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Method which runs the given input through the classifier
        :param action: dict of torch.Tensor(B,input_size), output of the tokenizer
        :param state: dict of torch.Tensor(B,input_size), output of the tokenizer
        :return: torch.Tensor(B, out_features)
        """
        state = self.state(self.shared(state))
        action = self.action(self.shared(action))
        return self.output(torch.concat((state, action), 1))


FOLDER_PATH_KEY = 'path_name'


def save_model(model: torch.nn.Module, folder: str, model_name: str, param_dicts: Dict = None) -> None:
    """
    Saves the model so it can be loaded after
    :param model_name: name of the model to be saved (non including extension)
    :param folder: path of the folder where to save the model
    :param param_dicts: dictionary of the model parameters that can later be used to load it
    :param model: model to be saved
    """
    # create folder if it does not exist
    folder_path = f"{folder}/{model_name}"
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), f"{folder_path}/{model_name}.th")
    # save model using TorchScript
    # model_scripted = torch.jit.script(model) # Export to TorchScript
    # model_scripted.save(f"{folder_path}/{model_name}.th") # Save

    # save dict
    if param_dicts is not None:
        save_dict(param_dicts, f"{folder_path}/{model_name}.dict")


def load_model(folder_path: Union[pathlib.Path,str]) -> Tuple[StateActionModel, Dict]:
    """
    Loads a model that has been previously saved using its name (model th and dict must have that same name)
    Only works for StateActionModel
    :param folder_path: folder path of the model to be loaded
    :return: the loaded model and the dictionary of parameters
    """
    if isinstance(folder_path, str):
        folder_path = pathlib.Path(folder_path)
    path = f"{folder_path.absolute()}/{folder_path.name}"
    # load dict
    dict_model = load_dict(f"{path}.dict")

    # set folder path
    dict_model[FOLDER_PATH_KEY] = str(folder_path)

    model = StateActionModel(download_bert=False, **dict_model)
    model.load_state_dict(torch.load(f"{path}.th", map_location='cpu'))
    
    # load model using TorchScript
    # model = torch.jit.load(f"{path}.th")
    return model, dict_model
