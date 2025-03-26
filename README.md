# vmm

This project contains code to create an adapted classifier for classifying borehole descriptions into textures, admixtures, and colors. 
It leverages [GEOBERTje](https://github.com/VITObelgium/geobertje), a language model trained on geological borehole descriptions.

The code consists of:

- A preprocessing module (`data/dataloader.py`): This module loads and preprocesses the lithology data.
- A dataloader module (`data/dataloader.py`): This module creates `torch` dataloaders for the lithology data, split into train, validation, and test sets.
    Moreover, it tokenizes the text data using the `GEOBERTje` tokenizer and obtains the embeddings for the tokens. By doing this in advance, we can speed up the training process.
- A model module (`models/independent.py`): This module contains the model architecture, which consists of 2-layered MLP classifier.
- A config module (`configs/`): This module contains the configuration files for the data and model.
- A main module (`main.py`): This module contains the main code to load the data, train the model, and evaluate the model.
  
## Setting Up Project

This project uses [Hatch](https://hatch.pypa.io/latest/) for environment management and dependency installation. To get started, install Hatch:

```
pip install hatch
```

Then, create the environment:

```
hatch env create
```

Activate the environment:

```bash
hatch shell
```

For more details, refer to the [Hatch documentation](https://hatch.pypa.io/latest/).

To register a kernel in a notebook using Hatch, you can activate your Hatch shell and spin up a notebook server which you can access through your browser or your favorite editor.

## Running the Code

### Configuration

Ensure you have the configuration files `data_config.yaml` and `model_config.yaml` in the `vmm/configs` directory. 
These files should contain the necessary configuration for the data and model.

### Run the code

The loading and preprocessing of the lithology data, as well as the model training and evaluation can be done through:

```bash
python vmm/main.py # run from vmm/src
```

If desired, the necessary modules can also be loaded into a notebook to run interactively.

## Logging

The project uses Python's built-in logging module to log messages. Logs will be displayed in the terminal when you run the code.



## License

`vmm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
