# vmm

This project is work in progress:

- [x] Add data preprocessing module
- [x] Add dataloader module
- [x] Add model module
- [ ] Add main.py

This project contains code to create an adapted classifier for classifying borehole descriptions into textures, admixtures, and colors. 
It leverages [GEOBERTje](https://github.com/VITObelgium/geobertje), a language model trained on geological borehole descriptions.

## Setting Up Project

This project uses [Hatch](https://hatch.pypa.io/latest/) for environment management and dependency installation. To get started, install Hatch:

```
pip install hatch
```

Then, create the environment:

```
hatch env create
```

For more details, refer to the [Hatch documentation](https://hatch.pypa.io/latest/).



## License

`vmm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
