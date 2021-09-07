# PyTorch Dynamo

This is an early experiment into using [PEP 523] to expose fusion
opportunities in PyTorch.  It dynamically rewrites Python bytecode in
order to extract sequences of PyTorch operations into an [FX Graph]
which is just in time compiled with a user-defined compiler.  It creates
this FX Graph through bytecode analysis, and is designed to generating
smaller graph fragments that can be mixed with Python execution.

The name is a reference/homage to [DynamoRIO], which dynamically translates
machine code.

![](TorchDynamo.png)

[PEP 523]: https://www.python.org/dev/peps/pep-0523/
[FX Graph]: https://pytorch.org/docs/stable/fx.html
[DynamoRIO]: https://dynamorio.org/


## Development Setup

Initial setup
```
git clone git@github.com:pytorch/benchmark.git torchbenchmark
cd torchbenchmark
./scripts/recreate_conda_environment.sh
cd ..

git clone git@github.com:jansel/torchdynamo.git
cd torchdynamo
conda activate torchbenchmark
python setup.py develop  # compiles C/C++ extension
pytest  # run tests
```

Run tests with:
```
conda activate torchbenchmark  # if not activated already
python setup.py develop && pytest
```


Run torchbench models with:
```
conda activate torchbenchmark  # if not activated already
python setup.py develop && python torchbench.py
```

