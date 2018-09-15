# Torch Baker

A wrapper for PyTorch that use chain-like syntax to custom training/testing process

# Requirements

This project is written and tested using `python@3.6.5` and `PyTorch@0.4.0`. Earlier versions might work, but I haven't try it.

# Installation

download the project and run
```
python setup.py install
```

# Usage

Customize the Recipe and send it to a Baker.

```python
import torchbaker as tb

class MyRecipe(tb.Recipe):
    # customize settings
    pass

if __name__ == '__main__':
    baker = tb.Baker()
    baker.prepare_recipe(MyRecipe())
    baker.train()
    baker.test()
```

Baker can run the training/testing loop. How the network is forwarded and updated is specified in the recipe.

## Pass

Baker abstracts and divides training/testing processes into passes. A pass tells how data is forwarded, the criterion, and optimzers used to update model parameters. For example, if we want to end to end train a network using MSELoss.

```python
class MyRecipe(tb.Recipe):
    def __init__(self):
        net = Net()
        self.modules = {
            'net': net,
            'mse': MSELoss()
        }
        self.optimizers = {
            'adam': Adam(net.parameters(), lr=1e-3)
        }
        self.dataloaders = {
            'train': MyDataLoader()
        }
        self.train_passes = [
            tb.Pass('''
                [in] => [data, target] # unpack input
                [data] => net => [pred]
                [pred, target] => mse => [loss]
            ''', lambda: self.variables.loss, optimizers=['adam'])
        ]
```

Not all passes need to be run on every iter. `torchbaker.Pass` offers a parameter `condition`, which accepts a method that returns a `bool`, checked before every time the Pass is executed. Passes will be executed sequentially on every iter, iff its condition returns `True`. `self.train_passes` will be used during training, and `self.test_passes` is for testing.

Some pre-defined conditions can be used. If a pass should run on every 5 iters, the code below will work:

```python
# This Pass will be called every 5 iters
tb.Pass('[in] => net => [o]', condition=self.true_every_iter(5))
```

Sometimes one may just want to perfrom a forward, without backward and parameter update. At that time, one can just write

```python
self.train_passes = [
    '[in] => net => [o]'
]
# this is the same as
self.train_passes = [
    tb.Pass('[in] => net => [o]')
]
```

i.e., strings in the passes list will be wrapped as Passes automatically.

Baker embeds `CEL` (Chained Elements Lines), a mini-language that defines data flows for Passes. Its syntax is simple.

+ Everything behind `#` is comment
+ `[]` means variable pack. Variables inside the pack is seperated by `,`. Auto unpacking is supported. Values of variables can be accessed in Python code from property `variables`, using dot or square brackets. For example `self.variables.loss` or `self.variables['loss']`
+ `[in]` is a built in variable, which holds the input data from train/test data loader. `self.dataloaders['train']` will be used for training, while `self.dataloaders['test']` will be used for testing.
+ `=>` and `->` passes data on to the next element. `->` will try to perform detach operations on data, which means this symbol may block gradient backward.
+ other elements are modules. they will be looked up from `self.modules`. Modules can be **ANY** callables, no matter it inherits `torch.nn.Module` or not.

`CEL` becomes handy when complex training process is involved.

```
# a meaningless example
[in] => net_1 => [t1] => net_2 => [t2] => net_3 => [t3]
[t1, t2, t3] -> net_4 => [o1, o2, o3]
[in, t1, t2, t3, o1, o2, o3] => my_criterion => [loss]
```

## Callbacks

`CEL` can simplify the configuration for training and testing, and it can also be used for doing something other than passing data. For example:

```python
def visualize(*args):
    # do some visualize stuff
    return args

# in the Recipe
self.modules = {
    # ...
    'visualize': visualize,  # register modules
    # ...
}

self.train_passes = [
    '# Other passes',
    ...,
    '[data] -> visualize'  # let's visualize it!
]
```

Of course, this is legal. As I metioned above, modules can be any callables, but it's not that sounding semantically, and one needs register new modules every time she wants to add a new function. A better solution is offered, which is using callbacks.

```python
@self.register_callback('train_iter_finish')
def visualize():
    # do some visualize stuff
    pass
```

To visualize on every 200 iters

```python
@self.register_callback('train_iter_finish')
@self.do_every_iter(200)
def visualize():
    # do some visualize stuff
    pass
```

Also visualize when train finish

```python
@self.register_callback('train_iter_finish')
@self.do_every_iter(200)
@self.register_callback('train_finish')
def visualize():
    # do some visualize stuff
    pass
```

Notice the sequence of applying decorators.

While no argument is given, functions still have access to `CEL` variables through `self.variables`.

Currently defined callbacks are:
+ `train_begin`
+ `train_finish`
+ `train_iter_begin`
+ `train_iter_finish`
+ `train_epoch_begin`
+ `train_epoch_finish`
+ `test_begin`
+ `test_finish`
+ `test_iter_begin`
+ `test_iter_finish`
+ `train_pass_begin`
+ `train_pass_finish`
+ `test_pass_begin`
+ `test_pass_finish`

Callback can also be registered before or after a certain pass
+ `train_pass_{pass.name}_begin`
+ `train_pass_{pass.name}_finish`
+ `test_pass_{pass.name}_begin`
+ `test_pass_{pass.name}_finish`

Passes can have name. If no name is given, the pass will be named as its index in the pass list.

```python
self.test_passes = [
    '[in] => netg => [o1] => netd => [o2]',  # Pass '0'
    tb.Pass('''
        [o1] -> netd => [o3]
    ''', name='second_pass_name')  # Pass 'second_pass_name'
]

@self.register_callback('test_pass_0_begin')
def before_pass_0():
    # do some stuff
    pass

@self.register_callback('test_pass_second_pass_name_finish')
def after_second_pass():
    # do some stuff
    pass
```

## Save & Load

Recipe has implemented methods `save` and `load`.

One needs register `save` to callbacks manually.

```python
# save on every 200 iters
@self.register_callback('train_iter_finish')
@self.do_every_iter(200)
# save when train finishes
@self.register_callback('train_finish')
def do_save():
    self.save()
```

`load` will raise `CheckpointError` when no valid checkpoint is found. One may wants to catch it manually.

```python
# override the load method
def load(self):
    try:
        super(MyRecipe, self).load()
    except tb.exceptions.CheckpointError:
        print('No checkpoint found, random initialize')
```

`load` will be called when Baker prepare the recipe if `recipe.need_resume` is set to `True`

By default, all registered modules will be saved to the checkpoint file. Use `torchbaker.no_save(modules)` to mark a module as not being saved automatically.

Some properties are offered to configure the save and load behaviour.

| property of Recipe   | description                                                                                                                            | default value     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| `checkpoint_dir`     | The directory to save checkpoint files. This directory will be created and initialized when the recipe is prepared.                    | `'./checkpoints'` |
| `max_checkpoint_num` | The maximum number of checkpoint files in the save directory. Old checkpoint files will be deleted if too many checkpoint files exist. | `1`               |

# Examples

A complete example of recognizing handwriting numbers in `MNIST` is provided in [mnist.py](./mnist.py).
