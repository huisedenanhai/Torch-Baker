from functools import wraps
from .exceptions import ArgumentError, CheckpointError
import torch
import os, errno, json, datetime


class BunchDictionary(object):
    def reset_values(self):
        self.__dict__ = {k: None for k, v in self.__dict__.items()}

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class Pass(object):
    def __init__(self, pstr: str, criterion=None, optimizers: list = None, condition=None, name=None):
        self.pstr = pstr
        if criterion is None:
            criterion = lambda: None
        self.criterion = criterion
        if optimizers is None:
            optimizers = []
        self.optimizers = optimizers
        if condition is None:
            condition = lambda: True
        self.condition = condition
        self.name = name


def do_when_condition(condition):
    """
    Decorator, do something when condition is satisfied
    :param condition: a callable, return a bool when called
    :return: the real decorator
    """

    def real_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if condition():
                fn(*args, **kwargs)

        return wrapper

    return real_decorator


class CheckpointManager(object):
    save_dir = './checkpoints'
    max_num = 1

    _checkpoints = []
    _index_file_name = 'index.tbc.json'
    _checkpoints_key = 'checkpoints'
    _checkpoint_time_key = 'time'
    _checkpoint_filename_key = 'filename'

    _index_file_dirty = False

    def init_dir(self):
        # ensure path
        try:
            os.makedirs(self.save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # ensure index file
        index_path = os.path.join(self.save_dir, self._index_file_name)
        if not os.path.exists(index_path):
            self._checkpoints = []
            self._index_file_dirty = True
            self.update_index_file()
            return
        # load checkpoints info
        with open(index_path, 'r') as fp:
            info = json.load(fp)
            if self._checkpoints_key not in info:
                self._checkpoints = []
                self._index_file_dirty = True
            else:
                self._checkpoints = info[self._checkpoints_key]
        self.update_index_file()

    def update_index_file(self, check_dirty=True):
        if check_dirty and not self._index_file_dirty:
            return
        index_path = os.path.join(self.save_dir, self._index_file_name)
        with open(index_path, 'w') as fp:
            json.dump({self._checkpoints_key: self._checkpoints}, fp, indent=2, sort_keys=True)
        self._index_file_dirty = False

    def _remove_range(self, start, end):
        if start >= end:
            return
        preserved_entries_front, entries_to_delete, preserved_entries_end = \
            self._checkpoints[:start], self._checkpoints[start:end], self._checkpoints[end:]
        preserved_entries = preserved_entries_front + preserved_entries_end
        for entry in entries_to_delete:
            path = os.path.join(self.save_dir, entry[self._checkpoint_filename_key])
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
        self._checkpoints = preserved_entries
        self._index_file_dirty = True
        self.update_index_file()

    def ensure_checkpoints_num(self):
        if len(self._checkpoints) <= self.max_num:
            return
        num_to_delete = len(self._checkpoints) - self.max_num
        self._remove_range(0, num_to_delete)

    # it should be caller's job to ensure filename not duplicates
    def save(self, state_dict, file_name):
        entry = {
            self._checkpoint_filename_key: file_name,
            self._checkpoint_time_key: datetime.datetime.now().strftime('%c')
        }
        file_path = os.path.join(self.save_dir, file_name)
        torch.save(state_dict, file_path)
        self._checkpoints.append(entry)
        self._index_file_dirty = True
        self.ensure_checkpoints_num()
        self.update_index_file()

    def load(self):
        # get the most recent available checkpoint
        lastest_path = None
        lastest_index = -1
        for i, entry in enumerate(reversed(self._checkpoints)):
            path = os.path.join(self.save_dir, entry[self._checkpoint_filename_key])
            if os.path.exists(path):
                lastest_path = path
                lastest_index = i
                break
        if lastest_index == -1:
            self._remove_range(0, len(self._checkpoints))
            raise CheckpointError('No valid checkpoint to load')
        self._remove_range(len(self._checkpoints) - lastest_index, len(self._checkpoints))
        state_dict = torch.load(lastest_path, map_location=lambda storage, loc: storage)
        return state_dict


no_save_attr = 'BAKER_NO_SAVE'


def no_save(module):
    setattr(module, no_save_attr, True)


def need_save(module):
    if not hasattr(module, 'state_dict') or (hasattr(module, no_save_attr) and getattr(module, no_save_attr)):
        return False
    return True


class Recipe(object):
    iter_num = 0  # the first iter is 0
    epoch_num = 0  # the first epoch is 0
    max_iter_num = 10 ** 5
    max_epoch_num = 10 ** 5
    modules = {}
    optimizers = {}
    dataloaders = {}
    train_passes = []
    test_passes = []
    need_resume = True
    checkpoint_dir = './checkpoints'
    max_checkpoint_num = 1
    saver = CheckpointManager()

    _name = None

    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    # stores intermediate variables in each pass
    # do not overwrite this property
    variables = BunchDictionary()

    # stores parsed passes
    parsed_train_passes = []
    parsed_test_passes = []

    __iter_num_key = 'iter_num'
    __epoch_num_key = 'epoch_num'
    __modules_key = 'modules'
    __optimizers_key = 'optimizers'

    def save(self):
        state_dict = {
            self.__iter_num_key: self.iter_num,
            self.__epoch_num_key: self.epoch_num,
            self.__modules_key: {k: v.state_dict() for k, v in self.modules.items() if need_save(v)},
            self.__optimizers_key: {k: v.state_dict() for k, v in self.optimizers.items() if need_save(v)}
        }
        filename = self.name + datetime.datetime.now().strftime('-%b-%d-%I%M%S%f%p-%G') + \
                   '-iter-{}.pth'.format(self.iter_num)
        self.saver.save(state_dict, filename)

    def load(self):
        state_dict = self.saver.load()
        self.iter_num = state_dict[self.__iter_num_key]
        self.epoch_num = state_dict[self.__epoch_num_key]
        for k, v in state_dict[self.__modules_key].items():
            self.modules[k].load_state_dict(v)
        for k, v in state_dict[self.__optimizers_key].items():
            self.optimizers[k].load_state_dict(v)

    def true_every_iter(self, num=1):
        """
        Get a condition that returns True on every {num} iters
        :param num: the interval
        :return: a condition that returns True on every {num} iters
        """
        return lambda: True if self.iter_num % num == 0 else False

    def true_every_epoch(self, num=1):
        """
        Get a condition that returns True on every {num} epochs
        :param num: the interval
        :return: a condition that returns True on every {num} epochs
        """
        return lambda: True if self.epoch_num % num == 0 else False

    def do_every_iter(self, num):
        """
        Decorator. Marked method can only be executed on every {num} iters
        self.iter_num starts from 1
        :param num: the interval
        :return: the real decorator
        """
        return do_when_condition(self.true_every_iter(num))

    def do_every_epoch(self, num):
        """
        Decorator. Marked method can only be executed on every {num} epochs
        self.epoch_num starts from 1
        :param num: the interval
        :return: the real decorator
        """
        return do_when_condition(self.true_every_epoch(num))

    __callbacks = {}

    def invoke(self, event: str):
        if event in self.__callbacks:
            for fn in self.__callbacks[event]:
                fn()

    def register_callback(self, event: str, mode='append'):
        def real_decorator(fn):
            if event not in self.__callbacks:
                self.__callbacks[event] = []
            if mode == 'append':
                self.__callbacks[event].append(fn)
            elif mode == 'overwrite':
                self.__callbacks[event] = [fn]
            else:
                raise ArgumentError(
                    'undefined mode {}. modes for register_callback could be [\'append\', \'overwrite\']')
            return fn

        return real_decorator

    def parse_kwargs(self, **kwargs):
        for k, v in kwargs:
            self.__setattr__(k, v)
