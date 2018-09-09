from .recipe import Recipe, Pass
from .exceptions import NotPreparedRecipe
import re
from functools import wraps
import collections


class ParsedPass(object):
    operations = []
    criterion = lambda: None
    optimizers = []
    condition = None
    name = None

    def run(self):
        for opt in self.optimizers:
            opt.zero_grad()
        tmp = tuple()
        for op in self.operations:
            tmp = op(*tmp)
        loss = self.criterion()
        if loss is not None:
            loss.backward()
        for opt in self.optimizers:
            opt.step()


def _is_variable_pack(ele_str):
    return ele_str[0] == '[' and ele_str[-1] == ']'


def _is_arrow(ele_str):
    return ele_str == '->' or ele_str == '=>'


def _wrap_result_tuple(fn):
    @wraps(fn)
    def fn_wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result if isinstance(result, tuple) else (result,)

    return fn_wrapper


class Baker(object):

    def _register_variables(self, ele_str):
        # dirty implementation, hope I can improve it later
        ele_str = ele_str.strip()
        variable_names = [s.strip() for s in ele_str[1:-1].split(',')]
        variables = self.recipe.variables
        for name in variable_names:
            if not hasattr(variables, name):
                variables[name] = None
        return variable_names

    def _is_module(self, ele_str):
        return ele_str in self.recipe.modules

    def _assign_variables(self, values: tuple, targets: tuple):
        t_len = len(targets)
        v_len = len(values)
        if v_len > t_len == 1:  # many to one
            values = values,
        elif t_len > v_len == 1:  # one to many
            v, = values
            if isinstance(v, collections.Iterable):
                # I know this is not the prefect method for checking iterable, but it's a usable one
                values = v
        # recalculate length
        t_len = len(targets)
        v_len = len(values)
        if t_len == v_len:
            for k, v in zip(targets, values):
                self.recipe.variables[k] = v
            return values
        if t_len < v_len:
            raise ValueError('too many values to unpack (expected {}, got {}))'.format(t_len, v_len))
        raise ValueError('not enough values to unpack (expected {}, got {})'.format(t_len, v_len))

    def _interp_pass(self, p: Pass, index: int):
        pp = ParsedPass()
        pp.optimizers = [self.recipe.optimizers[k] for k in p.optimizers]
        pp.criterion = p.criterion
        pp.condition = p.condition
        pp.name = p.name if p.name is not None else str(index)

        # decode str
        pstr = p.pstr
        lines = pstr.splitlines()
        for line in lines:
            # remove comments
            cmt = re.search(r'^(.*?)#.*$', line)
            if cmt is not None:
                line = cmt.group(1)
            line = line.strip()
            if len(line) == 0:
                continue
            # find all arrows
            arrow_indices = [index for m in re.finditer(r'[-=]>', line) for index in m.span()]
            # get all elements (include arrows) and strip white spaces
            elements = [line[i:j].strip() for i, j in zip([0] + arrow_indices, arrow_indices + [None])]
            elements = [e for e in elements if len(e) > 0]  # remove empty strings
            # handle the first element
            if not _is_variable_pack(elements[0]):
                raise SyntaxError('the first element of a line should be a variable pack. line: \'{}\''.format(line))
            variables = self._register_variables(elements[0])

            def fn():
                vs = tuple(variables)

                def op(*args, **kwargs):
                    return tuple(self.recipe.variables[v] for v in vs)

                return op

            pp.operations.append(fn())

            last_arrow = None
            for i in range(1, len(elements)):
                ele_str = elements[i]
                if i % 2 == 1:
                    # should be arrows
                    if not _is_arrow(ele_str):
                        raise SyntaxError('\'{0}\' should be an arrow. line: \'{1}\''.format(ele_str, line))
                    last_arrow = ele_str
                if i % 2 == 0:
                    # should be variable pack, model or loss
                    if _is_arrow(ele_str):
                        raise SyntaxError('invalid arrow \'{0}\' at line \'{1}\''.format(ele_str, line))
                    elif _is_variable_pack(ele_str):
                        variables = self._register_variables(ele_str)
                        if last_arrow == '->':
                            def op_detach_wrap():
                                vs = tuple(variables)

                                return lambda *args, **kwargs: self._assign_variables(
                                    tuple(v.detach() if hasattr(v, 'detach') else v for v in args), vs)

                            pp.operations.append(op_detach_wrap())
                        elif last_arrow == '=>':
                            def op_wrap():
                                vs = tuple(variables)
                                return lambda *args, **kwargs: self._assign_variables(args, vs)

                            pp.operations.append(op_wrap())

                    elif self._is_module(ele_str):
                        module = self.recipe.modules[ele_str]
                        if last_arrow == '->':
                            def op_model_detach_wrap():
                                m = module

                                @_wrap_result_tuple
                                def op(*args, **kwargs):
                                    return m(*(v.detach() if hasattr(v, 'detach') else v for v in args))

                                return op

                            pp.operations.append(op_model_detach_wrap())
                        elif last_arrow == '=>':
                            def op_model_wrap():
                                m = module

                                @_wrap_result_tuple
                                def op(*args, **kwargs):
                                    return m(*args)

                                return op

                            pp.operations.append(op_model_wrap())

                    else:
                        raise SyntaxError('invalid element \'{0}\'. line: \'{1}\''.format(ele_str, line))
        return pp

    def prepare_recipe(self, recipe: Recipe, **kwargs):
        self.recipe = recipe
        recipe.parse_kwargs(**kwargs)
        recipe.parsed_train_passes = []
        for i, p in enumerate(recipe.train_passes):
            if isinstance(p, str):
                p = Pass(p)
            recipe.parsed_train_passes.append(self._interp_pass(p, i))
        recipe.parsed_test_passes = []
        for i, p in enumerate(recipe.test_passes):
            if isinstance(p, str):
                p = Pass(p)
            recipe.parsed_test_passes.append(self._interp_pass(p, i))
        # init saver
        saver = recipe.saver
        saver.save_dir = recipe.checkpoint_dir
        saver.max_num = recipe.max_checkpoint_num
        saver.init_dir()
        # resume
        if recipe.need_resume:
            recipe.load()

    __module_modes = {'train': 'train',
                      'test': 'eval'}

    def _run_pass(self, p: ParsedPass, data, phase: str):
        recipe = self.recipe
        recipe.variables['in'] = data
        recipe.invoke('{}_pass_begin'.format(phase))
        recipe.invoke('{0}_pass_{1}_begin'.format(phase, p.name))

        # switch all modules to desired mode
        mode = self.__module_modes[phase]
        for n, m in recipe.modules.items():
            if hasattr(m, mode):
                getattr(m, mode).__call__()
        p.run()
        recipe.invoke('{}_pass_finish'.format(phase))
        recipe.invoke('{0}_pass_{1}_finish'.format(phase, p.name))

    def _train_main_loop(self):
        recipe = self.recipe
        train_loader = recipe.dataloaders['train']

        while True:
            # start new epoch
            if recipe.epoch_num >= recipe.max_epoch_num:
                return
            recipe.invoke('train_epoch_begin')

            for data in train_loader:
                # start new iter
                if recipe.iter_num >= recipe.max_iter_num:
                    return
                recipe.invoke('train_iter_begin')
                # run passes
                for p in recipe.parsed_train_passes:
                    if p.condition():
                        self._run_pass(p, data, 'train')
                # iter finishes
                recipe.invoke('train_iter_finish')
                recipe.iter_num += 1

            # epoch finishes
            recipe.invoke('train_epoch_finish')
            recipe.epoch_num += 1

    def train(self):
        if not hasattr(self, 'recipe'):
            raise NotPreparedRecipe
        recipe = self.recipe

        # start train
        recipe.invoke('train_begin')
        self._train_main_loop()
        recipe.invoke('train_finish')

    def _test_main_loop(self):
        recipe = self.recipe
        test_loader = recipe.dataloaders['test']

        for data in test_loader:
            recipe.invoke('test_iter_begin')
            # run passes
            for p in recipe.parsed_test_passes:
                if p.condition():
                    self._run_pass(p, data, 'test')
            recipe.invoke('test_iter_finish')

    def test(self):
        if not hasattr(self, 'recipe'):
            raise NotPreparedRecipe
        recipe = self.recipe
        # start test
        recipe.invoke('test_begin')
        self._test_main_loop()
        recipe.invoke('test_finish')
