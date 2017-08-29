import os

from chainer import Chain, Variable, cuda, serializers
from chainer.dataset import concat_examples
from chainer.iterators import SerialIterator


class InfluenceFunctionsCalculator(object):
    """Chain class which supports calculating various influence functions"""
    _infl_states = None

    # State for each param in links
    STATE_PARAM_ORIGINAL = 'param_original'
    STATE_HINV_V = 'HinvV'
    STATE_V = 'V'
    STATE_GRAD_ORIGINAL = 'grad_original'
    STATE_GRAD_PERTURBED = 'grad_perturbed'

    # State for input variable
    STATE_INPUT_GRAD_PERTURBED = 'input_grad_perturbed'
    STATE_INPUT_GRAD_ORIGINAL = 'input_grad_original'

    def __init__(self, target):
        """
        
        Args:
            target (Chain): model instance whose influence functions to be 
                            calculated
        """
        super(InfluenceFunctionsCalculator, self).__init__()
        self.target = target  # type: Chain

        # Init states
        self._init_infl_states()
        # Save original param...
        states = self._infl_states
        for name, param in self.target.namedparams():
            with cuda.get_device_from_array(param.data):
                state = states[name]
                state[self.STATE_PARAM_ORIGINAL] = param.data.copy()

    def _calc_and_register_grad(self, batch, key, lossfun, converter):
        self.target.cleargrads()
        loss = lossfun(*converter(batch))  # type: Variable
        loss.backward()
        # If we need grad of Variables, not only parameters, use below
        # loss.backward(retain_grad=True)

        states = self._infl_states
        for name, param in self.target.namedparams():
            with cuda.get_device_from_array(param.data):
                state = states[name]
                state[key] = param.grad.copy()

    def _init_infl_states(self):
        """Initilize `infl_states` dict

        This method initializes this calculator's states
        
        Currently, following keys are used for each link
         - param_origin
         - HinvV
         - V
         - grad_original
         - grad_perturbed

        """
        self._infl_states = {}
        for name, param in self.target.namedparams():
            #if name not in self._infl_states:
            self._infl_states[name] = {}

    def _clear_infl_states_for_calc_s_test(self):
        keys = [
            self.STATE_V,
            self.STATE_GRAD_ORIGINAL,
            self.STATE_GRAD_PERTURBED
        ]
        for name, param in self.target.namedparams():
            for key in keys:
                del self._infl_states[name][key]

    def calc_s_test(self, z_train, z_test, r=10, t=5000,
                    epsilon=1e-5,
                    lossfun=None,
                    converter=concat_examples):
        """
        
        Args:
            z_train: train dataset, basically it can be whole train dataset.
            z_test: test dataset, it should be one z_minibatch size.
            t (int): batch size used for one iteration when calculating grad of 
                     train dataset
            r (int): repeat time to update HinvV
            lossfun: loss function

        Returns:

        """
        if lossfun is None:
            # use self.target.__call__ as loss function if not set.
            lossfun = self.target

        states = self._infl_states

        self._calc_and_register_grad(z_test, 'V', lossfun, converter)

        # init HinvV
        for name, param in self.target.namedparams():
            with cuda.get_device_from_array(param.data):
                state = states[name]
                state['HinvV'] = state['V'].copy()

        # Train
        train_iter = SerialIterator(z_train, t)

        # Loop to calculate accurate HinvV
        for _ in range(r):
            train_batch = train_iter.next()

            # 1. Calc grad of original param
            self._calc_and_register_grad(train_batch, 'grad_original', lossfun,
                                         converter)

            # 2. Pertuabation of params and calc grad of perturbed param
            for name, param in self.target.namedparams():
                param = param + epsilon * states[name][self.STATE_HINV_V]
            self._calc_and_register_grad(train_batch, self.STATE_GRAD_PERTURBED, lossfun,
                                         converter)

            # 3. Revert params
            for name, param in self.target.namedparams():
                param = states[name][self.STATE_PARAM_ORIGINAL]
            #serializers.load_npz(self.target_filepath, self.target)

            # 4. Update HinvV
            # HinvV <- V + HinvV - (H dot HinvV)
            for name, param in self.target.namedparams():
                with cuda.get_device_from_array(param.data):
                    state = states[name]
                    state[self.STATE_HINV_V] = state[self.STATE_V] + state[self.STATE_HINV_V] - (state[self.STATE_GRAD_PERTURBED] - state[self.STATE_GRAD_ORIGINAL]) / epsilon

        # Here, all the repetition process end!
        # state['HinvV'] can be used as s_test.
        self._clear_infl_states_for_calc_s_test()

    def I_up_params(self, z, z_train, lossfun=None, converter=concat_examples):
        self.calc_s_test(z_train, z, lossfun=lossfun, converter=converter)
        # Here, -state['HinvV'] is I_up_params.

    def I_up_loss(self, z, lossfun=None, converter=concat_examples):
        """Calculate I_up_loss(z, z_test)
        
        Note that `calc_s_test` must be executed beforehand, z_test is used 
        in this method.
        
        Args:
            z: 
            lossfun: 
            converter: 

        Returns:

        """
        # TODO: currently, z must be 1 minibatch size.
        # Change this to use `elementwise_grad` so that we can calculate
        # I_up_loss for each z much more efficiently.

        if lossfun is None:
            # use self.target.__call__ as loss function if not set.
            lossfun = self.target

        # [Note] Use `grad_original` state
        self._calc_and_register_grad(z, self.STATE_GRAD_ORIGINAL, lossfun, converter)
        states = self._infl_states

        final_loss = 0
        for name, param in self.target.namedparams():
            with cuda.get_device_from_array(param.data):
                state = states[name]
                xp = cuda.get_array_module(param.data)
                final_loss += xp.sum(-state[self.STATE_HINV_V] * state[self.STATE_GRAD_ORIGINAL])
        return final_loss

    def _calc_and_register_input_grad(self, z, key, lossfun, converter):
        self.target.cleargrads()
        x_batch, y_batch = converter(z)
        x_var = Variable(x_batch)
        y_var = Variable(y_batch)
        loss = lossfun(x_var, y_var)  # type: Variable
        loss.backward(retain_grad=True)
        self._infl_states[key] = x_var.grad

    def I_pert_loss(self, z, epsilon=1e-5, lossfun=None, converter=concat_examples):
        if lossfun is None:
            # use self.target.__call__ as loss function if not set.
            lossfun = self.target

        states = self._infl_states
        # 1. Pertuabation of params and calc grad of perturbed param
        for name, param in self.target.namedparams():
            param = param + epsilon * states[name][self.STATE_HINV_V]
        self._calc_and_register_input_grad(z, self.STATE_INPUT_GRAD_PERTURBED,
                                           lossfun,
                                           converter)

        # 2. Revert params
        for name, param in self.target.namedparams():
            param = states[name][self.STATE_PARAM_ORIGINAL]
        self._calc_and_register_input_grad(z, self.STATE_INPUT_GRAD_ORIGINAL,
                                           lossfun,
                                           converter)
        final_loss = - (self._infl_states[self.STATE_INPUT_GRAD_PERTURBED] - self._infl_states[self.STATE_INPUT_GRAD_ORIGINAL]) / epsilon
        return final_loss
