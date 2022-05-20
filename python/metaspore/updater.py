#
# Copyright 2022 DMetaSoul
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import abc
import operator
import torch
import numpy

class TensorUpdater(abc.ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if not isinstance(value, float) or value <= 0.0:
            message = "learning_rate must be positive float; "
            message += "%r is invalid" % value
            raise ValueError(message)
        self._learning_rate = value

    def get_dense_data_shape(self, tensor):
        data_shape = tensor.item.shape
        if len(data_shape) == 1:
            return data_shape[0], 1
        else:
            return data_shape

    def get_dense_state_shape(self, tensor):
        normalized = self.get_dense_data_shape(tensor)
        return self.get_state_shape(tensor, normalized)

    def get_sparse_slice_data_shape(self, tensor):
        return tensor.item._checked_get_embedding_size(),

    def get_sparse_slice_state_shape(self, tensor):
        normalized = self.get_sparse_slice_data_shape(tensor)
        return self.get_state_shape(tensor, normalized)

    @property
    def states_per_param(self):
        return None

    def get_state_shape(self, tensor, data_shape):
        num = self.states_per_param
        if num is not None and num > 0:
            state_shape = list(data_shape)
            state_shape[-1] *= num
            return tuple(state_shape)
        return None

    def get_state_tensor(self, state, index):
        num = self.states_per_param
        dim = state.shape[-1]
        subscript = list(slice(None) for d in state.shape)
        subscript[-1] = slice(dim // num * index, dim // num * (index + 1))
        subscript = tuple(subscript)
        tensor = operator.getitem(state, subscript)
        return tensor

    def get_dense_state_tensor(self, state, index):
        return self.get_state_tensor(state, index)

    def get_sparse_state_tensor(self, state, index):
        return self.get_state_tensor(state, index)

    @abc.abstractmethod
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.learning_rate)

    @abc.abstractmethod
    def update_dense(self, name, param, grad, state):
        raise NotImplementedError

    @abc.abstractmethod
    def update_sparse(self, name, param, grad, state, indices, keys):
        raise NotImplementedError

    def __call__(self, name, param, grad, state, indices, keys):
        if grad.size == 0:
            return
        param = torch.from_numpy(param)
        grad = torch.from_numpy(grad)
        if state is not None:
            state = torch.from_numpy(state)
        if indices is None:
            self.update_dense(name=name, param=param, grad=grad, state=state)
        else:
            indices = torch.from_numpy(indices.view(numpy.int64))
            keys = torch.from_numpy(keys.view(numpy.int64))
            self.update_sparse(name=name, param=param, grad=grad, state=state, indices=indices, keys=keys)

# This updater does not update parameters, which can be used as the updater of
# read-only sparse or dense tensors to avoid allocating optimizer state.
class NoOpUpdater(TensorUpdater):
    def __repr__(self):
        return super().__repr__()

    def update_dense(self, name, param, grad, state):
        pass

    def update_sparse(self, name, param, grad, state, indices, keys):
        pass

class SGDTensorUpdater(TensorUpdater):
    def __repr__(self):
        return super().__repr__()

    def update_dense(self, name, param, grad, state):
        param -= self.learning_rate * grad

    def update_sparse(self, name, param, grad, state, indices, keys):
        param[indices] -= self.learning_rate * grad

class AdaGradTensorUpdater(TensorUpdater):
    def __init__(self, learning_rate, float_stable_eps=0.0, l2=0.0):
        super().__init__(learning_rate)
        if not isinstance(float_stable_eps, float) or float_stable_eps < 0.0:
            message = "float_stable_eps must be non-negative float; "
            message += "%r is invalid" % float_stable_eps
            raise ValueError(message)
        if not isinstance(l2, float) or l2 < 0.0:
            message = "l2 must be non-negative float; "
            message += "%r is invalid" % l2
            raise ValueError(message)
        self._float_stable_eps = float_stable_eps
        self._l2 = l2

    @property
    def float_stable_eps(self):
        return self._float_stable_eps

    @property
    def l2(self):
        return self._l2

    def __repr__(self):
        return '%s(%r, %r, %r)' % (self.__class__.__name__,
                                   self.learning_rate,
                                   self.float_stable_eps,
                                   self.l2)

    @property
    def states_per_param(self):
        return 1

    def update_dense(self, name, param, grad, state):
        square_sum = self.get_dense_state_tensor(state, 0)
        grad_tmp = grad + self.l2 * param
        square_sum += grad_tmp * grad_tmp
        param -= self.learning_rate * grad_tmp / (square_sum + self.float_stable_eps).sqrt()

    def update_sparse(self, name, param, grad, state, indices, keys):
        square_sum = self.get_sparse_state_tensor(state, 0)
        grad_tmp = grad + self.l2 * param[indices]
        square_sum[indices] += grad_tmp * grad_tmp
        param[indices] -= self.learning_rate * grad_tmp / (square_sum[indices] + self.float_stable_eps).sqrt()

class AdamTensorUpdater(TensorUpdater):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        if not isinstance(beta1, float) or beta1 < 0.0 or beta1 >= 1.0:
            message = "beta1 must be non-negative float less than 1.0; "
            message += "%r is invalid" % beta1
            raise ValueError(message)
        if not isinstance(beta2, float) or beta2 < 0.0 or beta2 >= 1.0:
            message = "beta2 must be non-negative float less than 1.0; "
            message += "%r is invalid" % beta2
            raise ValueError(message)
        if not isinstance(epsilon, float) or epsilon < 0.0:
            message = "epsilon must be non-negative float; "
            message += "%r is invalid" % epsilon
            raise ValueError(message)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (self.__class__.__name__,
                                       self.learning_rate,
                                       self._beta1,
                                       self._beta2,
                                       self._epsilon)

    @property
    def states_per_param(self):
        return 2

    def update_dense(self, name, param, grad, state):
        m = self.get_dense_state_tensor(state, 0)
        v = self.get_dense_state_tensor(state, 1)
        m[...] = self._beta1 * m + (1.0 - self._beta1) * grad
        v[...] = self._beta2 * v + (1.0 - self._beta2) * grad * grad
        param -= self.learning_rate * m / (v.sqrt() + self._epsilon)

    def update_sparse(self, name, param, grad, state, indices, keys):
        m = self.get_sparse_state_tensor(state, 0)
        v = self.get_sparse_state_tensor(state, 1)
        m[indices] = self._beta1 * m[indices] + (1.0 - self._beta1) * grad
        v[indices] = self._beta2 * v[indices] + (1.0 - self._beta2) * grad * grad
        param[indices] -= self.learning_rate * m[indices] / (v[indices].sqrt() + self._epsilon)

class AdamWTensorUpdater(TensorUpdater):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-2, amsgrad=False):
        super().__init__(learning_rate)
        if not isinstance(beta1, float) or beta1 < 0.0 or beta1 >= 1.0:
            message = "beta1 must be non-negative float less than 1.0; "
            message += "%r is invalid" % beta1
            raise ValueError(message)
        if not isinstance(beta2, float) or beta2 < 0.0 or beta2 >= 1.0:
            message = "beta2 must be non-negative float less than 1.0; "
            message += "%r is invalid" % beta2
            raise ValueError(message)
        if not isinstance(epsilon, float) or epsilon < 0.0:
            message = "epsilon must be non-negative float; "
            message += "%r is invalid" % epsilon
            raise ValueError(message)
        if not isinstance(weight_decay, float) or weight_decay < 0.0:
            message = "weight_decay must be non-negative float; "
            message += "%r is invalid" % weight_decay
            raise ValueError(message)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._amsgrad = amsgrad
        self._step = 0

    def __repr__(self):
        return '%s(%r, %r, %r, %r, %r, %r)' % (self.__class__.__name__,
                                               self.learning_rate,
                                               self._beta1,
                                               self._beta2,
                                               self._epsilon,
                                               self._weight_decay,
                                               self._amsgrad)

    @property
    def states_per_param(self):
        return 3 if self._amsgrad else 2

    def update_dense(self, name, param, grad, state):
        import torch.optim._functional as F
        m = self.get_dense_state_tensor(state, 0)
        v = self.get_dense_state_tensor(state, 1)
        if self._amsgrad:
            max_v = self.get_dense_state_tensor(state, 2)
        self._step += 1
        F.adamw((param,),
                (grad,),
                (m,),
                (v,),
                (max_v,) if self._amsgrad else (),
                (self._step,),
                amsgrad=self._amsgrad,
                beta1=self._beta1,
                beta2=self._beta2,
                lr=self.learning_rate,
                weight_decay=self._weight_decay,
                eps=self._epsilon,
                maximize=False)

    def update_sparse(self, name, param, grad, state, indices, keys):
        message = "AdamWTensorUpdater does not support sparse tensor; "
        message += "can not update tensor %r" % name
        raise RuntimeError(message)

class FTRLTensorUpdater(TensorUpdater):
    def __init__(self, l1=1.0, l2=120.0, alpha=0.5, beta=1.0):
        if not isinstance(l1, float) or l1 < 0.0:
            message = "l1 must be non-negative float; "
            message += "%r is invalid" % l1
            raise ValueError(message)
        if not isinstance(l2, float) or l2 < 0.0:
            message = "l2 must be non-negative float; "
            message += "%r is invalid" % l2
            raise ValueError(message)
        if not isinstance(alpha, float) or alpha < 0.0:
            message = "alpha must be non-negative float; "
            message += "%r is invalid" % alpha
            raise ValueError(message)
        if not isinstance(beta, float) or beta < 0.0:
            message = "beta must be non-negative float; "
            message += "%r is invalid" % beta
            raise ValueError(message)
        super().__init__(l1)
        self._l1 = l1
        self._l2 = l2
        self._alpha = alpha
        self._beta = beta

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (self.__class__.__name__,
                                       self._l1,
                                       self._l2,
                                       self._alpha,
                                       self._beta)

    @property
    def states_per_param(self):
        return 2

    def update_dense(self, name, param, grad, state):
        n = self.get_dense_state_tensor(state, 0)
        z = self.get_dense_state_tensor(state, 1)
        grad_square = grad * grad
        sigma = ((n + grad_square).sqrt() - n.sqrt()) / self._alpha
        z[...] = z + grad - sigma * param
        n[...] = n + grad_square
        param[...] = torch.where(torch.abs(z) <= self._l1,
            torch.tensor(0.0),
            -(z - torch.sign(z) * self._l1) / ((self._beta + n.sqrt()) / self._alpha + self._l2))

    def _sign(self, x):
        return torch.where(x > 0.0, torch.tensor(1.0), torch.tensor(-1.0))

    def update_sparse(self, name, param, grad, state, indices, keys):
        n = self.get_sparse_state_tensor(state, 0)
        z = self.get_sparse_state_tensor(state, 1)
        grad_square = grad * grad
        sigma = ((n[indices] + grad_square).sqrt() - n[indices].sqrt()) / self._alpha
        z[indices] = z[indices] + grad - sigma * param[indices]
        n[indices] = n[indices] + grad_square
        x = torch.tensor(0.0)
        y = -(z[indices] - self._sign(z[indices]) * self._l1) / ((self._beta + n[indices].sqrt()) / self._alpha + self._l2)
        condition = torch.abs(z[indices]) <= self._l1
        param[indices] = torch.where(condition, x, y)

# Exponential Moving Average tensor updater
# Useful for running_mean and running_var of BatchNorm operators.
# This is a very special updater.
class EMATensorUpdater(TensorUpdater):
    def __init__(self, momentum=0.1):
        if not isinstance(momentum, float) or momentum <= 0.0 or momentum >= 1.0:
            message = "momentum must be positive float less than 1.0; "
            message += "%r is invalid" % momentum
            raise ValueError(message)
        super().__init__(momentum)

    @property
    def momentum(self):
        return self.learning_rate

    def __repr__(self):
        return super().__repr__()

    def update_dense(self, name, param, grad, state):
        param[...] = (1 - self.momentum) * param + self.momentum * grad

    def update_sparse(self, name, param, grad, state, indices, keys):
        param[indices] = (1 - self.momentum) * param[indices] + self.momentum * grad
