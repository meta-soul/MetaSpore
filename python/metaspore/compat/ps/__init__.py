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

from metaspore import NodeRole as ActorRole
from metaspore import ActorConfig
from metaspore import PSRunner

from metaspore import EmbeddingSumConcat
from metaspore import EmbeddingRangeSum
from metaspore import EmbeddingLookup

from metaspore import TensorInitializer
from metaspore import DefaultTensorInitializer
from metaspore import ZeroTensorInitializer
from metaspore import OneTensorInitializer
from metaspore import NormalTensorInitializer
from metaspore import XavierTensorInitializer

from metaspore import TensorUpdater
from metaspore import NoOpUpdater
from metaspore import SGDTensorUpdater
from metaspore import AdaGradTensorUpdater
from metaspore import AdamTensorUpdater
from metaspore import FTRLTensorUpdater
from metaspore import EMATensorUpdater

from metaspore import Agent
from metaspore import Model
from metaspore import SparseModel
from metaspore import ModelMetric as ModelCriterion
from metaspore import DistributedTrainer

try:
    import pyspark
except ImportError:
    pass
else:
    from metaspore import PyTorchAgent
    from metaspore import PyTorchLauncher
    from metaspore import PyTorchModel
    from metaspore import PyTorchEstimator

from metaspore import __version__
from metaspore import _metaspore as _ps

from metaspore import nn
from metaspore import input
from metaspore import spark

from metaspore import initializer
from metaspore import updater
