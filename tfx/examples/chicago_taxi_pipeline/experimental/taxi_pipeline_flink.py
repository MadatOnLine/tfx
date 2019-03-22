# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Chicago taxi example using TFX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import logging
import os
from tfx.components.evaluator.component import Evaluator
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen
from tfx.components.example_validator.component import ExampleValidator
from tfx.components.model_validator.component import ModelValidator
from tfx.components.pusher.component import Pusher
from tfx.components.schema_gen.component import SchemaGen
from tfx.components.statistics_gen.component import StatisticsGen
from tfx.components.trainer.component import Trainer
from tfx.components.transform.component import Transform
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner
from tfx.orchestration.pipeline import PipelineDecorator
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import csv_input


# Instruction before running TFX airflow example on Flink:
#   1. Creates requirements.txt that contains the additional TFX dependency:
#        tensorflow==1.12
#        tfx==0.12
#      and set environment variable:
#        $ export USER_REQUIREMENTS=<path of requirements.txt>
#   2. Starts local Flink cluster and Beam job server by running:
#        $ .../tfx/examples/chicago_taxi/setup_beam_on_flink.sh
#   3. Have a Google could storage bucket that you have read and write access.
#      Copy the taxi_utils.py and data to the bucket.

# Directory and data locations (uses Google Cloud Storage).
_input_bucket = 'gs://my-bucket'
_output_bucket = 'gs://my-bucket'

# This example assumes that the taxi data is stored in <input_bucket>/taxi/data
# and the taxi utility function is in <input_bucket>/taxi.  Feel free to
# customize this as needed.
_data_root = os.path.join(_input_bucket, 'taxi/data/simple')
# Python module file to inject customized logic into the TFX components. The
# Transform and Trainer both require user-defined functions to run successfully.
_taxi_module_file = os.path.join(_input_bucket, 'taxi/taxi_utils.py')

# Path which can be listened to by the model server.  Pusher will output the
# trained model here.
_serving_model_dir = os.path.join(_output_bucket, 'serving_model/taxi_flink')
# Path contains outputs for each components.
_pipeline_root = os.path.join(_output_bucket, 'tfx')

# Directory and data locations. This example assumes all of the chicago taxi
# example code and metadata library is relative to $HOME, but you can store
# these files anywhere on your local filesystem.
_tfx_root = os.path.join(os.environ['HOME'], 'tfx')
_metadata_db_root = os.path.join(_tfx_root, 'metadata')
_log_root = os.path.join(_tfx_root, 'logs')

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2019, 1, 1),
}

# Logging overrides
logger_overrides = {
    'log_root': _log_root,
    'log_level': logging.INFO
}


# TODO(b/124066911): Centralize tfx related config into one place.
@PipelineDecorator(
    pipeline_name='chicago_taxi_flink',
    enable_cache=True,
    metadata_db_root=_metadata_db_root,
    pipeline_root=_pipeline_root,
    additional_pipeline_args={
        'logger_args':
            logger_overrides,
        'beam_pipeline_args': [
            # ----- Beam Args -----.
            '--runner=PortableRunner',
            # Points to the job server started in setup_beam_on_flink.sh
            '--job_endpoint=localhost:8099',
            # Points to the docker created in setup_beam_on_flink.sh
            '--environment_type=DOCKER',
            '--environment_config=' + os.environ['USER'] +
            '-docker-apache.bintray.io/beam/python-with-requirements',
            # TODO(BEAM-6754): Support multi core machines.  # pylint: disable=g-bad-todo
            # Note; We use 100 worker threads to mitigate the issue with
            # scheduling work between Flink and Beam SdkHarness. Flink can
            # process unlimited work items concurrently in a TaskManager while
            # SdkHarness can only process 1 work item per worker thread. Having
            # 100 threads will let 100 tasks execute concurrently avoiding
            # scheduling issue in most cases. In case the threads are exhausted,
            # beam print the relevant message in the log.
            '--experiments=worker_threads=100',
            # ----- Flink Args -----.
            # Set to 1 as local Flink cluster started in setup_beam_on_flink.sh
            # has 1 TaskManager and 1 Task Slot.
            '--parallelism=1',
        ],
    })
def _create_pipeline():
  """Implements the chicago taxi pipeline with TFX."""
  examples = csv_input(_data_root)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input_base=examples)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

  # Generates schema based on statistics files.
  infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

  # Performs anomaly detection based on statistics and data schema.
  validate_stats = ExampleValidator(
      stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output)

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      input_data=example_gen.outputs.examples,
      schema=infer_schema.outputs.output,
      module_file=_taxi_module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=_taxi_module_file,
      transformed_examples=transform.outputs.transformed_examples,
      schema=infer_schema.outputs.output,
      transform_output=transform.outputs.transform_output,
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Uses TFMA to compute a evaluation statistics over features of a model.
  model_analyzer = Evaluator(
      examples=example_gen.outputs.examples,
      model_exports=trainer.outputs.output,
      feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(specs=[
          evaluator_pb2.SingleSlicingSpec(
              column_for_slicing=['trip_start_hour'])
      ]))

  # Performs quality validation of a candidate model (compared to a baseline).
  model_validator = ModelValidator(
      examples=example_gen.outputs.examples, model=trainer.outputs.output)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model_export=trainer.outputs.output,
      model_blessing=model_validator.outputs.blessing,
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=_serving_model_dir)))

  return [
      example_gen, statistics_gen, infer_schema, validate_stats, transform,
      trainer, model_analyzer, model_validator, pusher
  ]


pipeline = AirflowDAGRunner(_airflow_config).run(_create_pipeline())
