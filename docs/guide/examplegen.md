# The ExampleGen TFX Pipeline Component

The ExampleGen TFX Pipeline component ingests
data into TFX pipelines. It consumes external files/services to generate
Examples which will be read by other TFX components.

*   Consumes: Data from external data sources such as CSV, TFRecord and BigQuery
*   Emits: tf.Example records

## ExampleGen and Other Components

ExampleGen provides data to components that make use of the
[TensorFlow Data Validation](tfdv.md) library, such as [SchemaGen](schemagen.md),
[StatisticsGen](statsgen.md), and [Example Validator](exampleval.md).  It also
provides data to [Transform](transform.md), which makes use of the
[TensorFlow Transform](tft.md) library, and ultimately to deployment targets
during inference.

## Developing an ExampleGen Component

For supported data sources (currently, CSV files, TFRecord files and results of
BigQuery queries) the ExampleGen pipeline component is typically very easy to
deploy and requires little customization. Typical code looks like this:

```python
from tfx.utils.dsl_utils import csv_input
from tfx.components.example_gen.csv_example_gen.component import CsvExampleGen

examples = csv_input(os.path.join(base_dir, 'data/simple'))
example_gen = CsvExampleGen(input_base=examples)
```

## Custom input/output split

To custom train/eval output split ratio, set the output_config for ExampleGen
component, for example:

```python
from  tfx.proto import example_gen_pb2

# train:eval=3:1
output = example_gen_pb2.Output(
             split_config=example_gen_pb2.SplitConfig(splits=[
                 example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=3),
                 example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
             ]))
examples = csv_input(base_dir)
example_gen = CsvExampleGen(input_base=examples, output_config=output)
```

For pre-split input source, set the input_config for ExampleGen component:

```python
from  tfx.proto import example_gen_pb2

# train split is 'base_dir/train/*', eval split is 'base_dir/eval/*'.
input = example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='train', pattern='train/*'),
                example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
            ])
examples = csv_input(base_dir)
example_gen = CsvExampleGen(input_base=examples, input_config=input)
```

Please refer to proto/example_gen.proto for details.
