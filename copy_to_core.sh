#!/bin/bash
set -ex

rsync -ra ~/torchdynamo/torchdynamo/ ~/pytorch/torch/dynamo
rsync -ra ~/torchdynamo/torchinductor/ ~/pytorch/torch/inductor
rsync -ra ~/torchdynamo/test/{dynamo,inductor} ~/pytorch/test/
rsync -ra ~/torchdynamo/benchmarks/ ~/pytorch/benchmarks/dynamo

for DIR in ~/pytorch/test/{dynamo,inductor} ~/pytorch/benchmarks/dynamo
do
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchdynamo/torch.dynamo/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchinductor/torch.inductor/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/_torch[.]inductor/_torchinductor/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's@pytorch/torch[.]dynamo@pytorch/torchdynamo@g'
done

lintrunner -a
