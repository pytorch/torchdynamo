#!/bin/bash
set -ex

rsync -ra ~/torchdynamo/torchdynamo/ ~/pytorch/torch/_dynamo
rsync -ra ~/torchdynamo/torchinductor/ ~/pytorch/torch/_inductor
rsync -ra ~/torchdynamo/test/{dynamo,inductor} ~/pytorch/test/
rsync -ra ~/torchdynamo/benchmarks/ ~/pytorch/benchmarks/dynamo

for DIR in ~/pytorch/test/{dynamo,inductor} ~/pytorch/benchmarks/dynamo
do
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchdynamo/torch._dynamo/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchinductor/torch._inductor/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/_torch[.]_inductor/_torchinductor/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's@pytorch/torch[.]_dynamo@pytorch/torchdynamo@g'
done

(cd ~/pytorch && (lintrunner -a || lintrunner -a))
