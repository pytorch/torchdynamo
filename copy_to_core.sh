#!/bin/bash
set -ex
cd `dirname $0`
(cd ../pytorch && make setup_lint)

rsync -ra ./torchdynamo/ ../pytorch/torch/_dynamo
rsync -ra ./torchinductor/ ../pytorch/torch/_inductor
rsync -ra ./test/{dynamo,inductor} ../pytorch/test/
rsync -ra ./benchmarks/ ../pytorch/benchmarks/dynamo

for DIR in ../pytorch/test/{dynamo,inductor} ../pytorch/benchmarks/dynamo
do
  # Rename everything
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchdynamo/torch._dynamo/g'
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/torchinductor/torch._inductor/g'
  # Fix variable names
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's/_torch[.]_inductor/_torchinductor/g'
  # Fix github urls
  find $DIR -name '*.py' | xargs -n1 -- sed -i 's@pytorch/torch[.]_dynamo@pytorch/torchdynamo@g'
done

# run lintrunner twice to workaround:
# error: Two different linters proposed changes for the same file
(cd ../pytorch && (lintrunner -a || lintrunner -a))
