#!/bin/bash

# clear cache
rm -rf /tmp/torchinductor_*/

# start several process at the same time
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
python test_multi.py &
wait
