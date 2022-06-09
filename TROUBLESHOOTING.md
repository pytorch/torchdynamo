# Troubleshooting

## Graph Breaks

## Excessive Recompilation
When torchdynamo compiles a function (or part of one), it makes certain assumptions
about locals and globals in order to allow compiler optimizations, and expresses these
assumptions as guards that check particular values at runtime.  If any of these guards
fail, Dynamo will recompile that function (or part) up to `torchdynamo.config.cache_size_limit` times.  If your program is hitting the cache limit, you will need to determine which guards are failing and whether it can be addressed by 
... to be continued

## Other errors
Please file an issue on github, and include a stack trace / repro.