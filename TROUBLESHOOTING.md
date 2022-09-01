# Troubleshooting

Torchdynamo is still in active development, and many of the reasons for graph breaks and excessive recompilation will be fixed with upcoming support for [tracing dynamic tensor shapes](https://docs.google.com/document/d/1QJB-GOnbv-9PygGlOMXwiO9K6vVNm8sNg_olixJ9koc/edit?usp=sharing), more careful choices for guards and better tuned heurstics.

In the mean time, you may need to diagnose a particular issue and determine if it is easy to work around with a change to your model, or file an issue for support.

We're also actively developing debug tools, profilers, and improving our errors/warnings.  Please give us feedback if you have an issue with this infra, or an idea for an improvement.


## Diagnosing Runtime Errors
... Insert steps to narrow error to before/after fx graph generation/and then wich tool to narrow the scope of the error




## Graph Breaks
Given a program like this,

```
@torchdynamo.optimize(...):
def some_fun(x):
   ...

some_fun(x)

```

Torchdynamo will attempt to compile all of the torch/tensor operations within some_fun into a single FX graph, but it may fail to capture everything into one graph.

Some graph break reasons are insurmountable to torchdynamo, and can't be easily fixed.
- calling into a C extension other than torch is invisible to torchdynamo, and could do arbitrary things without torchdynamo being able to introduce necessary guards to ensure that the compiled program would be safe to reuse

### Identifying the cause of a graph break

To identify all graph breaks in a program and the associated reasons for the breaks, `torchdynamo.explain` can be used. This tool runs Torchdynamo on the supplied function and aggregates the graph breaks that are encountered. Here is an example usage:

```python
from typing import List
import torch
import torchdynamo

def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")
    if b.sum() < 0:
        b = b * -1
    return x * b

explained = torchdynamo.explain(toy_example, torch.randn(10), torch.randn(10))
print(explained[0])

"""
Dynamo produced 3 graphs, with 2 graph break and 6 ops. 
 Break reasons: 

1. call_function BuiltinVariable(print) [ConstantVariable(str)] {} 
   File "t2.py", line 16, in toy_example
    print("woo")
 
2. generic_jump 
   File "t2.py", line 17, in toy_example
    if b.sum() < 0:
 """
```

To throw an error on the first graph break encountered, `nopython` mode can be used. This disables Torchdynamo's python fallback, and only succeeds if the entire program is convertible to a single graph. Example usage:

```python
@torchdynamo.optimize(<compiler>, nopython=True)
def toy_example(a, b):
   ...
```

## Excessive Recompilation
When torchdynamo compiles a function (or part of one), it makes certain assumptions
about locals and globals in order to allow compiler optimizations, and expresses these
assumptions as guards that check particular values at runtime.  If any of these guards
fail, Dynamo will recompile that function (or part) up to `torchdynamo.config.cache_size_limit` times.  If your program is hitting the cache limit, you will first need to determine which guard is failing and what part of your program is triggering it.

The [recompilation profiler](#recompilation-profiler) automates the process of setting torchdynamo's cache limit to 1 and running your program under an observation-only 'compiler' that records the causes of any guard failures.  You should be sure to run your program for at least as long (as many iterations) as you were running when you ran into trouble, and the profiler will accumulate statistics over this duration.

If your program exhibits a bounded amount of dynamism, you may be able to tune the torchdynamo cache limit to allow for each variation to be compiled and cached, but if the cache limit is too high you may find the cost of recompilation outweighs any optimization benefits.

```
torchdynamo.config.cache_size_limit = <your desired cache limit>
```

Torchdynamo plans to support many common cases of dynamic tensor shapes, such as varying batch size or sequence length.  It does not plan to support rank-dynamism.  In the mean time, setting a specific cache limit can be used in coordination with bucketing techniques to achieve an acceptable number of recompilations for some dynamic models.

```
prof = torchdynamo.utils.CompilationProfiler()
opt_my_model = torchdynamo.optimize(prof)(my_model)
opt_my_model()
print(prof.report())
```


## File an Issue
You should feel encouraged to [file a github issue](https://github.com/pytorch/torchdynamo/issues) and expect a timely response.

Before filing an issue, read over the README.md, TROUBLESHOOTING.md, and search for similar issues.

When filing an issue, please include
- a minimal repro script
- a description of the error
- the expected behavior
- your OS/python/pytorch version
