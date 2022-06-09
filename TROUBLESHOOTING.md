# Troubleshooting

Torchdynamo is still in active development, and many of the reasons for graph breaks and excessive recompilation will be fixed with upcoming support for [tracing dynamic tensor shapes](https://docs.google.com/document/d/1QJB-GOnbv-9PygGlOMXwiO9K6vVNm8sNg_olixJ9koc/edit?usp=sharing), more careful choices for guards and better tuned heurstics.

In the mean time, you may need to diagnose a particular issue and determine if it is easy to work around with a change to your model, or file an issue for support.

We're also actively developing debug tools, profilers, and improving our errors/warnings.  Please give us feedback if you have an issue with this infra, or an idea for an improvement.

## File an Issue
You should feel encouraged to [file a github issue](https://github.com/pytorch/torchdynamo/issues) and expect a timely response.

Before filing an issue, read over the README.md, TROUBLESHOOTING.md, and search for similar issues.

When filing an issue, please include
- a minimal repro script
- a description of the error
- the expected behavior
- your OS/python/pytorch version

## Graph Breaks
Given a program like this,

```
with torchdynamo.optimize(...):
   some_fun(x)
   ...
```

Torchdynamo will attempt to compile all of the torch/tensor operations within some_fun into a single FX graph, but it may fail to capture everything into one graph.

Some graph break reasons are insurmountable to torchdynamo, and can't be easily fixed.
- calling into a C extension other than torch is invisible to torchdynamo, and could do arbitrary things without torchdynamo being able to introduce necessary guards to ensure that the compiled program would be safe to reuse

### Identifying the cause of a graph break
It is possible to get an aggregate list of graph break reasons by using the [recompilation profiler](#recompilation-profiler).

[No Python mode](#nopython-mode) makes torchdynamo throw an error at the first graph break, giving you the ability to jump into a stack trace.

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

## Debug/Profiling Tools

### Recompilation Profiler
The recompilation profiler collects information about the guard failures and graph breaks that occur when running your program. It ignores the current setting of `torchdynamo.config.cache_size_limit`, instead behaving as if `cache_size_limit = 1`, in order to capture the worst case behavior.

If graph breaks are observed, the profiler report will list the reasons for each graph break.  If recompilations of particular graphs are observed, The failing guards for each recompilation instance are listed separately.

```
prof = torchdynamo.utils.CompilationProfiler()
with torchdynamo.optimize(prof):
   my_model()
print(prof.report())
```

### Nopython Mode
`torchdynamo.optimize(<compiler>, nopython=True)` causes torchdynamo to throw an exception on the first graph break, rather than falling back to python transparently as it does by default.  This can be a useful tool for finding the source of a graph break and getting a stack trace.

### Debug Mode
Setting `torchdynamo.config.debug = True` offers verbose information about each compiled graph, and what it guards on.  It is generally useful for learning what torchdynamo is doing, and how it is treating your program.
