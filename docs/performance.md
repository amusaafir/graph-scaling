### Performance

We have compared the performance of our tool (single node) with Datagen (distributed), PaRMAT (single thread) and GScaler on the [DAS-5](https://www.cs.vu.nl/das5/). We have used 8 (CPU) nodes when running the Datagen generator; and ran PaRMAT using the default options. 
During this performance test we either scale up the Com-Orkut graph to the edge sizes specified in the table below when using a scaling tool, or generate them from scratch using a generator.

We specifically report the total execution time in seconds. In case the execution time took more than 3 hours, we report a timeout (T/O).

```markdown
|                   | This work (s) | GScaler (s) | ParMAT (s) | Datagen (s) |
|-------------------|--------------:|------------:|-----------:|------------:|
|   3 billion edges |          3020 |         DNF |       2401 |         T/O |
|   2 billion edges |          2644 |         DNF |        776 |         T/O |
|   1 billion edges |          1396 |         DNF |        589 |         T/O |
| 500 million edges |           750 |         DNF |        258 |         T/O |
| 250 million edges |           436 |         DNF |         76 |         T/O |
```

GScaler was unable to scale the input graph to a size of 250 million edges. While Datagen was able to generate graphs of much larger size (including edge weight computation), it took hours using 8 nodes to accomplish this. For all the graph sizes, PaRMAT performed the fastest.  
