# Microbatch benchmark

Effective batch256, same 14 randomly sampled length-sorted batches per setting; first2 warm-up, next12 timed. BF16, same architecture, Adam1e-4. Disposable random models; no production checkpoint writes. Includes data reads, backward, clipping and optimizer update. Independent processes. 8 and16 repeated in reverse order; results stable.

|Microbatch|Median seconds/update|Peak allocated MiB|Peak reserved MiB|512-frame stress passed|
|---|---:|---:|---:|---|
|8|0.633|1652|1760|True|
|16|0.560|2665|2948|True|
|32|0.595|4701|5660|True|
|64|0.775|8765|9954|True|

Recommendation:16, accumulate16 times. About12% shorter updates (about14% more throughput) than8 in this benchmark. Padding-frame inflation:8=1.042x,16=1.092x,32=1.200x,64=1.435x. 64 leaves little reserve and is slower; larger candidates not tested. Peak numbers are PyTorch allocations/reservations, not total device memory including display. No production training settings changed. Changing microbatch can change dropout draws/numerical rounding, so not bitwise-equivalent training.
