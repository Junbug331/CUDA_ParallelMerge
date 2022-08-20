# CUDA Merging Two sorted Array

---

## Build and Run
```shell
mkdir build
cd build
cmake -GNinja ..
ninja
./CUDA_ParallelMerge
```


## Key Points

- Use `co_rank` function to predetermine the lengths of two input sub arrays that are needed for result sub array.  Sum of the lengths of two input sub array = length of the result sub array
    - ex) A: 1 2 4 5 6,   B: 2 3 5 6 9,    C[n] ?
        - C[0] → 0 from A{}, 0 from B{}
        - C[1] → 1 from A{1}, 0 from B{}
        - C[2] → 2 from A{1, 2}, 0 from B{}  or   1 from A{1}, 1 from B{2}
        - C[3] → 2 from A{1, 2}, 1 from B{1}
        - C[4] → 2 from A{1, 2}, 2 from B{2, 3}
        - C[5] → 3 from A{1, 2, 4}, 2 from B{2, 3}
- Use shared memory to reduce memory overhead.
- Use shared memory as circular buffer to increase memory bandwidth.

## co_rank function

- co-rank : co rank `C[n]` is (`i` ,`j`) from input array `A`, `B` such that in order to make `n` length `C` array, `i` elements from `A` and `j` elements from `B` are required. Note that **`i` + `j` = `n`**. Hence, If one can calculate `i`, then `j` is easily derived by **`n` - `i`**
- By exploiting the fact that two arrays are **sorted**, complexity is O(log N)


    - Note that this only calculate `i`

## Using Shared Memory

- Task is divided **not** by number of threads but rather by grid dimension(block) and the size of tile(shared memory).
- Result array is divided by number of block, then in each block divided sub array is further divided again by the tile.
    - ex) Length of result vector m+n = 36000, `blocksPerGrid` 8, `threadsPerBlock` 128, `tile_size` 1024,
        - ceil(36000/8) = 4500 → Length of result vector divided by number of blocks, `C_len`
        - ceil(4500/1024) = 5 → **Number of iteration** needed to calculate C_len
            - 1024 + 1024 + 1024 + 1024 + 404
        - At each iteration, sub-array of C_len(1024,.. 404) is then calculated.
            - Number of threads in a block fills tile(shared memory)
            - Once filled, each threads calculate to calculate sub-array of C_len
                - ceil(1024/128) = 8
                - C[8], C[16] .., C[1024]


## Circular Buffer

- Note that, in above implementation, 1024 size of memory are assigned shared memory for A(`shareA`) and B(`shareB`) to calculate C sub array of length 1024. This implies that total size of 2048 are assigned but only half of them are used. Then entire 1024 length of share memory for A and B are loaded again, overriding the ones that not used.
    - ex) tile_size = 8,
        - iteratoration_0
            - A: 1 3 4 5 8 9 10 14
            - B: 1 2 5 6 7 12 15 15
            - C[8] : A{1 3 4 5} B{1 2 5 6}
        - iteration_1
            - A: 8 9 10 14 … new elements …
            - B: 7 12 1515 … new elements …
        - Note that same elements that are loaded before are loaded again
- This is can be addressed by using circular buffer
    - ex) tile_size = 8
        - iteration_0
            - A: 1 3 4 5 5 9 10 14, start index : 0, consumed elements: 5 {1 3 4 5 5}
            - B: 1 2 5 6 7 12 15 15, start index : 0, consumed elements: 3 {1 2 5}
            - C[8] : A{1 3 4 5 5} B{1 2 5}
        - iteration_1
            - A: **15 19 20 21** **23** 9 10 14, start index : 5
                - 5 new elements are loaded
            - B: **15 16 18** 6 7 12 15 15, start index: 3
                - 3 new elements are loaded
- Note that only *`tile_size`* - `consumed_elements` elements are loaded.