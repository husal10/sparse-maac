Time break-even N* vs dense_flash | largest N that fits in GPU memory, per benchmark batch size

| mode              | B=8                        | B=16                     |
|:------------------|:---------------------------|:-------------------------|
| block             | N* 894; fits all <= 8192   | N* 865; fits all <= 8192 |
| centroid          | N* 739; fits all <= 8192   | N* 576; fits all <= 8192 |
| cluster           | N* 3631; fits all <= 4096  | N* >2048; fits <= 2048   |
| dense_flash       | N* ref; fits all <= 8192   | N* ref; fits all <= 8192 |
| dense_loop        | N* >512; fits all <= 512   | N* >512; fits all <= 512 |
| dense_loop_shared | N* >512; fits all <= 512   | N* >512; fits all <= 512 |
| hybrid            | N* >2048; fits <= 2048     | N* >2048; fits <= 2048   |
| knn               | N* >4096; fits all <= 4096 | N* >2048; fits <= 2048   |
| lsh               | N* >2048; fits <= 2048     | N* >2048; fits <= 2048   |
| random            | N* 3317; fits all <= 4096  | N* >2048; fits <= 2048   |
