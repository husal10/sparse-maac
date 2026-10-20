Time break-even N* vs dense_flash | largest N that fits in GPU memory, per benchmark batch size

| mode              | B=8                        | B=16                      | B=32                      |
|:------------------|:---------------------------|:--------------------------|:--------------------------|
| block             | N* 895; fits all <= 8192   | N* 805; fits all <= 8192  | N* 1041; fits all <= 8192 |
| block+flex        | N* 1180; fits all <= 8192  | N* 1057; fits all <= 8192 | N* 1114; fits all <= 8192 |
| centroid          | N* 1065; fits all <= 8192  | N* 700; fits all <= 8192  | N* 559; fits all <= 8192  |
| cluster           | N* 2999; fits all <= 4096  | N* >2048; fits <= 2048    | N* 2016; fits <= 2048     |
| cluster+ebag      | N* 2782; fits all <= 4096  | N* 1795; fits all <= 4096 | N* 1697; fits <= 2048     |
| dense_flash       | N* ref; fits all <= 8192   | N* ref; fits all <= 8192  | N* ref; fits all <= 8192  |
| dense_loop        | N* >512; fits all <= 512   | N* >512; fits all <= 512  | N* >256; fits <= 256      |
| dense_loop_shared | N* >512; fits all <= 512   | N* >512; fits all <= 512  | N* >256; fits <= 256      |
| hybrid            | N* >2048; fits <= 2048     | N* >2048; fits <= 2048    | N* >1024; fits <= 1024    |
| hybrid+ebag       | N* >4096; fits all <= 4096 | N* >2048; fits <= 2048    | N* >2048; fits <= 2048    |
| knn               | N* 3625; fits all <= 4096  | N* >2048; fits <= 2048    | N* >2048; fits <= 2048    |
| knn+ebag          | N* 3370; fits all <= 4096  | N* 2749; fits all <= 4096 | N* >2048; fits <= 2048    |
| lsh               | N* >2048; fits <= 2048     | N* >2048; fits <= 2048    | N* >1024; fits <= 1024    |
| lsh+ebag          | N* >4096; fits all <= 4096 | N* >2048; fits <= 2048    | N* >1024; fits <= 1024    |
| meanfield         | N* 2336; fits all <= 4096  | N* 1441; fits all <= 4096 | N* 1419; fits <= 2048     |
| random            | N* 2961; fits all <= 4096  | N* 2010; fits <= 2048     | N* 2026; fits <= 2048     |
| random+ebag       | N* 2610; fits all <= 4096  | N* 1753; fits all <= 4096 | N* 1650; fits <= 2048     |
