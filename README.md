# CluWords: Exploiting Semantic Word Clustering Representation for Enhanced Topic Modeling

This is the code for the paper:

Viegas, Felipe and Canuto, Sérgio and Gomes, Christian and Luiz, Washington and Rosa, Thierson and Ribas, Sabir and Rocha, Leonardo and Gonçalves, Marcos André. CluWords: Exploiting SemanticWord Clustering Representation for Enhanced Topic Modeling.The Twelfth ACM International Conference on Web Search and Data Mining (WSDM ’19)

Build docker container:

```docker build -t cluwords `pwd` ```

Run docker container:

```docker run --rm --name cluwords -v `pwd`:/cluwords -i -t cluwords /bin/bash```

To run the code:

```python3 main.py -h```

### Cite
If you find this code useful in your research, please, consider citing our paper:

```@inproceedings{viegas2019cluwords,
title={CluWords: Exploiting SemanticWord Clustering Representation for Enhanced Topic Modeling},
author={Viegas, Felipe and Canuto, Sérgio and Gomes, Christian and Luiz, Washington and Rosa, 
Thierson and Ribas, Sabir and Rocha, Leonardo and Gonçalves, Marcos André},
booktitle={The Twelfth ACM International Conference on Web Search and Data Mining (WSDM ’19)},
year={2019},
organization={ACM}
}```
