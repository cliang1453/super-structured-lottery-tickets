# Structured Super Lottery Tickets in BERT

This repo contains our codes for the paper ["Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization"](https://arxiv.org/abs/2105.12002) (ACL 2021).

</br>

## Getting Start
1. python3.6 </br>
   Reference to download and install : https://www.python.org/downloads/release/python-360/
2. install requirements </br>
   ```> pip install -r requirements.txt```

</br>

## Data
1. Download data </br>
   ``` sh download.sh``` </br>
   Please refer to download GLUE dataset: https://gluebenchmark.com/
2. Preprocess data </br>
   ```> sh experiments/glue/prepro.sh```</br>
For more data processing details, please refer to this [repo](https://github.com/namisan/mt-dnn).

</br>

## Verifying Phase Transition Phenomenon
1. Fine-tune a pre-trained BERT model with single task data, compute importance scores, and generate one-shot structured pruning masks at multiple sparsity levels. E.g., for MNLI, run </br>
   ```
   ./scripts/train_mnli.sh GPUID
   ```

2. Rewind and evaluate the winning, random, and losing tickets at multiple sparsity levels. E.g., for MNLI, run </br>
   ```
   ./scripts/rewind_mnli.sh GPUID
   ```
You may try tasks with smaller sizes (e.g., SST, MRPC, RTE) to see a more pronounced phase transition.


</br>

## Multi-task Learning (MTL) with Tickets Sharing
1. Identify a set of super tickets for each individual task.

   - Identify winning tickets at multiple sparsity levels for each individual task. E.g., for MTDNN-base, run
      ```
      ./scripts/prepare_mtdnn_base.sh GPUID
      ```
      We recommend to use the same optimization settings, e.g., learning rate, optimizer and random seed, in both the ticket identification procedures and the MTL. We empirically observe that the super tickets perform better in MTL in such a case.

   - **[Optional]** For each individual task, identify a set of super tickets from the winning tickets at multiple sparsity levels. You can skip this step if you wish to directly use the set of super tickets identified by us. If you wish to identify super tickets on your own (This is recommended if you use a different optimization settings, e.g., learning rate, optimizer and random seed, from those in our scripts. These factors may affect the candidacy of super tickets.), we provide the template scripts
      ```
      ./scripts/rewind_mnli_winning.sh GPUID
      ./scripts/rewind_qnli_winning.sh GPUID
      ./scripts/rewind_qqp_winning.sh GPUID
      ./scripts/rewind_sst_winning.sh GPUID
      ./scripts/rewind_mrpc_winning.sh GPUID
      ./scripts/rewind_cola_winning.sh GPUID
      ./scripts/rewind_stsb_winning.sh GPUID
      ./scripts/rewind_rte_winning.sh GPUID
      ```
      These scripts rewind the winning tickets at multiple sparsity levels. You can manually identify the set of super tickets as the set of winning tickets that perform the best among all sparsity levels.

2. Construct multi-task super tickets by aggregating the identified sets of super tickets of all tasks. E.g., to use the super tickets identified by us, run
   ```
   python construct_mtl_mask.py
   ```
   You can modify the script to use the super tickets identified by yourself.

3. MTL with tickets sharing. Run
   ```
   ./scripts/train_mtdnn.sh GPUID
   ```

</br>

## MTL Benchmark

MTL evaluation results on GLUE dev set averaged over 5 random seeds.

| Model        |  MNLI-m/mm (Acc)  |      QNLI (Acc)    |  QQP (Acc/F1) |     SST-2 (Acc)    | MRPC (Acc/F1) |     CoLA (Mcc)     |  STS-B (P/S) |      RTE (Acc)    |  Avg Score  |  Avg Compression  |
| -------------------- | ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- |  ----------- | ----------- | ----------- | ----------- |
|         MTDNN, base  |  84.6/84.2  |  90.5  |  90.6/87.4  |  92.2 |   80.6/86.2   | 54.0  |  86.2/86.4  |  79.0  |  82.4  | 100%  |
| Tickets-Share, base  |  84.5/84.1  |  91.0  |  90.7/87.5  |  92.7 |   87.0/90.5   | 52.0  |  87.7/87.5  |  81.2  |  83.3  | 92.9% |
|         MTDNN, large |  86.5/86.0  |  92.2  |  91.2/88.1  |  93.5 |   85.2/89.4   | 56.2  |  87.2/86.9  |  83.0  |  84.4  | 100%  |
| Tickets-Share, large |  86.7/86.0  |  92.1  |  91.3/88.4  |  93.2 |   88.4/91.5   | 61.8  |  89.2/89.1  |  80.5  |  85.4  | 83.3% |



</br>

## Citation

```
@article{liang2021super,
  title={Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization},
  author={Liang, Chen and Zuo, Simiao and Chen, Minshuo and Jiang, Haoming and Liu, Xiaodong and He, Pengcheng and Zhao, Tuo and Chen, Weizhu},
  journal={arXiv preprint arXiv:2105.12002},
  year={2021}
}

@article{liu2020mtmtdnn,
  title={The Microsoft Toolkit of Multi-Task Deep Neural Networks for Natural Language Understanding},
  author={Liu, Xiaodong and Wang, Yu and Ji, Jianshu and Cheng, Hao and Zhu, Xueyun and Awa, Emmanuel and He, Pengcheng and Chen, Weizhu and Poon, Hoifung and Cao, Guihong and Jianfeng Gao},
  journal={arXiv preprint arXiv:2002.07972},
  year={2020}
}
```

</br>

## Contact Information
For help or issues related to this package, please submit a GitHub issue. For personal questions related to this paper, please contact Chen Liang (cliang73@gatech.edu).
