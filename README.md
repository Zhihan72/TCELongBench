# TCELongBench

The official dataset and code for TCELongBench are provided here. See our paper "[Analyzing Temporal Complex Events with Large Language Models? A Benchmark towards Temporal, Long Context Understanding](https://arxiv.org/abs/2406.02472)" for further details.

We refer to the complex events composed of many news articles over an extended period as Temporal Complex Event (TCE). This paper proposes a novel approach using Large Language Models (LLMs) to systematically extract and analyze the event chain within TCE, characterized by their key points and timestamps. We establish a benchmark, named TCELongBench, to evaluate the proficiency of LLMs in handling temporal dynamics and understanding extensive text. This benchmark encompasses three distinct tasks - reading comprehension, temporal sequencing, and future event forecasting. In the experiment, we leverage retrieval-augmented generation (RAG) method and LLMs with long context window to deal with lengthy news articles of TCE. Our findings indicate that models with suitable retrievers exhibit comparable performance with those utilizing long context window.

# Dataset

Please click [this link](https://drive.google.com/drive/folders/1sca15cVDE9zkersh2kT510HPwtgCakp6?usp=sharing) to download our dataset, including news articles and outlines of TCEs and TCELongBench.

## TCE

```
{
  "ce_id": xxx,
  "date": xx-xx-xxxx,
  "Md5_id": xxxxxxxxxxxxxxx,
  "text": "xxxxxxx"
}
```

## Outline

```
{
  "ce_id" :
  "date" :
  "point" :
  "point_id" :
  "denoise_val":
  "dup_loc" :
  "dup_val" :
  "sim_loc" :
  "sim_val" :
  "keep_val" :
}
```

## TCELongBench

For TLB-detail task
```
{
  "ce_id" : xxx,
  "Md5_1" : xxxxxxxxxxxxxxx,
  "Md5_2" : xxxxxxxxxxxxxxx,
  "day_1" : xx-xx-xxxx,
  "day_2" : xx-xx-xxxx,
  "point_id" : xx,
  "question" : "xxxxxxx",
  "answer" : "xxxxxxx",
  "evidence" : "xxxxxxx",
  "choices" : "xxxxxxx",
  "shuffle_order" : "xxxxxxx",
  "ground_truth" : X
}
```

For TLB-order task
```
{
  "ce_id" :
  "common_ent" :
  "points_id" :
  "day" :
  "choices" :
  "ground_truth" :
}
```

For TLB-forecast task
```
{
  "ce_id" : xxx,
  "Md5_1" : xxxxxxxxxxxxxxx,
  "Md5_2" : xxxxxxxxxxxxxxx,
  "day_1" : xx-xx-xxxx,
  "day_2" : xx-xx-xxxx,
  "point_id" : xx,
  "question" : "xxxxxxx",
  "answer" : "xxxxxxx",
  "evidence" : "xxxxxxx",
  "choices" : "xxxxxxx",
  "shuffle_order" : "xxxxxxx",
  "ground_truth" : X
}
```

# Code

## Outline Extraction

## Generate-then-verify Paradigm

## Experiment

