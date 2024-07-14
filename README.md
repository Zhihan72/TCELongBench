# TCELongBench

The official dataset and code for TCELongBench are provided here. See our paper "[Analyzing Temporal Complex Events with Large Language Models? A Benchmark towards Temporal, Long Context Understanding](https://arxiv.org/abs/2406.02472)" for further details.

We refer to the complex events composed of many news articles over an extended period as Temporal Complex Event (TCE). This paper proposes a novel approach using Large Language Models (LLMs) to systematically extract and analyze the event chain within TCE, characterized by their key points and timestamps. We establish a benchmark, named TCELongBench, to evaluate the proficiency of LLMs in handling temporal dynamics and understanding extensive text. This benchmark encompasses three distinct tasks - reading comprehension, temporal sequencing, and future event forecasting. In the experiment, we leverage retrieval-augmented generation (RAG) method and LLMs with long context window to deal with lengthy news articles of TCE. Our findings indicate that models with suitable retrievers exhibit comparable performance with those utilizing long context window.

# Dataset

Please click [this link](https://drive.google.com/drive/folders/1sca15cVDE9zkersh2kT510HPwtgCakp6?usp=sharing) to download our dataset, including news articles and outlines of TCEs and TCELongBench.

## TCE

News articles within TCEs are saved in `TCE_News_Articles.json` and contains the following attributes:

```
{
  "ce_id": [int] The TCE id,
  "date": [str] The date of this news article,
  "Md5_id": [str] The unique id of this news article,
  "text": [str] The content of this news article,
}
```

Note that `TCE_ce_id.txt` contains the list of TCE id that is used in constructing our TCELongBench.

## Outline

Outline points within TCEs are saved in `outline_points.csv` and contains the following attributes:

```
{
  "ce_id": [int] The TCE id,
  "date": [str] The date of this outline point,
  "point": [str] The content of this outline point,
  "point_id": [int] This id is used for differentiating the outline points in the same TCE on the same date. We use the combination of date and point_id, i.e. (date, point_id), to identify each outline point in the TCE,
  "denoise_val": [bool] 1 for noising outline point that should be deleted; otherwise 0,
  "dup_loc": [list] This list consists of the information of points that this outline point duplicate. We use (date, point_id) to locate these points and save them in a list, 
  "dup_val": [bool] 1 for outline point that duplicate point(s) with earlier date(s); otherwise 0,
  "sim_loc": [list] This list consists of the information of points that share high similarity scores with this outline point . We use (date, point_id) to locate these points and save them in a list, 
  "sim_val": [bool] 1 for outline point that shares high similarity score(s) with point(s) with earlier date(s); otherwise 0,
  "keep_val": [bool] 1 for outline point whose denoise_val, dup_val and sim_val are all 0,
}
```

## TCELongBench

### TLB-detail

The training, development, and test sets of TLB-detail task are saved in `TLB_detail_TrainSet.csv`, `TLB_detail_DevSet.csv`, and `TLB_detail_TestSet.csv`, respectively. They contain the following attributes:
```
{
  "ce_id" : [int] The TCE id,
  "Md5_1" : [str] The unique id for the news article that support the ground truth,
  "Md5_2" : [str] The unique id for the news article that is used for creating noising choice,
  "day_1" : [str] The date of news article with Md5_1 id,
  "day_2" : [str] The date of news article with Md5_2 id,
  "point" : [str] The outline point used for generating the question,
  "point_id" : [str] The point id for this outline point,
  "question" : [str] The generated question,
  "answer" : [str] The correct answer to this question,
  "choices" : [str] Choices for this question,
  "shuffle_order" : [str] The shuffled order of four choices, in which (a) is the correct answer,
  "ground_truth" : [str] The order of the correct answer (a),
}
```
### TLB-order

The training, development, and test sets of TLB-order task are saved in `TLB_order_TrainSet.csv`, `TLB_order_DevSet.csv`, and `TLB_order_TestSet.csv`, respectively. They contain the following attributes:
```
{
  "ce_id": [int] The TCE id,
  "common_ent": [str] The common entity that the points share with each other,
  "points": [str] The points within this ordering question,
  "points_id": [str] The corresponding point ids of these points,
  "day": [str] The corresponding dates of these points,
  "choices" :
  "ground_truth" :
}
```

### TLB-forecast

The training, development, and test sets of TLB-forecast task are saved in `TLB_forecast_TrainSet.csv`, `TLB_forecast_DevSet.csv`, and `TLB_forecast_TestSet.csv`, respectively. They contain the following attributes:
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

# Code for Dataset Construction

## Outline Extraction

The prompt and code for outline extraction of TCE are saved in `outline_extraction` folder

## Generate-then-verify Paradigm

The prompt and code for generate-then-verify paradigm of TCE are saved in `generate_then_verify_paradigm` folder

# Contact

For any issues or questions, kindly email us at: ZHANG Zhihan (zhangzhihan22@m.fudan.edu.cn).

# Citation

```
@misc{zhang2024analyzingtemporalcomplexevents,
      title={Analyzing Temporal Complex Events with Large Language Models? A Benchmark towards Temporal, Long Context Understanding}, 
      author={Zhihan Zhang and Yixin Cao and Chenchen Ye and Yunshan Ma and Lizi Liao and Tat-Seng Chua},
      year={2024},
      eprint={2406.02472},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.02472}, 
}
```
