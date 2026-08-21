# FOROH Experimental Plan

FOROH의 실험은 단순한 benchmark 성능 비교가 아니라, **각 기여가 실제로 필요한지 하나씩 검증하는 구조**로 설계한다. 따라서 실험 순서도 baseline leaderboard가 아니라 FOROH의 핵심 주장에 대응하도록 배치한다.

## Core contributions

1. ordinal prediction을 class 또는 threshold 문제가 아니라 **single polar coordinate estimation**으로 재정의한다.
2. 각 grade를 prototype point가 아니라 **azimuthally free level set**으로 표현한다.
3. grade-specific parameter 없이 하나의 shared progression coordinate를 사용함으로써 **data-efficient ordinal inductive bias**를 제공한다.
4. latent progression coordinate와 grading convention을 분리함으로써 **score-system-independent representation**으로 확장할 가능성을 제공한다.

모든 실험은 위 기여 중 하나를 직접 검증해야 한다.

---

## 1. Is a Polar Ordinal Coordinate Necessary?

FOROH의 첫 번째 gate는 angular geometry 자체가 필요한지 확인하는 것이다.

FOROH:

$$
\theta = \arccos(u^\top w), \qquad
\hat y = \frac{\theta}{\pi}C_{\max}
$$

성능 향상이 실제 angular parameterization 때문인지, normalization 또는 projector capacity 때문인지 분리해야 한다.

### Compared models

| Model | Normalized embedding $u$ | Axis $w$ | Score |
|---|---:|---:|---|
| Matched Euclidean Huber | No | No | scalar regression |
| Normalized Cosine | Yes | Yes | $\frac{1-u^\top w}{2}C_{\max}$ |
| FOROH | Yes | Yes | $\frac{\arccos(u^\top w)}{\pi}C_{\max}$ |

Euclidean regression도 단순 `backbone -> Linear(1)`로 구현하지 않는다. FOROH와 동일한 projector depth, width, dropout, embedding budget을 사용하고 마지막 scalar readout만 변경한다.

반드시 맞출 조건:

- backbone
- projector depth / width
- embedding dimension
- dropout
- optimizer
- learning rate
- augmentation
- training schedule
- Huber loss
- data split

### Primary question

> Does angular parameterization provide benefit beyond scalar regression and normalization?

FOROH가 matched Euclidean regression 및 normalized cosine보다 일관된 이점을 보이지 못하면 이후의 geometric claim은 약해진다.

---

## 2. Does the Level-Set Representation Matter?

FOROH의 두 번째 핵심 기여는 grade를 point가 아니라 level set으로 표현하는 것이다.

$$
\mathcal R_y = \left\{\cos\theta_y w + \sin\theta_y v : v\in S^{d-2}\right\},
\qquad
\theta_y = \frac{\pi y}{C_{\max}}
$$

grade supervision은 polar coordinate만 제한하고 azimuthal coordinate는 직접 고정하지 않는다.

### Point-prototype control

모든 grade prototype을 동일 meridian 위에 둔다.

$$
a_y = \cos\theta_y w + \sin\theta_y q,
\qquad q\perp w
$$

모든 grade가 동일한 $q$를 공유한다. 이 경우 polar angle과 azimuth가 모두 고정된다.

### FOROH level set

FOROH에서는 $\theta=\theta_y$만 supervision하고 $v$는 자유롭게 둔다.

따라서 비교의 본질은 다음 하나다.

- Point prototype: polar angle + azimuth fixed
- FOROH: polar angle fixed, azimuth free

backbone, projector, hypersphere dimension, optimizer, grade angular positions는 모두 동일하게 유지한다.

### Evaluation

- MAE
- QWK
- Macro-F1
- same-grade intra-class angular spread
- between-grade angular separation
- validation-to-test generalization gap
- small-data condition에서의 degradation

### Primary question

> Is constraining only the ordinal coordinate better than collapsing each grade to a point?

이 실험은 FOROH의 level-set novelty를 직접 검증하는 핵심 ablation이다.

---

## 3. Where Does FOROH Sit Among Existing Ordinal Methods?

첫 두 gate에서 FOROH geometry의 필요성을 확인한 뒤 기존 ordinal methods와 동일 조건 비교를 수행한다.

### Principal comparison

| Method | Representative paradigm |
|---|---|
| CE | nominal classification |
| Matched Huber | scalar ordinal regression |
| CORAL | cumulative threshold ordinal regression |
| CORN | conditional ordinal regression |
| GOL | learned geometric ordinal representation |
| FOROH | predefined polar ordinal coordinate |

필요하면 이후 Weighted CE, CDW-CE, HCOR 계열 및 최근 ordinal methods를 추가한다.

### GOL comparison

GOL은 ordinal relation을 만족하는 flexible representation geometry를 학습한다. 반면 FOROH는 하나의 polar coordinate를 미리 정의한다.

핵심 비교는 단순 QWK leaderboard가 아니라:

- learned flexible ordinal geometry
- single-axis predefined ordinal geometry

사이의 차이다.

가능하면 progression 구조가 다른 task를 함께 사용해 FOROH가 어떤 종류의 ordinal problem에 특히 적합한지 확인한다.

---

## 4. Does the Shared Coordinate Improve Data Efficiency?

FOROH는 class별 decision structure 대신 하나의 progression coordinate를 공유한다. 의료 small-data 환경에서 이 inductive bias가 더 안정적인지 검증한다.

### Small-n experiment

training patients를 patient level에서 다음 비율로 subsampling한다.

- 100%
- 50%
- 25%
- 10%

각 setting은 최소 3개, 가능하면 5개의 random seed로 반복한다.

### Metrics

- MAE
- QWK
- Macro-F1
- minority-grade recall
- seed-wise standard deviation
- full-data 대비 performance drop
- degradation slope

$$
\Delta M(p)=M(p)-M(100\%)
$$

### Primary question

> Does FOROH degrade more slowly as supervision becomes scarce?

full-data 최고 성능보다 데이터가 줄어들수록 상대적으로 더 안정적인지가 더 중요한 evidence다.

---

## 5. Is the Shared Coordinate Robust to Sparse Endpoint Supervision?

small-n과 imbalance는 분리한다.

- small-n: 전체 cohort가 작아짐
- imbalance: 전체 sample budget은 최대한 유지하고 특정 grade supervision만 감소

예를 들어 rare severe grade retain ratio를 100%, 50%, 25%, 10%로 줄인다.

### Main metrics

- class-wise recall
- extreme-grade recall
- Macro-F1
- MAE
- extreme-grade MAE

### Hypothesis

> A shared ordinal coordinate may remain stable when endpoint supervision is sparse.

`arccos`가 minority gradient를 자동 증폭한다는 설명은 사용하지 않는다.

---

## 6. Equal-Spacing Stress Test

FOROH는

$$
\theta_y=\frac{\pi y}{C_{\max}}
$$

으로 adjacent grades에 equal angular spacing을 부여한다. 이는 implementation detail이 아니라 핵심 structural assumption이다.

underlying continuous variable을 가진 controlled setting에서 다음 discretization을 비교한다.

- Equal spacing
- Mildly unequal spacing
- Strongly unequal spacing

이상적인 characterization은 모든 setting에서 최고 성능이 아니라, unequal spacing이 커질수록 FOROH의 advantage가 감소하는 패턴이다.

이를 통해 다음과 같은 적용 범위를 정의한다.

> FOROH is particularly appropriate when ordinal labels approximate a single, approximately evenly spaced progression coordinate.

---

## 7. Single-Axis Stress Test

FOROH는 결국 하나의 sufficient ordinal coordinate를 가정한다.

$$
u \mapsto u^\top w \mapsto \theta
$$

synthetic experiment에서 latent dimensionality를 직접 조절한다.

### One-dimensional progression

$$
y=f(s_1)
$$

### Multi-dimensional progression

$$
y=f(s_1,s_2)
$$

필요하면 $y=f(s_1,s_2,s_3)$까지 확장한다.

FOROH와 GOL 같은 flexible geometric method를 함께 비교한다.

### Primary hypothesis

> FOROH should be strongest when ordinal progression is effectively one-dimensional.

latent dimensionality가 증가하면서 flexible geometry의 상대적 장점이 커지는지 본다. 이 실험은 FOROH를 universal ordinal model이 아니라 명시적인 single-axis inductive bias를 가진 방법으로 characterization한다.

---

## 8. Can the Coordinate Transfer Across Grading Systems?

앞선 equal-spacing / single-axis stress test에서 적용 조건을 확인한 뒤 수행한다.

FOROH의 central representation은 $\theta\in[0,\pi]$로 grade cardinality와 독립적이다. 새로운 grading system에서는 readout scale을 바꾼다.

$$
\hat y^{(C')}=\frac{\theta}{\pi}C'_{\max}
$$

동일한 underlying construct에 대해 서로 다른 ordinal discretization을 정의하고 Scheme A에서 학습한 representation을 Scheme B로 transfer한다.

비교 setting:

- zero-shot rescaling
- frozen representation + calibration
- few-shot adaptation
- head-only retraining
- full retraining

### Primary question

> Does FOROH learn a progression coordinate that survives a change in grading convention?

성공하면 latent progression과 grading convention을 분리할 수 있다는 임상적 주장을 직접 지지한다.

---

## 9. What Is Preserved in the Free Azimuth?

FOROH의 level-set formulation은

$$
u=\cos\theta w+\sin\theta v
$$

로 쓸 수 있다. 하지만 $v$가 자동으로 phenotype을 의미하거나 severity와 독립이라는 보장은 없다.

$$
v=\frac{u-(u^\top w)w}{\|u-(u^\top w)w\|}
$$

### Severity probes

- $\theta \rightarrow y$
- $v \rightarrow y$
- $u \rightarrow y$

### Phenotype probes

- $\theta \rightarrow$ phenotype
- $v \rightarrow$ phenotype
- $u \rightarrow$ phenotype

phenotype 분석은 반드시 same-grade subset 또는 grade-conditioned evaluation으로 수행한다.

이상적인 패턴은:

- $\theta$: severity strongly predictable
- $v$: severity weakly predictable
- $v$: within-grade phenotype predictable

이다. 단, $I(v;y)=0$을 구조적으로 보장한다고 주장하지 않는다. 이 실험의 목적 자체가 그것을 측정하는 것이다.

---

## 10. Temporal Extension

Temporal FOROH는 현재 static paper의 필수 실험으로 두지 않는다.

먼저 static setting에서 $\theta$가 meaningful ordinal coordinate인지, $v$가 useful free residual인지 검증한 뒤 진행한다.

longitudinal extension에서는

$$
u(t)=\cos\theta(t)w+\sin\theta(t)v(t)
$$

로 두고 $v_t$가 현재 severity 이상으로 미래 progression 정보를 제공하는지 확인한다.

$$
P(\Delta y_{t+1}\mid\theta_t,v_t)
\quad\text{vs}\quad
P(\Delta y_{t+1}\mid\theta_t)
$$

이 실험은 후속 연구로 남긴다.

---

# Experimental Priority

| Priority | Experiment | Contribution tested |
|---:|---|---|
| **1** | Euclidean Huber vs normalized cosine vs FOROH | polar angular coordinate의 필요성 |
| **2** | Point prototype vs level set | azimuthal freedom의 필요성 |
| **3** | CE / Huber / CORAL / CORN / GOL / FOROH | 기존 ordinal paradigm 대비 positioning |
| **4** | Small-n | shared coordinate의 data efficiency |
| **5** | Imbalance stress | sparse endpoint supervision robustness |
| **6** | Equal-spacing stress | equal-angle assumption의 적용 범위 |
| **7** | Single-axis synthetic stress | 1D progression assumption의 적용 범위 |
| **8** | Score-system transfer | grading-system-independent coordinate 가능성 |
| **9** | $v$ probing | level-set residual의 의미 |
| **10** | Temporal extension | longitudinal trajectory representation |

---

# Immediate Phase-1 Gate

현재 가장 먼저 계산해야 할 모델은 네 개로 제한한다.

1. **Matched Euclidean Huber**
2. **Normalized cosine regression**
3. **Hyperspherical point prototype**
4. **FOROH**

우선 하나의 대표 dataset과 동일 fold, 동일 backbone에서 수행한다. 현재 주력 설정은 LIMUC ResNet50 fold 0을 사용한다.

## Gate A

> FOROH > matched scalar/cosine controls?

angular coordinate 자체에 실질적인 가치가 있는가.

## Gate B

> FOROH level set > point prototype?

azimuthal freedom이 실제 generalization에 기여하는가.

두 질문에 긍정적인 결과가 확인된 뒤에만 다음으로 확장한다.

- multi-fold
- additional datasets
- GOL
- small-n
- score transfer

---

# Experimental Story

실험 섹션은 단순한 `baseline -> ablation -> 추가 분석`이 아니라 다음 질문을 순서대로 닫는다.

1. 단순 scalar regression과 normalization으로 설명되지 않는 angular coordinate의 효과가 있는가?
2. 동일 ordinal angle에서도 point collapse보다 level set이 유리한가?
3. 기존 ordinal / geometric methods 사이에서 FOROH는 어디에 위치하는가?
4. shared coordinate inductive bias가 small-n / imbalance에서 실제로 유효한가?
5. equal-spacing / single-axis assumption은 언제 맞고 언제 깨지는가?
6. 조건이 맞는 상황에서 $\theta$가 다른 grading convention으로 transfer되는가?
7. level-set이 남긴 자유도 $v$가 실제 non-severity information을 보존하는가?

최종 narrative:

> Is the coordinate necessary?  
> -> Is the level set necessary?  
> -> When does the inductive bias help?  
> -> What clinical capability does it enable?
