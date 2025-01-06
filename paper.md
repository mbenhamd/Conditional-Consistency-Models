# Conditional Consistency Guided Image Translation

# and Enhancement

## Amil Bhagat, Milind Jain, A. V. Subramanyam

```
Indraprastha Institute of Information Technology (IIIT), Delhi, India
Email:{amil21309@iiitd.ac.in, milind21165@iiitd.ac.in, subramanyam@iiitd.ac.in}
```
```
Abstract—Consistency models have emerged as a promising
alternative to diffusion models, offering high-quality generative
capabilities through single-step sample generation. However,
their application to multi-domain image translation tasks, such
as cross-modal translation and low-light image enhancement
remains largely unexplored. In this paper, we introduce Con-
ditional Consistency Models (CCMs) for multi-domain image
translation by incorporating additional conditional inputs. We
implement these modifications by introducing task-specific con-
ditional inputs that guide the denoising process, ensuring that
the generated outputs retain structural and contextual infor-
mation from the corresponding input domain. We evaluate
CCMs on 10 different datasets demonstrating their effective-
ness in producing high-quality translated images across mul-
tiple domains. Code is available https://github.com/amilbhagat/
Conditional-Consistency-Models.
Index Terms—Conditional synthesis, consistency models, cross-
modal translation, low-light image enhancement, medical image
translation
```
### I. INTRODUCTION

```
Generative modelling has witnessed rapid progress in recent
years, evolving from the early successes of Generative Adver-
sarial Networks (GANs) [1] to the more stable and controllable
diffusion models [2]. In the early stages, the introduction of
conditional GANs (cGANs) [3] made it possible to guide
the generation process using auxiliary information such as
class labels or paired images. This allowed models to learn
structured mappings between input and target domains in a
variety of conditional image-to-image translation tasks, includ-
ing semantic segmentation, image synthesis, style transfer and
other applications [4], [5]. Despite their successes, conditional
GANs often suffer from mode collapse, training instability, and
the necessity of intricate adversarial training schemes.
On the other hand, score-based diffusion models [2] have
emerged as a robust alternative for image generation and
translation. These models add noise to the data and learn to
invert this process via score estimation, producing high-quality
outputs. However, these models require many iterative steps
to produce high quality images. Recently, consistency models
[6] have been proposed as a new class of generative models.
Instead of relying on iterative refinement steps like diffusion,
consistency models directly learn mappings from any noise
level to data.
However, despite their advantages, consistency models have
been explored primarily in unconditional settings. The ex-
tension of these models to conditional, multi-domain tasks
```
```
remains under-explored. This paper focuses on two representa-
tive tasks of significant practical value and complexity: cross-
modal translation and low-light image enhancement (LLIE).
```
```
Cross-modal translation is a crucial task in surveillance and
medical systems where different modalities may reveal com-
plementary information. The LLVIP dataset [7] exemplifies
the significance of this translation, as methods must effectively
learn visible to infrared mappings while preserving the spatial
details. Similarly, BCI [8] provides a challenging medical
dataset of paired hematoxylin and eosin (HE) and immuno-
histochemical techniques (IHC) images. As routine diagnosis
with IHC is very expensive, translation task from HE to IHC
can be very helpful in bringing down the overall cost. On the
other hand, LLIE is critical in computer vision applications
where images suffer from insufficient illumination, leading to
poor image details. Methods such as SID [9], GSAD [10] have
demonstrated impressive results using data-driven approaches
trained on paired dark and well-exposed images. Nonetheless,
these methods often rely on adversarial training setups which
are known to have convergence issues, or explicit iterative
sampling strategy which is time consuming.
```
```
While existing approaches to these problems range from
traditional Retinex-based methods [11] to deep convolutional
neural networks [9], many either lack the adaptability required
for multi-domain translation or are computationally expensive
at inference time. Conditional diffusion models can produce
excellent results but at the cost of significant computational
overhead during sampling [12]. Conditional GANs, though
fast, may exhibit mode collapse or require carefully tuned
adversarial training setups [13].
```
```
In this work, we introduce Conditional Consistency Models
(CCMs) for multi-domain image translation tasks. CCMs in-
novate on traditional consistency models by incorporating ad-
ditional conditional inputs, such as, visible image for visible to
infrared translation, HE image for HE to IHC translation, and
low-light image for LLIE, respectively. By carefully designing
the network architecture to take the conditional input and
training process, CCMs yield highly efficient inference while
preserving important local and global spatial details from
the conditional input. Unlike other techniques, our method is
applicable for both translation and image enhancement.
```
## arXiv:2501.01223v2 [cs.CV] 3 Jan 2025


### II. RELATEDWORKS

A. Cross-modal Image Translation

Generative approaches for image translation have evolved
significantly. In case of visible to infrared translation, GAN-
based methods have been widely explored, and models such
as ThermalGAN [14], TMGAN [15], and InfraGAN [16] have
achieved notable success in generating high-quality IR images.
These methods often require paired RGB-IR datasets, which
are scarce in practice. Leeet. al.propose an unsupervised
method using edge-guidance and style control in a mutli-
domain image translation framework. MappingFormer [17]
uses low and high frequency features for visible to infrared
image generation. These features are fused and a dual contrast
learning module further finds an effective mapping between
the unpaired data. Similarly, [18] also study the unpaired data
translation.
In case of medical images, BCI [8] addresses translation
of HE images to IHC images. They propose a multi-scale
structure in a Pix2Pix framework. The multi-scale pyramid
enables the model to capture both global and local features
effectively leading to improved image generation quality. On
similar lines, GAN based translation is investigated in [19]–
[21].

B. Low Light Image Enhancement

GAN Based Methods:EnlightenGAN [22] adopts a global
and local discriminator to directly map low-light images to
normal-light images. This is trained in unsupervised manner.
Similarly, other unsupervised methods show further improve-
ment in the enhanced images [23].
Diffusion Based Methods: Diff-Retinex [24] combines
Retinex decomposition with diffusion-based conditional gen-
eration. It separates images into illumination and reflectance
components, which are then adjusted using diffusion networks.
ExposureDiffusion [25] integrates physics-based exposure pro-
cesses with a diffusion model. Instead of starting from pure
noise, it begins with noisy images. It uses adaptive residual
layers to refine image details and achieves better generalization
and performance with fewer parameters. AnlightenDiff [26]
formulates the problem of LLIE as residual learning. It decom-
poses the difference between normal and low light image into a
residual component and a noise term. PyDiff [27] introduces a
pyramid resolution sampling strategy to speed up the diffusion
process and employs a global corrector to address degradation
issues. GSAD [10] uses a curvature-regularized trajectory in
the diffusion process to preserve fine details and enhance
global contrast. It also incorporates uncertainty-guided reg-
ularization to adaptively focus on challenging regions. These
methods use paired datasets.

C. Preliminaries: Consistency Models

Consider a continuous trajectory{rt}t∈[ε,T]connecting a
clean data samplerε at timeεto a noisy samplerT at a
larger timeT. The probability flow ODE ensures a bijective
mapping between data and noise at different time scales. A
consistency model [6] leverages this structure to learn a direct,

```
single-step mapping from a noisy sample at any timetback
to the clean samplerε. Formally, a consistency model defines
a consistency functiongφsuch that:
gφ(rt,t) =rε ∀t∈[ε,T]. (1)
```
```
Thisself-consistencyproperty ensures that no matter the noise
levelt, the model consistently recovers the same underlying
clean data sample.
Boundary Condition and Parameterization:A key require-
ment is theboundary condition, stating that at the minimal
noise levelε, the model should act as the identitygφ(rε,ε) =
rε. To naturally incorporate this condition, consistency models
employ a parameterization that respects this constraint. A
common approach uses a combination of skip connections and
time-dependent scaling:
```
```
gφ(r,t) =askip(t)r+aout(t)Gφ(r,t), (2)
where,askip(t)andaout(t)are scalar functions oftthat regulate
the contributions of the inputr and the learned function
Gφ(r,t), respectively. The boundary condition is satisfied by
settingaskip(ε) = 1andaout(ε) = 0, ensuring thatgφ(r,ε) =r.
As the noise scaletincreases, the influence ofGφ(r,t)
grows throughaout(t), allowing the model to move away
from the identity mapping and perform increasingly complex
denoising transformations. This parameterization naturally em-
beds the boundary condition in the model structure and main-
tains differentiability across noise scales.
Training Consistency Models:Training a consistency model
involves enforcing self-consistency at multiple noise levels.
Given a sampler, a noise vectorz∼ N( 0 ,I), and a pair of
noise scales(tn,tn+1)∈[ε,T], we form noisy inputsrtn=
r+tnz, rtn+1=r+tn+1z. The modelgφshould produce
the same clean outputrεfor bothrtn andrtn+1. Thus, the
training loss encourages:
gφ(rtn+1,tn+1) =gφ(rtn,tn) =rε.
```
```
By minimizing a suitable distance measure between these
outputs, the network learns to invert the noise injection step
at any arbitrary noise levelt.
III. PROPOSEDMETHOD
Given a paired datasetD ={vi,ri}Ni=1 of RGBvi ∈
RH×W×Cand its corresponding pairri∈RH×W×C. In case
of visible to infrared image translation,vis the RGB image
andris the infrared image. For BCI dataset [8],vis the
HE image andris the IHC image. In case of LLIE, vis
the ill-exposed image andris the well-exposed image. Our
aim is to learn a mapping from a given input and condition
image to its respective paired image. More formally, we learn
a functionG:v×r→r, wherev×rdenotes a pair. In our
proposed method,Gis parameterized via a consistency model
[28]. In order to generate images which semantically align
with the input image, we make use of conditional synthesis.
Here, we use an image pair (v,r) as the condition and input,
respectively, and train the model to generate onlyr. Figure 1
illustrates the proposed method.
```

A. Conditional Consistency

To adapt consistency models to conditional multi-domain
image translation, we introduce a conditional inputv. We
define the conditional consistency function as,

```
gφ(r,v,t) =
```
### (

```
r t=ε,
Gφ(r,v,t), t∈(ε,T],
```
### (3)

wherevprovides conditional information that guides the
denoising process. Asgφinverts the noise injection step, it
also conditions onvto ensure that the generated imager
corresponds to the infra-red, IHC, or well-exposed pair ofv.
In addition, the consistency function must follow the bound-
ary condition, that is:

```
gφ(rε,v,ε) =rε. (4)
```
As the consistency function must map noisy data from any
time-step to the clean data, using eq 3 and eq 4, we can write,

gφ(rtn+1,v,tn+1) =gφ(rtn,v,tn) =rε.
We parameterize the conditional consistency model as,
gφ(r,v,t) =askip(t)r+aout(t)Gφ(r,v,t). (5)
In eq 5, the boundary condition, wheregφ(r,v,t) =r, is
obtained fort=ε,askip(ε) = 1andaout(ε) = 0. This ensures
that the model outputs the clean imagerat the minimal noise
scale. As the noise scale increases, the generative component
Gφ(r,v,t)plays a more significant role, allowing the network
to learn complex transformations required for translatingv
intor. We present the training process in Algoirthm 1.

Fig. 1. Model Architecture. Our model can take a pair of visible-infrared, HE-
IHC, or, low light and well-exposed images. Visible, HE or low light image
acts as a conditional input. The noise as per time steptis added to the input
infrared, IHC or well exposed image. This noisy image is then concatenated
with the condition input and fed to the U-Net. The model can be sampled to
obtain the infrared, IHC or enhanced image.

B. Training and Sampling

Training Algorithm:In order to generate the desired output
given the inputs, we make use of conditional synthesis. Here,
we use an image pair (v,r) as the input and train the model
to generate onlyr.Gφtakes a 2C channel input and gives
aCchannel output. In our method, we concatenatevandr
across channels to obtain a 2 Cchannel input.vonly acts as
a conditional input and we do not add any noise to it.
We adopt the step scheduleM(.)as given in [28] and the
distance functiond(.,.)is pseudo-Huber loss given in [28].

```
Algorithm 1:Conditional Consistency Training (CCT)
Input:Paired datasetD={(vi,ri)}Ni=1, initial model
parameterφ, learning rateη, step schedule
M(·), distance functiond(·,·), weighting
functionλ(·)
Initializeφ−←φ,k← 0 ;
repeat
Sample(v,r)∼Dandn∼U[1,M(k)−1]
Samplez∼N( 0 ,I)
Compute the loss:
```
```
L(φ,φ−)←λ(tn)d
```
### 

```
gφ(r+tn+1z,v,tn+1),
```
```
gφ−(r+tnz,v,tn)
```
### 

```
Update the model parameters:
```
```
φ←φ−η∇φL(φ,φ−)
φ−←φ
```
```
Incrementk←k+ 1
untilConvergence;
```
```
Sampling : The unconditional sampling procedure starts
from Gaussian noiseˆrTand evaluates the consistency model
r←gφ(ˆrT,T). To incorporate conditions, we now have:
```
```
r←gφ(ˆrT,v,T). (6)
In practice, we concatenate theC channel noise with the
condition inputvand evaluategφ. This modified algorithm
ensures that sampling is guided by the conditional inputv.
We sample through single-step generation only.
```
```
IV. EXPERIMENT
In the experiments we show that our method is applicable to
different tasks and based on the conditional input, the paired
output can be generated. We compare our results with SOTA
methods in image translation and LLIE tasks, and demonstrate
that our method achieves competitive results. Additionally, we
would like to emphasize that our method generalizes well to
medical dataset also.
```
```
A. Datasets
We evaluate our proposed method on multiple datasets,
including LLVIP [7], BCI [8], LOLv1 [29], LOLv2 [30],
and SID [9]. The LLVIP dataset comprises 15,488 pairs of
visible and thermal images captured under low-light condi-
tions, of which 12,025 pairs are for training and 3,463 are
reserved for testing. The BCI dataset comprises 9,746 images,
organized into 4,873 pairs of Hematoxylin and Eosin (HE),
and Immunohistochemistry (IHC) images. Among these, 3,
pairs are used for training, while the remaining 977 pairs are
used for testing. LOLv1 comprises 485 training-paired low-
light and normal-light images and 15 testing pairs. LOLv2 is
divided into LOLv2-real and LOLv2-synthetic subsets, each
containing 689 and 900 training pairs and 100 testing pairs,
```

respectively. For SID dataset, we used the subset captured
by the Sonyα7S II camera. It comprises 2,463 short-/long-
exposure RAW image pairs, of these, 1,865 image pairs are
used for training, while 598 pairs are used for testing. In
addition to the above benchmarks, we tested our method on
five unpaired datasets, LIME [31], NPE [32], MEF [33] and
DICM [34] and VV^1 , that are low-light images and used to
check the LLIE potential of a model.

B. Implementation Details

LLVIP images are randomly cropped to 512x512 and then
resized to 128x128 for training over 1000 epochs. In case of
BCI, images are randomly cropped to 256x256 and trained for
500 epochs. LOLv1, LOLv2-real LOLv2-synthetic and SID
images are randomly cropped to 128x128 for training over
500 epochs for SID, and 1500 epochs for the rest. The values
ofaskip(t),aout(t),λ(tn),M(.), andd(.,.)are set to default
values given in [28].

C. Evaluation Metrics

We utilize Peak Signal-to-Noise Ratio (PSNR) and Struc-
tural Similarity Index Measure (SSIM) to assess the quality of
the generated images. Additionally, for datasets lacking paired
data, we employ the Naturalness Image Quality Evaluator
(NIQE) [35].

```
V. RESULTS
```
A. LLVIP and BCI Results

We evaluate the performance on the LLVIP dataset in two
different ways. The image is either randomly cropped into
512x512, or the full image is resized to 256x256 pixels. We
present the results in Table I. Our model outperforms other
methods. We also show the qualitative results in Fig. 2. It can
be observed that different objects are well represented with
slight deterioration in spatial details.

```
TABLE I
RESULTS ONLLVIP. TEST IMAGE RESOLUTION IS 512 ×512. (*)
REPRESENTS EVALUATION WHERE THE FULL IMAGE WAS RESIZED TO
256 ×256. HIGHER VALUES INDICATE BETTER PERFORMANCE. BEST
METHOD IS HIGHLIGHTED IN BOLD FONT.
```
```
Methods PSNR (dB) SSIM
CycleGAN [8] 11.22 0.
BCI [8] 12.19 0.
pix2pixGAN* [7] 10.76 0.
Ours 13.11 0.
Ours* 12.59 0.
```
The results for BCI are reported in Table II. Our method
performs significantly better in SSIM scores and gives com-
petitive values for PSNR despite being trained on 256x256 and
evaluated on 1024x1024 full image. The qualitative results are
shown in Fig. 3.

(^1) https://sites.google.com/site/vonikakis/datasets
(a) (b) (c)
Fig. 2. Comparison of (a) Visible, (b) Ground Truth Infrared, and (c)
Generated Infrared images.
TABLE II
RESULTS ONBCI. TEST IMAGE RESOLUTION IS 1024 ×1024. BEST
METHOD IS HIGHLIGHTED IN BOLD FONT.
Methods PSNR (dB) SSIM
CycleGAN [8] 16.20 0.
LASP[19] 17.86 0.
PSPStain [21] 18.62 0.
PPT [20] 19.09 0.
pix2pixHD [8] 19.63 0.
BCI [8] 21.16 0.
Ours 18.29 0.
(a) (b) (c)
Fig. 3. Comparison of (a) HE, (b) Ground Truth IHC, and (c) Generated IHC
images
B. LOL-v1, LOL-v2 and SID Results
We report comparisons for LOLv1, LOLv2-real, and
LOLv2-synthetic datasets in Table III. Our method shows a
strong performance. However, the results are lower than the
SOTA methods. We show the qualitative results in Figure 4.
We can see that the generated images closely matches to the
ground truth.
The results for the Sony subset of the SID dataset [9] are


```
TABLE III
COMPARISON WITHSOTA METHODS. BESTMETHOD IS HIGHLIGHTED IN BOLD FONT.
```
```
Methods PSNR (dB)LOL-v1SSIM PSNR (dB)LOL-v2-realSSIM PSNR (dB)LOL-v2-syntheticSSIM PSNR (dB)SID SSIM
```
```
SID [9] 14.35 0.43 13.24 0.44 15.04 0.61 16.97 0.
IPT [36] 16.27 0.50 19.80 0.81 18.30 0.81 20.53 0.
UFormer [37] 16.36 0.77 18.82 0.77 19.66 0.87 18.54 0.
Sparse [30] 17.20 0.64 20.06 0.81 22.05 0.90 18.68 0.
RUAS [38] 18.23 0.72 18.37 0.72 16.55 0.65 18.44 0.
FIDE [39] 18.27 0.66 16.85 0.67 23.22 0.92 19.02 0.
AnlightenDiff [26] 21.72 0.81 20.65 0.83 - - - -
Diff-Retinex [24] 21.98 0.86 - - - - - -
SNR-Net [40] 24.61 0.84 21.48 0.84 24.14 0.92 22.87 0.
Retinexformer [41] 25.16 0.84 22.80 0.84 25.67 0.93 24.44 0.
PyDiff [27] 27.09 0.93 24.01 0.87 19.60 0.87 - -
GSAD [10] 27.84 0.87 28.82 0.89 28.67 0.94 - -
Ours 21.10 0.78 22.72 0.79 22.00 0.87 20.97 0.
```
shown in Table III. We can see that the proposed method
achieves notable performance.

```
(a) (b) (c) (d)
```
Fig. 4. Comparison of results on different datasets: Firs row: LoL-v1, Second
Row: LoLv2-real, Last row: LoLv2-synthetic. Columns represent: (a) Low-
Light Input, (b) RetinexFormer, (c) Ours, and (d) Ground Truth.

```
TABLE IV
NIQESCORES ARE COMPUTED WITH THE FULL RESOLUTION OF THE
IMAGES. LOWER VALUES INDICATE BETTER PERFORMANCE.
```
```
Methods DICM LIME MEF NPE VV
EnGAN [22] 3.57 3.71 3.23 4.11 -
KinD [42] - 3.88 3.34 3.92 -
DCC-Net [43] 3.70 4.42 4.59 3.70 3.
GSAD [10] 3.28 4.32 3.40 3.55 2.
Retinexformer [41] 2.85 3.70 3.14 3.64 2.
Ours 3.09 3.67 2.96 3.65 4.
```
C. NIQE Scores

We compare the naturalness image quality (NIQE) across
DICM, LIME, MEF, NPE, and VV datasets in Table IV. Our
model achieves the best results for LIME and MEF, and the
second-best results for DICM. As the model does not see
these datasets during training, better NIQE scores indicate

```
(a) (b) (c)
Fig. 5. First row: LIME, Second row: NPE, Third row: MEF, Fourth row:
DICM, Last row: VV. Columns: (a) Input Image, (b) RetinexFormer, (c) Ours.
```
```
stronger generalization to unseen domains. For VV, a high-
resolution dataset, using a model trained on LoLv2-synthetic
with 128x128 patches causes a performance drop.
Compared to SOTA method GSAD [10] in LOL-v1 and
LOL-v2, we perform better in DICM, LIME and MEF. In
case of VV, [10] performs better. Compared to SOTA method
Retinexformer [41] in SID, our method does better for LIME
and MEF, and gives approximately same performance in NPE.
The qualitative results in Fig. 5 clearly show enhanced details.
```

### VI. CONCLUSION

In this work, we introduced Conditional Consistency Mod-
els (CCMs) for cross-modal image translation and enhance-
ment tasks. The conditional input guides the denoising process
and generates output corresponding to the paired conditional
input. Unlike existing methods such as GANs and diffusion
models, CCMs achieve superior results without requiring
iterative sampling or adversarial training. Distinct from other
works, our method can be adopted for different tasks of
translation or enhancement and shows competitive results in
both the tasks. Extensive experiments on benchmark datasets
demonstrate the superior performance of our method. In future,
we aim to explore further generalization of CCMs to additional
conditional tasks and investigate improvements in conditional
guidance mechanisms.


