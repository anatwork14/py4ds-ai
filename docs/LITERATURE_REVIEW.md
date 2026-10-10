# Literature Review — Intel Natural Scene Classification

**Status:** source-verified background; actual results from this repository are documented in [`FINAL_REPORT.md`](FINAL_REPORT.md) and [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Checked 2026-10-08 (UTC+07:00 local time). Published paper results are not substituted for Intel dataset results.

## Research problem

Given a single photograph of a natural or urban scene, predict exactly one of six labels: buildings, forest, glacier, mountain, sea, street. Challenges include clutter, viewpoint and illumination changes, and visual overlap (e.g., buildings within a street scene or mountains beside a glacier). The public Intel dataset is distributed through https://www.kaggle.com/datasets/puneet6060/intel-image-classification. Widely reported counts are seg_train=14,034 labeled, seg_test=3,000 labeled and seg_pred=7,301 unlabeled images, typically 150×150; verify against the downloaded release before using numbers as this project's measurements. These counts are also stated in independent dataset-use documentation at https://github.com/Anshuman33/Nature-Images-Classification; that GitHub page is supporting context, **not** the original research dataset paper. The dataset license and publication provenance must be verified at download time.

**Question:** under the same train/validation/test split and measured compute budget, how much do local handcrafted features, pretrained fixed embeddings, and fine-tuned CNNs differ in predictive performance and class-specific errors?

## Reviewed sources and how they motivate the experiments

### 1. Dalal & Triggs (2005): Histogram of Oriented Gradients

**Verified citation:** Navneet Dalal and Bill Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005, pp. 886–893, DOI: 10.1109/CVPR.2005.177. Author publication index: https://lear.inrialpes.fr/pubs/

**Problem/method:** represent local edge and gradient-orientation structure with spatial histograms and contrast normalization for object appearance classification, originally **human detection**, not Intel natural-scene classification. **Why relevant:** inexpensive, understandable feature-engineering baseline; potentially good at texture/shape cues but unable to learn high-level semantic scene composition end to end. **Experiment:** grayscale, consistently resized image → HOG → training-fitted StandardScaler → optional training-fitted PCA → tuned LinearSVC. Compare performance per scene class; do not attribute any Intel score to this paper.

### 2. Lowe (2004): Scale-Invariant Feature Transform

**Verified citation:** David G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*, International Journal of Computer Vision 60, 91–110 (2004), DOI: 10.1023/B:VISI.0000029664.99615.94. Publisher: https://doi.org/10.1023/B:VISI.0000029664.99615.94

**Problem/method:** scale- and rotation-invariant keypoint detection/descriptors supporting matching across viewpoint, illumination and noise variation. **Why relevant:** local descriptors can represent distinctive scene structures. **Limitations:** variable number of features per image and computational cost; SIFT descriptors alone do not form fixed-length vectors. **Experiment:** use Bag-of-Visual-Words to aggregate SIFT descriptors; handle images with zero keypoints explicitly.

### 3. Csurka et al. (2004): Bag of Keypoints / Visual Words

**Verified citation:** Gabriella Csurka, Christopher R. Dance, Lixin Fan, Jutta Willamowski and Cédric Bray, *Visual Categorization with Bags of Keypoints*, ECCV workshop (2004). Paper: https://www.cs.princeton.edu/courses/archive/fall09/cos429/papers/csurka-eccv-04.pdf

**Problem/method:** quantize local image descriptors into a discrete visual vocabulary and summarize images as fixed-length visual-word histograms; the paper compares classifiers for visual categorization. **Why relevant:** converts variable-size SIFT descriptor sets into a form usable by SVM/LogisticRegression. **Limitations:** loses much of the descriptors' spatial configuration, and vocabulary quality depends on sampled training descriptors and vocabulary size. **Experiment:** learn MiniBatchKMeans vocabulary strictly on training samples, construct train/val/test histograms using that frozen vocabulary, tune vocab size by validation only.

A related and earlier vector-quantization / retrieval formulation is Sivic and Zisserman's *Video Google* (ICCV 2003), https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/. Distinguish visual **retrieval** from supervised scene **classification**.

### 4. He et al. (2016): Residual Networks

**Verified citation:** Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun, *Deep Residual Learning for Image Recognition*, CVPR 2016, pp. 770–778. Primary: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html

**Problem/method:** residual connections reformulate layers as residual functions, improving optimization of deep convolutional networks. **Why relevant:** ResNet18 provides an accessible backbone with pretrained ImageNet weights and a differentiable fine-tuning path. **Experiment:** compare frozen ImageNet ResNet18 embeddings + linear classifier, pretrained ResNet18 with trainable six-way head, and selected fine-tuning of final convolutional block. **Limitations:** pretrained ImageNet labels/domain differ from these six scenes; data augmentation and transfer protocol matter. The paper's ImageNet results are **not Intel results**.

### 5. Kornblith, Shlens & Le (2019): Transfer learning behavior

**Verified citation:** Simon Kornblith, Jonathon Shlens and Quoc V. Le, *Do Better ImageNet Models Transfer Better?*, CVPR 2019, pp. 2661–2671. Primary: https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html

**Problem/method:** systematic study of ImageNet-trained representations across downstream classification tasks with both fixed feature extraction and fine-tuning. **Finding relevant here:** transfer ability depends on training choices and task/domain; a pretrained backbone is valuable but its benefits should be **tested rather than assumed**. **Experiment:** hold image set constant and compare frozen representations against fine-tuning, with documented pretrained transforms and comparable metrics.

### 6. Selvaraju et al. (2017): Grad-CAM for visual explanations

**Verified citation:** Ramprasaath R. Selvaraju et al., *Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization*, ICCV 2017, pp. 618–626. Primary: https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html

**Problem/method:** use class gradients at a CNN's final convolutional feature map to form coarse class-discriminative localization heatmaps. **Why relevant:** helps inspect where a fine-tuned CNN is looking, especially mountain/glacier or buildings/street confusions. **Limitations:** heatmaps are qualitative/approximate explanations, not verified causal ground truth and not automatically valid for HOG/SVM classifiers. **Experiment:** save original image, predicted/truth labels, confidence, saliency overlay, layer used, and a short cautionary interpretation.

### Optional extensions (only if runtime budget permits)

- Mark Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018. Primary: https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html . Use if latency/parameter efficiency becomes a primary research question.
- Mingxing Tan and Quoc Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019, Proceedings of Machine Learning Research 97:6105–6114. Primary: https://proceedings.mlr.press/v97/tan19a.html . Use as an optional additional backbone. Its published ImageNet benchmark is **not** an Intel baseline.
- Tong He et al., *Bag of Tricks for Image Classification with Convolutional Neural Networks*, CVPR 2019, pp. 558–567. Primary: https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks_CVPR_2019_paper.html . Useful for controlled augmentation/optimization ablations.

## Hypotheses: proposals, not proven findings

- H1: pretrained ResNet18 embeddings + linear classifier will outperform at least one handcrafted-feature baseline on macro-F1.
- H2: end-to-end CNN fine-tuning may improve class-specific recall relative to a frozen feature extractor, but may overfit and consume more compute.
- H3: the six categories show systematic confusions in semantically overlapping scene pairs; inspect confusion matrices rather than selecting pairs in advance as guaranteed outcomes.
- H4: preprocessing choices (train/val split integrity and correct pretrained transforms) meaningfully affect the reliability of comparisons.

These hypotheses may be rejected. Report unfavorable outcomes honestly.

## Project experiment outcomes from saved artifacts

| Experiment ID | Representation | Learner | Tuned on | Intel validation macro-F1 | Intel test macro-F1 | Source artifact |
|---|---|---|---|---:|---:|---|
| E00 | none | training-majority class (`mountain`) | validation reference | 0.050619 | NOT RUN | `runs/seed-42-phash-reviewed/classical/hog/metrics.json` |
| E01 | HOG | LogisticRegression, C=0.01 | train/val | 0.676893 | NOT RUN | `runs/seed-42-phash-reviewed/classical/hog/metrics.json` |
| E02 | SIFT/BoVW, 128 words | LogisticRegression, C=1 | train/val | 0.592861 | NOT RUN | `runs/seed-42-phash-reviewed/classical/sift-bovw/metrics.json` |
| E03 | frozen ResNet18 | LinearSVC, C=0.1 | train/val | 0.903315 | NOT RUN | `runs/seed-42-phash-reviewed/resnet18/imagenet-v1/metrics.json` |
| E04 | pretrained ResNet18 | trainable 6-class head | train/val | 0.908711 | NOT RUN | `runs/seed-42-phash-reviewed/cnn18/head-imagenet-v1/metrics.json` |
| E05 | pretrained ResNet18 | layer4 fine-tuned | train/val; test once after freeze | 0.929137 | 0.931004 | `runs/seed-42-phash-reviewed/final-evaluation/metrics.json` |
| E06 optional | small CNN from scratch | end-to-end CNN | train/val | NOT RUN | NOT RUN | — |

Only E05 has a held-out test score because only the frozen winner was evaluated. The other test cells intentionally remain NOT RUN; do not infer or copy test scores for unselected candidates. The metrics and selected-checkpoint provenance are detailed in `FINAL_REPORT.md`.

## Protocol and source discipline

- Fix and record original raw dataset hashes, manifest creation code, exclusions and split counts before any model selection.
- Explicitly name main comparison metric (recommend macro-F1) and record accuracy, weighted F1, class-specific metrics, confusion matrices, runtime, parameter count and per-image predictions.
- Search/tune on training folds or a fixed validation partition; final test once after model/epoch/config selection. See scikit-learn's official cross-validation guide: https://scikit-learn.org/stable/modules/cross_validation.html and nested-CV example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html .
- Torchvision pretrained preprocessing must be extracted from the relevant weights object rather than guessed: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html (verify installed torchvision version).
- Document papers separately from the Intel benchmark and cite each claim next to its source.
- Verify page, year, venue, author list, DOI and actual method when converting this review into the final bibliography; avoid irrelevant citations or fabricated direct Intel-specific performance.
- This is a structured research starting point, not a systematic literature review; explicitly explain search inclusion/exclusion and review limitations if labeling the report "systematic."
