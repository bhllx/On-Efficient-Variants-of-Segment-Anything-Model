# On-Efficient-Variants-of-Segment-Anything-Model
[[Paper]](https://arxiv.org/abs/2410.04960)

The Segment Anything Model (SAM) is a foundational model for image segmentation tasks, known for its strong generalization across diverse applications. However, its impressive performance comes with significant computational and resource demands, making it challenging to deploy in resource-limited environments such as mobile devices. To address this, a variety of SAM variants have been proposed to enhance efficiency without sacrificing accuracy. This survey provides the first comprehensive review of these efficient SAM variants. We begin by exploring the motivations driving this research. We then present core techniques used in SAM and model acceleration. This is followed by an in-depth analysis of various acceleration strategies, categorized by approach. Finally, we offer a unified and extensive evaluation of these methods, assessing their efficiency and accuracy on representative benchmarks, and providing a clear comparison of their overall performance.

![image](https://github.com/user-attachments/assets/025e0687-a66c-4a90-a9a2-a0c0b667423a)


## Content

## Efficient SAM Variants
### Accelerating SegAny
Segment Anything (SegAny), i.e. the promptable segmentation task, is the **foundation task of SAM**, whose goal is to return a valid mask with any given prompt (e.g. a point, a box, a mask, and text). 

Variants below focus on accelerating SegAny: 

| Model | Paper | Code | Key Features |
|:---:|:---:|:---:|:---:|
|FastSAM|[arXiv](https://arxiv.org/abs/2306.12156)|[Github](https://github.com/CASIA-IVA-Lab/FastSAM)|Reformulate SAM’s pipeline with YOLOv8-Seg for SegEvery and the later prompts-guided selection for SegAny.|
|SqueezeSAM|[arXiv](https://arxiv.org/abs/2312.06736)||Substitute SAM’s architecture with UNet-based encoder-decoder.|
|EfficientSAM|[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xiong_EfficientSAM_Leveraged_Masked_Image_Pretraining_for_Efficient_Segment_Anything_CVPR_2024_paper.pdf)|[Github](https://github.com/yformer/EfficientSAM)|Leverage SAMI pre-trained ViT-T/ViT-S as lightweight image encoder|
|RAP-SAM|[arXiv](https://arxiv.org/abs/2401.10228)|[Github](https://github.com/xushilin1/RAP-SAM)|Construct with a lite backbone and a unified dynamic convolution decoder, with addpters for multi-purpose segmentation.|
|SAM 2|[arXiv](https://arxiv.org/abs/2408.00714)|[Github](https://github.com/facebookresearch/sam2)|Apply Hiera as backbone and introduce memory mechanism for video tasks.|
|MobileSAM|[arXiv](https://arxiv.org/abs/2306.14289)|[Github](https://github.com/ChaoningZhang/MobileSAM)|Leverage encoder-only distillation from SAM’s ViT to MobileSAM’s TinyViT.||
|ESAM|[ResearchGate](https://www.researchgate.net/publication/375895620_Efficient_SAM_for_Medical_Image_Analysis)||Replace the image encoder with EfficientFormerV2 and conduct holistic distillation from a expert model.|
|NanoSAM||[Github](https://github.com/NVIDIA-AI-IOT/nanosam)|Distill from MobileSAM with ResNet18 as backbone and optimize with TensorRT.|
|RepViT-SAM|[arXiv](https://arxiv.org/abs/2312.05760)|[Github](https://github.com/THU-MIG/RepViT/tree/main/sam)|Substitute the image encoder with pure CNN-based RepViT and leverage MobileSAM’s distillation pipeline.|
|EdgeSAM|[arXiv](https://arxiv.org/abs/2312.06660)|[Github](https://github.com/chongzhou96/EdgeSAM)|Substitue SAM’s image encoder with RepViT and adopt a novel prompt-in-the-loop distillation|
|EfficientViT-SAM|[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.pdf)|[Github](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_sam)|Adopt the EfficientViT with ReLU linear attention as backbone and distill it from ViT-H.|
|FastSAM3D|[MICCAI2024](https://papers.miccai.org/miccai-2024/312-Paper2456.html)|[Github](https://github.com/arcadelab/FastSAM3D)|Replace the image encoder with a ViT-Tiny variant and incorporate the Dilated Attention and FlashAttention for efficiency.|
|SAM-Lightening|[arXiv](https://arxiv.org/abs/2403.09195)||A 2D version of FastSAM3D.|
|RWKV-SAM|[arXiv](https://arxiv.org/abs/2406.19369)||Adopt linear attention model RWKV into building efficient image encoder.|
|TinySAM|[arXiv](https://arxiv.org/abs/2312.13789)|[Github](https://github.com/xinghaochen/TinySAM)|Leverage full-stage distillation with TinyViT as backbone, and adopt 8-bit quantization on encoder to get Q-TinySAM, and propose the hierarchical sampling strategy to accelerate SegEvery task.|
|PTQ4SAM|[CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Lv_PTQ4SAM_Post-Training_Quantization_for_Segment_Anything_CVPR_2024_paper.pdf)|[Github](https://github.com/chengtao-lv/PTQ4SAM)|Eliminate the detrimental modal distribution and take the adaptive quantization on different distribution.|
|PQ-SAM|[ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01627.pdf)||Transfer the activation distribution into quantization-friendly distribution by truncating, grouping and learnable transformation.|  
|SlimSAM|[NeurIPS2024](https://nips.cc/virtual/2024/poster/94649)|[Github](https://github.com/czg1225/SlimSAM)|Divide image encoder into two substructures and conduct structured pruning in an alternative manner.|
|SuperSAM|[arXiv](https://arxiv.org/abs/2501.08504)||Apply the one-shot Neural Architecture Search with pruning-based methods to build up a supernetwork of SAM.|
|SAMfast|[PyTorch Blog](https://pytorch.org/blog/accelerating-generative-ai/)|[Github](https://github.com/pytorch-labs/segment-anything-fast)|A rewrote version of SAM with pure, nature Pytorch optimizations.|

### Accelerating SegEvery
Segment Everything (SegEvery), i.e. the all-masks generation task, is an extension of SegAny task, which aims to segment all objects in a picture.

Variants below focus on accelerating SegEvery: 

| Model | Paper | Code | Key Features |
|:---:|:---:|:---:|:-------------:|
|FastSAM|[arXiv](https://arxiv.org/abs/2306.12156)|[Github](https://github.com/CASIA-IVA-Lab/FastSAM)|Directly leverage YOLOv8-Seg to segment everything in high efficiency.|
|MobileSAMV2||[Github](https://github.com/ChaoningZhang/MobileSAM)|Object-aware prompt sampling based on the external YOLOv8 detector.|
|TinySAM|[arXiv](https://arxiv.org/abs/2312.13789)|[Github](https://github.com/xinghaochen/TinySAM)|Hierarchical sampling strategy for efficient prompts selection.|
|Lite-SAM|[ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05077.pdf)||LiteViT as lightweight backbone and AutoPPN for efficient prompts generation.|
|AoP-SAM|[OpenReview](https://openreview.net/forum?id=mrs7Z7eoyT)||Generate prompts iteratively by coarse prediction and fine-grained filtering.|

**Note**: Variants like FastSAM and TinySAM propose efficient strategies for both tasks, so we put them in both lists.

## Citation
```
  @artical{sun2024efficientvariantssegmentmodel,
        title={On Efficient Variants of Segment Anything Model: A Survey}, 
        author={Xiaorui Sun and Jun Liu and Heng Tao Shen and Xiaofeng Zhu and Ping Hu},
        journal={arXiv preprint arXiv:2410.04960},
        year={2024}
  }
