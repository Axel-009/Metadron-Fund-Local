# Qwen2.5-Omni
<p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/Omni_logo.png" width="400"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen25-Omni-a2505ce0d5514e">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwenlm.github.io/blog/qwen2.5-omni/">Blog</a>&nbsp&nbsp | &nbsp&nbsp📚 <a href="https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks">Cookbooks</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2503.20215">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo ">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni">API</a>
<!-- &nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://gallery.pai-ml.com/#/preview/deepLearning/cv/qwen2.5-vl">PAI-DSW</a> -->
</p>

We release **Qwen2.5-Omni**, the new flagship end-to-end multimodal model in the Qwen series. Designed for comprehensive multimodal perception, it seamlessly processes diverse inputs including text, images, audio, and video, while delivering real-time streaming responses through both text generation and natural speech synthesis. Let's click the video below for more information 😃

<a href="https://youtu.be/yKcANdkRuNI" target="_blank">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/video_cover.png" alt="Open Video"/>
</a>


## News
* 2025.06.12: Qwen2.5-Omni-7B ranked first among open source models in the spoken language understanding and reasoning benchmark [MMSU](https://arxiv.org/abs/2506.04779).
* 2025.06.09: Congratulations to our open source Qwen2.5-Omni-7B for ranking first in the [MMAU](https://sakshi113.github.io/mmau_homepage/#leaderboard) leaderboard, and first in the [MMAR](https://github.com/ddlBoJack/MMAR) of open source models in the audio understanding and reasoning evaluation!
* 2025.05.16: We release 4-bit quantized Qwen2.5-Omni-7B (GPTQ-Int4/AWQ) models that maintain comparable performance to the original version on multimodal evaluations while reducing GPU VRAM consumption by over 50%+. See [GPTQ-Int4 and AWQ Usage](#gptq-int4-and-awq-usage) for details, and models can be obtained from Hugging Face ([GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-AWQ)) and ModelScope ([GPTQ-Int4](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4)|[AWQ](https://modelscope.cn/models/Qwen/Qwen2.5-Omni-7B-AWQ))
* 2025.05.13: [MNN Chat App](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#releases) support Qwen2.5-Omni now, let's experience Qwen2.5-Omni on the edge devices! Please refer to [Deployment with MNN](#deployment-with-mnn) for information about memory consumption and inference speed benchmarks.
* 2025.04.30: Exciting! We We have released Qwen2.5-Omni-3B to enable more platforms to run Qwen2.5-Omni. The model can be downloaded from [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Omni-3B). The [performance](#performance) of this model is updated, and please refer to [Minimum GPU memory requirements](#minimum-gpu-memory-requirements) for information about resource consumption. And for best experience, [transformers](#--transformers-usage) and [vllm](#deployment-with-vllm) code have update, you can pull the [official docker](#-docker) again to get them.
* 2025.04.11: We release the new vllm version which support audio ouput now! Please experience it from source or our docker image.
* 2025.04.02: ⭐️⭐️⭐️ Qwen2.5-Omni reaches top-1 on Hugging Face Trending! 
* 2025.03.29: ⭐️⭐️⭐️ Qwen2.5-Omni reaches top-2 on Hugging Face Trending! 
* 2025.03.26: Real-time interaction with Qwen2.5-Omni is available on [Qwen Chat](https://chat.qwen.ai/). Let's start this amazing journey now!
* 2025.03.26: We have released the [Qwen2.5-Omni](https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e). For more details, please check our [blog](https://qwenlm.github.io/blog/qwen2.5-omni/)!


## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Key Features](#key-features)
  - [Model Architecture](#model-architecture)
  - [Performance](#performance)
- [Quickstart](#quickstart)
  - [Transformers Usage](#--transformers-usage)
  - [ModelScope Usage](#-modelscope-usage)
  - [GPTQ-Int4 and AWQ Usage](#gptq-int4-and-awq-usage)
  - [Usage Tips](#usage-tips)
  - [Cookbooks for More Usage Cases](#cookbooks-for-more-usage-cases)
  - [API inference](#api-inference)
  - [Customization Settings](#customization-settings)
- [Chat with Qwen2.5-Omni](#chat-with-qwen25-omni)
  - [Online Demo](#online-demo)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [Real-Time Interaction](#real-time-interaction)
- [Deployment with vLLM](#deployment-with-vllm)
- [Deployment with MNN](#deployment-with-mnn)
- [Docker](#-docker)
<!-- - [Citation](#citation) -->

## Overview 
### Introduction
Qwen2.5-Omni is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. 

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/qwen_omni.png" width="80%"/>
<p>

### Key Features

* **Omni and Novel Architecture**: We propose Thinker-Talker architecture, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. We propose a novel position embedding, named TMRoPE (Time-aligned Multimodal RoPE), to synchronize the timestamps of video inputs with audio.

* **Real-Time Voice and Video Chat**: Architecture designed for fully real-time interactions, supporting chunked input and immediate output.

* **Natural and Robust Speech Generation**: Surpassing many existing streaming and non-streaming alternatives, demonstrating superior robustness and naturalness in speech generation.

* **Strong Performance Across Modalities**: Exhibiting exceptional performance across all modalities when benchmarked against similarly sized single-modality models. Qwen2.5-Omni outperforms the similarly sized Qwen2-Audio in audio capabilities and achieves comparable performance to Qwen2.5-VL-7B.

* **Excellent End-to-End Speech Instruction Following**: Qwen2.5-Omni shows performance in end-to-end speech instruction following that rivals its effectiveness with text inputs, evidenced by benchmarks such as MMLU and GSM8K.

### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/overview.png" width="80%"/>
<p>

### Performance

We conducted a comprehensive evaluation of Qwen2.5-Omni, which demonstrates strong performance across all modalities when compared to similarly sized single-modality models and closed-source models like Qwen2.5-VL-7B, Qwen2-Audio, and Gemini-1.5-pro. In tasks requiring the integration of multiple modalities, such as OmniBench, Qwen2.5-Omni achieves state-of-the-art performance. Furthermore, in single-modality tasks, it excels in areas including speech recognition (Common Voice), translation (CoVoST2), audio understanding (MMAU), image reasoning (MMMU, MMStar), video understanding (MVBench), and speech generation (Seed-tts-eval and subjective naturalness).

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/bar.png"/>
<p>

[Full benchmark tables omitted for brevity - see source file for complete performance data]

## Quickstart

Below, we provide simple examples to show how to use Qwen2.5-Omni with 🤖 ModelScope and 🤗 Transformers.

The codes of Qwen2.5-Omni has been in the latest Hugging face transformers and we advise you to install with command:
```
pip install transformers==4.52.3
pip install accelerate
```
or you might encounter the following error:
```
KeyError: 'qwen2_5_omni'
```
and you can also use our [official docker image](#-docker) to start without building from source.

We offer a toolkit to help you handle various types of audio and visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved audio, images and videos. You can install it using the following command and make sure your system has `ffmpeg` installed:

```bash
# It's highly recommended to use `[decord]` feature for faster video loading.
pip install qwen-omni-utils[decord] -U
```

If you are not using Linux, you might not be able to install `decord` from PyPI. In that case, you can use `pip install qwen-omni-utils -U` which will fall back to using torchvision for video processing. However, you can still [install decord from source](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source) to get decord used when loading video.

We are preparing [cookbooks](https://github.com/QwenLM/Qwen2.5-Omni/tree/main/cookbooks) for many capabilities, including audio understanding, voice chatting, screen recording interaction, video information extracting, omni chatting and more. Welcome to learn more!

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX

@article{Qwen2.5-Omni,
  title={Qwen2.5-Omni Technical Report},
  author={Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin},
  journal={arXiv preprint arXiv:2503.20215},
  year={2025}
}
```

<br>