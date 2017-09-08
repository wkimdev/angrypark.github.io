---
title: "[GAN] DiscoGAN 논문 이해하기"
layout: post
date: 2017-08-03 20:30
image: /assets/images/2017-08-03-DCGAN-paper-reading/background.jpg
headerImage: true
tag:
- gan
- pytorch
- deep learning
- ybigta
- disco-gan
category: blog
author: angrypark
description: Disco-GAN 논문 차근차근 이해하기
# jemoji: '<img class="emoji" title=":ramen:" alt=":ramen:" src="https://assets.github.com/images/icons/emoji/unicode/1f35c.png" height="20" width="20" align="absmiddle">'
---

<span style="color:#7C7877; font-family: 'Apple SD Gothic Neo'; font-weight:200">

이번 시간에는 GAN의 응용 논문 중 우리나라 SK T-Brain에서 발표되서 인정받고 있는 Learning to Discover Cross-Domain Relations
with Generative Adversarial Networks(이하 Disco-GAN)을 소개하겠습니다. 연세대학교 빅데이터 학회 YBIGTA GAN팀에서는 다음과 같은 순서를 GAN을 공부하고 있으며, 클릭하면 내용을 확인하실 수 있습니다.

**Study 순서**
- [GAN 논문 이해하기]()
- [GAN Code 분석하기]()
- [Conditional-GAN 논문 이해하기]()
- [Conditional-GAN Code 분석하기]()
- [GAN으로 내 핸드폰 손글씨 만들기]()
- [DCGAN 논문 이해하기]()
- [DCGAN Code 분석하기]()
- [DCGAN pytorch로 구현하기]()
- [InfoGAN 논문 이해하기]()
- [InfoGAN Code 분석하기]()
- [DiscoGAN 논문 이해하기]()

---


**목차**

- [0. Abstract](#0-abstract)
- [1. Introduction](#1-introduction)
- [2. Model](#2-model)
  - [2.1. Formulation](#21-formulation)
  - [2.2. Notation and Architecture](#22-notation-and-architecture)
  - [2.3. GAN with a Reconstruction Loss](#23-gan-with-a-reconstruction-loss)
  - [2.4. Our Proposed Model : Discovery GAN](#24-our-proposed-model-discovery-gan)
- [Reference](#reference)

---
> 논문 pdf :
[Learning to Discover Cross-Domain Relations
with Generative Adversarial Networks, 2016](https://arxiv.org/pdf/1703.05192.pdf)
---
## GAN Review
![gan-workflow](/assets/images/2017-08-03-DCGAN-paper-reading/gan-workflow.jpg)

> [[GAN] First GAN](https://angrypark.github.io/First-GAN/)
[[GAN] 1D Gaussian Distribution Generation](https://angrypark.github.io/GAN-tutorial-1/)

---
## 0. Abstract
지금까지는 GAN을 통해 사람이 구별하기 어려운 가짜 이미지를 만들어내는 데 목표를 두었고 어느정도 성공하였습니다. 그러나 우리가 원하는 목적의 가짜 이미지를 만들기 위해서는 그 이상의 목표가 해결되어야 합니다. 예를 들면, 소 그림을 넣으면 비슷한 느낌의 말을 만들어준다던가, 여러분의 사진을 넣으면 같은 사진인데 성별만 바뀌는 이미지를 만들어준다던가 말이죠. 이 논문에서는 이를 어느정도 해결하였는데요, 이번 글에서는 문제를 어떻게 정의내리고, 어떤 구조와 컨셉을 통해 이를 해결하였는지 살펴보도록 하겠습니다.

## 1. Introduction
사람들은 두개의 다른 도메인이 주어졌을 때 그 관계를 쉽게 찾아냅니다. 예를 들어, 영어로 구성된 문장을 프랑스어로 번역하여 주어진다면, 그 두 문장의 관계를 사람들은 쉽게 찾아낼 수 있습니다(의미는 같다, 언어는 다르다). 또한, 우리는 우리가 입고 있는 정장과 비슷한 스타일을 가지고 있는 바지와 신발을 쉽게 찾아낼 수 있습니다. 같은 스타일을 가지되 도메인만 정장에서 바지나 신발로 옮겨주는 것이죠.

그럼, 과연 기계도 서로 다른 두 개의 도메인 이미지의 관계를 찾아낼 수 있을까요? 이 문제는 “한 이미지를 다른 조건이 달려 있는 이미지로 재생성할 수 있을까?”라는 문제로 재정의됩니다. 다시 말하면, 같은 이미지인데 한 도메인에서 다른 도메인으로 mapping해주는 함수를 찾을 수 있는가의 문제인 것이죠. 사실 이 문제는 최근 엄청난 관심을 받고 있는 GAN에서 이미 어느 정도 해결되었습니다. 그러나 GAN의 한계는 사람이나 다른 알고리즘이 직접 명시적으로 짝지은 데이터를 통해서만 문제를 해결할 수 있다는 것입니다. (예, 국방무늬를 갖고 있는 옷을 바꾸면 국방무늬를 가진 신발이 되!)

명시적으로 라벨링된 데이터는 쉽게 구해지지 않으며, 많은 노동력과 시간을 필요로 합니다. 더군다나, 짝 중 하나의 도메인에서라도 그 사진이 없는 경우 문제가 생기고, 쉽게 짝짓기 힘들 정도로 훌륭한 선택지가 다수 발생하기도 하죠. 따라서, 이 논문에서는 우리는 2대의 다른 도메인에서 그 어떠한 explicitly pairing 없이 관계를 발견하는 것을 목표로 합니다(관계를 '발견'한다고 해서 DiscoGAN입니다.)

이 문제를 해결하기 위해서, 저자는 Discover cross-domain relations with GANs를 새롭게 제안하였습니다. 이전에 다른 모델들과 달리, 우리는 그 어떤 라벨도 없는 두 개의 도메인 데이터 셋을 pre-training없이 train합니다.(이하 A,B 도메인이라 명시할께요). Generator는 A도메인의 한 이미지를 input으로 해서 B도메인으로 바꿔줍니다. DiscoGAN의 핵심은 두개의 서로 다른 GAN이 짝지어져 있다는 것인데, 각각은 A를 B로, B를 A로 바꿔주는 역할을 해줍니다. 이 때의 핵심적인 전제는 하나의 도메인에 있는 모든 이미지를 다른 도메인의 이미지로 표현할 수 있다는 것입니다.

 결론부터 이야기 하자면 DiscoGAN은 Toy domain 과 real world image dataset에서 다 cross-domain relations를 알아내는 데 적합하다는 것을 확인할 수 있었습니다. 단순한 2차원 도메인에서 얼굴이미지 도메인으로 갔을 때에도 DiscoGAN 모델은 mode collapse problem에 좀더 robust하다는 것도 확인할 수 있었죠. 또한 얼굴, 차, 의자, 모서리, 사진 사이의 쌍방향 mapping에도 좋은 이미지 전환 성능을 보여주었습니다. 전환된 이미지는 머리 색, 성별, orientation 같은 특정한 부분만 바뀔수도 있었습니다.

## 2. Model
우리는 DiscoGAN이 어떤 문제들을 해결할 수 있는지 알아보았습니다. 이제 이 모델이 어떻게 이 문제를 해결하는 지 좀 더 자세히 분석해보죠.

### 2.1. Formulation
관계라는 것은 $$G_{AB}$$로 정의내려질 수 있습니다. 즉 $$G_{AB}$$ 도메인 $$A$$에 있는 성분들을 $$B$$로 바꿔주는 것이죠. 완전 비지도 학습에서는, $$G_{AB}$$와 $$G_{BA}$$는 모두 처음에 전혀 정의내려질 수 없습니다. 따라서, 일단 모든 관계는 1대1 대응으로 만들어주고 시작합니다. 그러면 자연스럽게, $$G_{AB}$$는 $$G_{BA}$$의 역관계가 됩니다.

함수 Gab의 범위는 도메인 A에 있는 모든 xa가 도메인 B에 있는 Gab(xa)로 연결된 것입니다.

자 이를 목적함수로 표현해봅시다. 이상적으로는, 보시는 것처럼 GBA GAB(xA) = xa이면 됩니다. 하지만 이런 제한식은 너무 엄격해서 이를 만족시키기 어렵습니다(사실 불가능하죠. generate해서 원래 사진 그대로 나온다는게 ㅎㅎ).  따라서 여기서는 d(GBA * GAB(xA), xA)를 최소화하려고 합니다. 비슷하게, d(GAB * GBA(xB), xB)도 최소화해야합니다. 이를 Discriminator와 generative adversarial loss가 들어간 loss로 표현하면 다음과 같습니다.

### 2.2. Notation and Architecture
구조를 살펴보죠. GAB는 Rdlrh, DB는 입니다.

### 2.3. GAN with a Reconstruction Loss
처음에는 기존 GAN을 약간 변형한 구조를 생각했었다고 합니다(그림 2-a). 기존 GAN은 input이 gaussian noise였던 것 기억하시나요? 여기서는 일단 input을 도메인 A의 image로 해줍니다. 이를 기반으로 generator가 fake image를 만들어내고, 이를 Discriminator는 도메인 B의 이미지와 함께 넣어서 무엇이 진짜인지를 구분하게 합니다. 즉, Generator의 입장에서는 비록 input은 도메인 A였지만, Discriminator를 속이기 위해서는 도메인 B와 유사한 이미지를 만들어야 한다는 것이죠. 이렇게만 잘 학습이된다면, Generator는 앞서 GAB의 역할을 충실히 할 수 있게 됩니다. 도메인 A를, 도메인 B로 바꿔주는 역할을 해주는 것이죠.
 하지만 이는 A에서 B로 가는 mapping만 배우게 됩니다. 동시에 B에서 다시 A로 가는 mapping도 학습하기 위해서 그림 2-b에서처럼 두번째 generator를 추가하게 됩니다. 또한 reconstruction loss도 추가하는데요, 이러한 과정들을 통해, 각각의 generator는 input 도메인에서 output 도메인으로 mapping하는 것은 물론 그 관계까지 discover하게 됩니다. 이 때 각각의 loss function은 다음과 같습니다.

### 2.4. Our Proposed Model : Discovery GAN
 최종적으로 이 논문에서 구현한 모델은 앞서 언급했던 그림 2-b의 모델 2개를 서로 다른 방향으로 이어주는 것입니다(그림 2-c). 각각의 모델은 하나의 도메인에서 다른 도메인으로 학습하며, 각각은 reconstruction을 통해 그 관계도 학습하게 됩니다.(ABA의 BA와 BAB의 BA는 다릅니다.) 이 때 GAB의 두 개의 Generator와 GBA의 두 개의 Generator는 서로 파라미터를 공유합니다. 그리고 xBA와 xAB는 각각 LDA, LDB로 들어가게 됩니다. 이전 모델과 중요한 차이는 두 도메인의 input 이미지가 다 reconstruct되었으며 그에 따라 두 개의 reconstruction loss(LCONSTA, LCONSTB)가 생성된다는 것입니다.

이처럼 두 개의 모델을 짝지어줌으로서 전체 Generator의 loss는 다음과 같이 정의합니다.

비슷하게 전체 Discriminator의 loss는 다음과 같이 정의합니다.


여기까지 DiscoGAN의 성능, 발전 과정, 해결할 수 있는 문제들, 그리고 각각의 구조와 loss function을 알아보았습니다. 다음에는 코드로, 몇몇 실험에 대해 어떻게 문제를 정의 내리고 해결했는지 알아보겠습니다. 아디오스~

---
## Reference
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Alec Radford el ec, 2016](https://arxiv.org/abs/1511.06434)

[초짜 대학원생의 입장에서 이해하는 Deep Convolutional Generative Adversarial Network (DCGAN) (1)](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-1.html)

[초짜 대학원생의 입장에서 이해하는 Deep Convolutional Generative Adversarial Network (DCGAN) (2)](http://jaejunyoo.blogspot.com/2017/02/deep-convolutional-gan-dcgan-2.html)

[김태훈: 지적 대화를 위한 깊고 넓은 딥러닝 (Feat. TensorFlow) - PyCon APAC 2016](https://www.youtube.com/watch?v=soJ-wDOSCf4&t=890s)

[Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)

---
