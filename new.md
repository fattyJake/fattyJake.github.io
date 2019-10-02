---
layout: post
comments: true
title: "When the Loss Won't Converge"
date: 2019-09-06 20:00:00
categories: [DeepLearning]
mathjax: true
---

> While deep learning models have been proved to be powerful to apply in industrial applications, they certainly cost us a lot more time to make them right compared to developing machine learning pipelines, especially when the data is a lot and complex, or model scale is large, we often encounter issue of underfitting or loss not converging. The article will talk a little bit my experience of how to solve such problem.


{: class="table-of-content"}
* TOC
{:toc}

## Diagnose the Issue

What is converging? Some folks define converging as training set loss value eventually being lower than ```10e-06```. I'll say there's no exact definition, but we do know after several epochs' training training loss does not significantly decrease, there gotta be something wrong. Here also it brought up another discussion of overfitting vs underfitting. Here we're talking about underfitting that the model cannot fit the dataset well, and the symptom would be both training loss and validation loss fluctuate a lot since the beginning of the training and nearly don't drop during training:

<div style="text-align: center"><img src="../images/underfitting.png" width="600px" /></div>

<center> <i>Fig. 1. An underfitting example.</i> </center>

Overfitting on the other hand is model fit to the training set too much to generalize such good performance to data outside training set. The symptom of overfitting is both training loss and validation loss decrease at begining but after a certain timestamp, validation loss stop decrease (even begin increase) while training loss continue to drop to eventually near 0:

<div style="text-align: center"><img src="../images/overfitting.png" width="600px" /></div>

<center> <i>Fig. 2. An overfitting example.</i> </center>



Hope this post helps explain stuffs!