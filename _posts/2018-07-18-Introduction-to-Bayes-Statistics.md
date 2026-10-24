---
layout: post
date: 2018-07-18 12:00:00 -0500
image: /data/regressionfit.gif
---

## Introduction to Bayesian Statistics - Part 1

### Introduction

"Frequentist vs Bayesian" reasoning/inference has been an important debate in the field of statistics. In this article we will discuss introductory concepts of Bayesian statistics - only in terms of philosphy; the math will be discussed in a future post.

### Three focus areas of Bayesian statistics

1. Prior: As kids we were afraid of the darkness of night, even though did not observe that the nights were more threatening than the days. Why is that? It’s our instincts — we 'believe' this to be the truth without any evidence. This is equivalent to a prior. For a more concrete example, watch [this video](https://www.youtube.com/watch?v=1RA2Zy_IZfQ&t=915s) from 15:15 to 16:30.
2. Data: As we grow up, we observe many people with similar attributes (economic status, demographics, etc.). We follow the news. Implicitly we are collecting data.
3. Posterior: Our understanding of threats posed by the dark changes continuously based on the data we collect. The updated understanding of this threat is equivalent to a posterior.

### Why Bayesian?

As insecure beings in a practically infinite universe, we humans have been victims of our belief system. Through careful observation and (thought) experiments we managed to let go of many beliefs. Eg: We assumed that arrival of comets caused destruction of crops, spread of epidemics and/or fall of kings/kingdoms. Through careful observation by astronomers over centuries, we understood that comets, just like the Earth, are heavenly bodies that revolve around the Sun. We also understood that the appearance of comets were mere coincidences and that irrigation, superstition and war were the causes of disaster.

### A Trivial Example

When we look at a coin, we start with the trivial assumption that the coin is unbiased (P(H) = P(T) = 0.5). This is our prior. When we toss the coin, we start observing a sequence of H/T outcomes. This is our data. As we observe the results of tossing the coin multiple times, we update our assumption on the coin (P(Unbiased \| Data)). It is difficult for us to continue to believe that the coin is unbiased if we observe a streak of 10 consecutive heads (or 10 consecutive tails). However, this conclusion may be defeated if we observe 10 consecutive results of opposite type in the next 10 tosses. Getting a concrete estimate for P(H) looks like an endless process, as stated by the frequentist definition: P(H) = N(H)/N(Tosses), where N(Tosses) -> infinity.

### Conclusion

We don’t have infinite time to perform experiments (Darwin’s theory and cosmology can be considered as practical exceptions, but human beings did not perform these experiments). We are limited by the data we observe over the duration of the experiment(s). Hence, our reasoning using Bayesian statistics will lead to a personalized version of reality, which depends on the weightage given to the prior and the data.

In our darkness example, if we give extremely high weightage to the prior and extremely low or zero weightage to the data, we will never overcome our fear of the dark. We are tuned to learn a local (+ personalized) optimal solution from the data. In a way we are Bayesians.

### Inspiration

I'm grateful to Prof. Richard McElreath for releasing his course called "Statistical Rethinking" on YouTube. I strongly recommend it to people who have a slight mathematical inclination.