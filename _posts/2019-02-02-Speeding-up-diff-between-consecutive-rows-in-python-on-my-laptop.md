## Speeding-up 'diff' between Consecutive Rows in Python on My Laptop

### Introduction

I'm currently pursuing independent research on large financial data. Data engineering is an integral part of the analysis as financial data is often not in the format required for analysis. Financial data may be provided in the form of transactions/updates or in the form of aggregates. While it is easy to go from transaction level to aggregate level, it is difficult to do the opposite.

My laptop configuration is decent: Ubuntu 18.04, 16 GB DDR4, Intel core i7–8750H (6 + 6 virtual core) @ 2.2 GHz and a GPU (not relevant here). But brute force is not a good idea — code run time was ~ 4 hours for processing one day's data, which is not acceptable.

### Data (samples):

**Without mentioning the type of data, here are few anonymized samples (dy's can be positive or negative):**

<figure>
  <img src="../../../data/example1.jpg">
  <figcaption>Example 1</figcaption>
</figure>

<figure>
  <img src="../../../data/example2.jpg">
  <figcaption>Example 2</figcaption>
</figure>

<figure>
  <img src="../../../data/example3.jpg">
  <figcaption>Example 3</figcaption>
</figure>

