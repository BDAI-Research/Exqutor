<hr/>

# Exqutor TPC-H Query Plans

## Exqutor with Vector Indexes (ECQO)
An optimization technique for VAQs with vector indexes, integrating ECQO into query planning by applying ANN searches only to vector tables. These results are reused at execution time without incurring redundant overhead.
### TPC-H Q3 Query Plan
<div align="center">
  <img src="figures/q3_ecqo.jpg" alt="Q3 plan with ECQO" width="80%"  style="border:2px solid black;">
</div>

### TPC-H Q5 Query Plan
<div align="center">
  <img src="figures/q5_ecqo.jpg" alt="Q5 plan with ECQO" width="80%">
</div>

### TPC-H Q8 Query Plan
<div align="center">
  <img src="figures/q8_ecqo.jpg" alt="Q8 plan with ECQO" width="80%">
</div>

### TPC-H Q9 Query Plan
<div align="center">
    <img src="figures/q9_ecqo.jpg" alt="Q9 plan with ECQO" width="80%">
</div>

### TPC-H Q10 Query Plan
<div align="center">
    <img src="figures/q10_ecqo.jpg" alt="Q10 plan with ECQO" width="80%">
</div>

### TPC-H Q11 Query Plan
<div align="center">
    <img src="figures/q11_ecqo.jpg" alt="Q11 plan with ECQO" width="80%">
</div>

### TPC-H Q12 Query Plan
<div align="center">
    <img src="figures/q12_ecqo.jpg" alt="Q12 plan with ECQO" width="80%">
</div>

### TPC-H Q20 Query Plan
<div align="center">
    <img src="figures/q20_ecqo.jpg" alt="Q20 plan with ECQO" width="80%">
</div> 

## Exqutor without Vector Indexes (Sampling)
A sampling-based cardinality estimation approach for VAQs without vector indexes. Exqutor employs adaptive sampling to improve accuracy by dynamically adjusting the sample size based on Q-Error through a momentum-based adjustment mechanism and
a learning rate scheduler.

### TPC-H Q3 Query Plan
<div align="center">
    <img src="figures/q3_sampling.jpg" alt="Q3 plan with Sampling" width="80%">
</div>

### TPC-H Q8 Query Plan
<div align="center">
    <img src="figures/q8_sampling.jpg" alt="Q8 plan with Sampling" width="80%">
</div>  

### TPC-H Q10 Query Plan
<div align="center">
    <img src="figures/q10_sampling.jpg" alt="Q10 plan with Sampling" width="80%">
</div> 

### TPC-H Q12 Query Plan
<div align="center">
    <img src="figures/q12_sampling.jpg" alt="Q12 plan with Sampling" width="80%">
</div>

<br/>
<hr/>

# Exqutor TPC-DS Query Plans

### TPC-DS Q7 Query Plan
<div align="center">
    <img src="figures/ds_q7.jpg" alt="TPC-DS Q7 plan" width="80%">
</div>

### TPC-DS Q12 Query Plan
<div align="center">
    <img src="figures/ds_q12.jpg" alt="TPC-DS Q12 plan" width="80%">
</div>

### TPC-DS Q19 Query Plan
<div align="center">
    <img src="figures/ds_q19.jpg" alt="TPC-DS Q19 plan with Sampling" width="80%">
</div>