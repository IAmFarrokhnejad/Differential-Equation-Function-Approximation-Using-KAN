# Differential-Equation-Function-Approximation-Using-KAN
 

## Useful material: 
1. Sample implementation of Chebyshev KAN (The first variation): https://github.com/SynodicMonth/ChebyKAN


## Todo:

- ~Draw the architecture and add it to the paper; modify it if needed.~ 
- ~Do all citations from reference 21 and onward.~
- ~Address all the issues that professor mentioned.~
- Add a cover page.

### Target Journal: https://link.springer.com/journal/500/submission-guidelines 
- Check for writing style of the papers from the journal above.
---

### 1. **Chebyshev KAN (7 - Chebyshev Polynomial-Based KAN)**
- **Strengths**: 
  - Chebyshev polynomials provide excellent approximation properties with rapid convergence and numerical stability. 
  - Particularly effective for high-dimensional problems or those requiring efficient parameter use.
  - Well-suited for smooth, nonlinear functions and approximating solutions of ODEs where the solution is expected to be stable and differentiable.
- **Suitability**:
  - Likely effective for ODE (1) due to its complexity in terms of variable coefficients and expected smooth solution.
  - Offers less flexibility for functions with sharp discontinuities or highly localized features, potentially making it less ideal for ODE (2), which might involve oscillatory behavior due to the cosine term.

---

### 2. **rKAN (8 - Rational KAN)**
- **Strengths**:
  - Uses rational functions, allowing it to handle functions with singularities, asymptotic behavior, or sharp transitions.
  - Suitable for problems requiring precision over wide value ranges.
- **Suitability**:
  - Strong candidate for ODE (2) due to the presence of oscillatory and possibly rapidly changing behavior introduced by the cosine term.
  - May also work well for ODE (1), but its rational basis may not exploit the smoothness and stability as effectively as Chebyshev KAN.

---

### 3. **SigKAN (9 - Signature-Weighted KAN)**
- **Strengths**:
  - Enhances KAN with path signature features, capturing geometric and temporal dependencies.
  - Well-suited for sequential or time-dependent data and can model complex relationships dynamically.
- **Suitability**:
  - Overkill for ODE problems where standard numerical properties (e.g., orthogonality, stability) suffice.
  - More appropriate for datasets with rich sequential patterns, making it less optimal for these ODEs unless time series-like behavior needs deeper modeling.

---

### Recommendations:
- For **Example 1**: **Chebyshev KAN** is preferred due to the need for stability and its efficient handling of smooth nonlinear functions with variable coefficients.
- For **Example 2**: **rKAN** is better suited as it handles oscillatory behaviors effectively and is designed to manage functions with localized or asymptotic features.

**SigKAN** might be suitable for either of the examples to leverage its advanced geometric feature capture.

### Accomplishments/Discoveries:
- V 1.3.1: All examples work for basic KAN, however, further optimization is required.
- V 2.0: Updated loss function, performance is comparable to that of reference [3].
- V 2.2: At this point there are only 2 methods left to further optimize the network: We must either define the architecture on our own(which may require a copious amount of time) to have more contrtol over the architecture, or start using other variations of KAN (KAN variations directory) subtype of KAN.(which is likely to involve the first method).
- V 2.3: Definitive basic KAN implementation.
