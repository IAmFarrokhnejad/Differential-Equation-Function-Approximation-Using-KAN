# Differential-Equation-Function-Approximation-Using-KAN
 
## Todo:
- Get this reference: https://asmedigitalcollection.asme.org/computingengineering/article-abstract/20/6/061004/1082608/Solution-of-Biharmonic-Equation-in-Complicated
- ~~Algorithm optimization for V 0.1~~
- ~~Fix the bugs in V 1.1.1.~~
- Fix the issue with the Jupyter Code V 1.1 (Only for the first example).
- ~~Add the references that cited the initial work.~~
- ~~Convert all base codes to Jupyter.~~
- ~~Explore all useful subtypes of KAN (references 7, 8, and 9) to figure out whether they can prove useful for our examples or not.~~
- Get the following paper and add the 3rd example from it to the work. https://link.springer.com/article/10.1007/s00500-022-07529-3


### Important References:
- SUseful subtypes of KAN: 7-9
- KAN for ODE: 18 and 23
- Related works with the exact same problems to solve: 1-3


## Important Update on subtypes of KAN: 
3 architectures discussed in the provided documents[7-9] — **Chebyshev KAN**, **rKAN (Rational KAN)**, and **SigKAN** — each have distinct strengths and suitability depending on the problem type. Comparison below:

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
- V 1.1 of the Jupyter codes works perfectly for example 2.