
# Introduction to Linear Algebra

Linear algebra is a type of math that helps us work with groups of numbers, like rows in a table or grids like spreadsheets. 

It’s how we describe and move data around — adding things together, stretching them, or rotating them. In data science and machine learning, linear algebra is important because it lets computers understand patterns in data, make predictions, and even learn on their own.

It’s the math behind images, recommendations, and smart tools — and it all starts with understanding simple things like vectors and matrices.

---

## 1 What Is a Vector?

A **vector** is just a list of numbers.  
Think of it as a single row of values, like someone’s age, height, weight, directions:

$$
\begin{bmatrix}
25 & 170 & 68
\end{bmatrix}
$$

It can have multiple dimensions, $\mathbb{R}^{n}$ <- which means REAL numbers to the n-th dimension

- A 1D vector: $\begin{bmatrix} 25 \end{bmatrix}$
- A 2D vector: $\begin{bmatrix} 25 \\ 35 \end{bmatrix}$

Standard vector is written as $\begin{bmatrix} x_{1} \\ x_{2} \\ \vdots \\ x_{m} \end{bmatrix}$ and as you can see it is COLUMN-wise.
 
Every item inside is refered to $x_{m}$. 

## 2 Vector Operations

With the vectors definition set, we can now move on to operations on vectors.

$$\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} + \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1 + y_1 \\ y_1 + y_2 \end{bmatrix}$$

Let's say two vectors $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$, lets do addition and subtraction on them and see what happens.

$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 1 + 3 \\ 2 + 4  \end{bmatrix} = \begin{bmatrix} 4 \\ 6  \end{bmatrix}$

The above is the same with subractions. However, when it comes to multiplying two vectors, it's a little different. There are two different methods of multiplying, the dot-product and the Hadamard Product.
The dot product gives you a single number and it measures **how aligned** the two vectors are. 

### Dot-Product

$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^n a_i b_i$$

The above can look like jibberish to people that hasn't had previous knowledge on math notations but we're here to learn. Let's break it down to understand what it is trying to say in plain english.

"Let there be two vectors, **vector a** and **vector b**, when doing a dot-product multiplication on these two vectors, you are to multiple vector a's first element with the vector b's first element, do this for every element, and then add them together"

The last notation that looks like an E is just a shorthand symbol for 'summation', sum all the $a_ib_i$ for every i in the vectors. Let's do an example:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix} = (1\cdot3) + (2\cdot4) = 3 + 8 = 12$

This is the typical dot-product being done. There is also a geometric interpretation of the dot product:

$$\vec{a}\cdot\vec{b} = ||\vec{a}||\cdot||\vec{b}|| cos(\theta)$$

This equations says that the dot product between vector a and vector b is equal to the product of the length of vector a, b and the cosine theta. 

$||a||$ the notation $|| ||$ is called the **norm**. The norm of a vector is the length of the vector. 
 For vector $\vec{a} = \begin{bmatrix} a_1 \\ a_2 \\ \dots \\ a_n \end{bmatrix}$, the norm is:

$$
||\vec{a}|| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}
$$

This is just the Pythagorean theorem in higher dimensions!

If we swap around the variable to solve for theta (the angle between the two vectors) we get:

$$cos(\theta) = \dfrac{\vec{a}\cdot\vec{b}}{||\vec{a}||\cdot||\vec{b}|| }$$

Let's do an example:

$\begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\begin{bmatrix} 3 \\ 4 \end{bmatrix}$

$$\dfrac{a\cdot b}{||a|| \cdot ||b||}$$ 

$$= \dfrac{(1 \cdot 3) + (2 \cdot 4)}{(\sqrt{1^2 + 2^2}) \cdot (\sqrt{3^2 + 4^2})}$$

$$= \dfrac{3 + 8}{(\sqrt{1 + 4}) \cdot (\sqrt{9 + 16})}$$

$$= \dfrac{11}{(\sqrt{5}) \cdot (\sqrt{25})} \approx{cos(0.9839)} \approx{{10.3}^\circ}$$

A small angle like 10.3° means the vectors are pointing in nearly the same direction — they're fundamentally similar. This kind of geometric reasoning is powerful in fields like NLP and high-dimensional machine learning, where surface differences (like length or magnitude) might hide deeper alignment.

## 4. What Is a Matrix?

A **matrix** is a grid of numbers, like a table:

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

That’s a 2×3 matrix (2 rows, 3 columns). You can think of it as a collection of vectors stacked either as rows or columns.

In data science, a matrix often holds your entire dataset:
- Rows = data points  
- Columns = features

---

## 5. Matrix Operations

### Addition / Subtraction

Add or subtract matrices of the same size by doing element-by-element operations:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
+
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
=
\begin{bmatrix}
6 & 8 \\
10 & 12
\end{bmatrix}
$$

### Scalar Multiplication

Multiply every element by a scalar:

$$
3 \cdot
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
=
\begin{bmatrix}
3 & 6 \\
9 & 12
\end{bmatrix}
$$

### Matrix-Vector Multiplication

Multiply a matrix and a vector (when dimensions match):

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\cdot
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
\begin{bmatrix}
1x_1 + 2x_2 \\
3x_1 + 4x_2
\end{bmatrix}
$$

This happens a lot in machine learning: weights × input = prediction.

### Matrix-Matrix Multiplication

Multiply two matrices if the *inner dimensions* match:

$$
(m \times n) \cdot (n \times p) = m \times p
$$

Example:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\cdot
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
=
\begin{bmatrix}
1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\
3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

---

## 6. Identity and Zero Matrices

### Identity Matrix

Think of it as the “do nothing” matrix. Multiplying by it doesn’t change the original:

$$
A \cdot I = A
$$

Example of a 2×2 identity matrix:

$$
I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### Zero Matrix

All elements are zero. Acts like adding zero in regular math:

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

---

## 7. Transpose

Transposing a matrix flips it over its diagonal. Notation:

$$
A^T
$$

Example:

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
\quad \Rightarrow \quad
A^T =
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
$$

---

## 8. Linear Independence and Span

- A set of vectors is **linearly independent** if none of them can be written as a combination of the others.  
- The **span** of a set is all the vectors you can make using their linear combinations.

Example: vectors

$$
\vec{v}_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad
\vec{v}_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}
$$

span the entire 2D space $\mathbb{R}^2$.

---

## 9. Matrix Rank

The **rank** of a matrix is the number of linearly independent rows or columns.

High rank = lots of unique information.  
Low rank = some rows or columns are combinations of others (redundant info).

---

## 10. Systems of Linear Equations

You can solve systems like:

$$
Ax = b
$$

Where:  
- $A$ is a matrix of coefficients  
- $x$ is a vector of unknowns  
- $b$ is the result/output vector

Example:

$$
\begin{bmatrix}
2 & 1 \\
1 & 3
\end{bmatrix}
\cdot
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
5 \\
6
\end{bmatrix}
$$


## 11. Eigenvalues and Eigenvectors (Intro)

These are special vectors that don’t change direction when a matrix is applied:

$$
A \cdot v = \lambda v
$$

Where:

- $v$ is the eigenvector
- $\lambda$ is the eigenvalue

These show up in **PCA**, **stability**, and **graph theory**.

---

## 12. Real World Applications in ML

- **Dot product** → similarity (cosine similarity)
- **Matrix multiplication** → neural networks
- **Norms** → regularization (L1, L2)
- **Eigenvectors** → PCA (dimensionality reduction)

---