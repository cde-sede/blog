# **Deriving a Basic 3D Projection Matrix from First Principles**

Before diving into modern graphics APIs and their complex matrix stacks, it's useful-and enlightening-to derive the most basic form of a **3D perspective projection matrix** yourself. At its core, this matrix transforms a 3D point in space so it appears correctly on a 2D screen, with farther objects looking smaller.

This article will guide you through the step-by-step logic of building such a matrix from scratch, without shortcuts or hand-waving.

---

## Depth perspective


### Step 1: What Is Perspective Projection?

In a perspective view, the farther an object is from the camera, the smaller it appears. This effect is fundamental to human vision and is mimicked in 3D graphics through **perspective projection**.

A simple mathematical way to simulate this is:

$$x_{\text{proj}} = \frac{x}{z}, \quad y_{\text{proj}} = \frac{y}{z}$$

This tells us that to project a 3D point $(x, y, z)$ onto a 2D plane, we scale its $x$ and $y$ coordinates based on how far it is (its $z$).

---

### Step 2: Homogeneous Coordinates and Matrix Form

In computer graphics, we use 4D **homogeneous coordinates** to make projections compatible with matrix multiplication. This means we represent the point $(x, y, z)$ as a 4D vector:

$$
\mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

We want to find a $4 \times 4$ matrix $P$ such that:

$$
P \cdot \mathbf{v} = \begin{bmatrix} x' \\ y' \\ z' \\ w \end{bmatrix}
$$

and after dividing by $w$ (called **perspective divide**), we get:

$$
\left( \frac{x'}{w}, \frac{y'}{w}, \frac{z'}{w} \right) = \left( \frac{x}{z}, \frac{y}{z}, 1 \right)
$$

So we want:

* $x' = x$
* $y' = y$
* $z' = z$
* $w = z$

Let's construct a matrix that accomplishes this.

---

### Step 3: Build the Projection Matrix

We start with the identity matrix and modify the last row to produce $w = z$:

$$
P = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
$$

Apply $P$ to $\mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$:

$$ P \cdot \mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ z \end{bmatrix} \Rightarrow \text{after divide by } w = z: \left( \frac{x}{z}, \frac{y}{z}, 1 \right) $$

This achieves the perspective projection: shrinking $x$ and $y$ as $z$ increases.

---

### Step 4: Interpretation

This basic matrix ignores real-world factors like:

* Field of view (FOV)
* Aspect ratio
* Near/far clipping planes

But it illustrates the key idea: **make $w = z$, and use perspective divide** to simulate depth.

Modern projection matrices generalize this idea to handle screen boundaries, clipping planes, and normalization into device coordinates. But the core idea remains the same.

---

### Conclusion

The basic 3D projection matrix is not magic. It's a straightforward application of how vision works: the farther something is, the smaller it looks. By using homogeneous coordinates and choosing $w = z$, we make this behavior matrix-friendly.

Understanding this simple form gives you a solid intuition for how real projection matrices work-and makes 3D math a lot less mysterious.

## **Adding Clipping and Normalization**

In the first part, we built a minimal 3D perspective projection matrix that maps a 3D point $(x, y, z)$ to 2D screen space using just $(x/z, y/z)$. While this gives a basic perspective effect, it lacks essential features:

* Depth normalization to the range $[-1, 1]$
* Clipping (rejecting geometry behind the camera or too far away)
* Aspect ratio and field-of-view (FOV) control

In this part, we derive a **generalized perspective projection matrix** that includes all these features.

---

### Step 1: Goals of a Full Projection Matrix

We want a matrix that:

1. **Applies perspective**: $x/z$, $y/z$
2. **Maps visible depth range** $[z_{\text{near}}, z_{\text{far}}]$ into normalized device coordinates (NDC) in $[-1, 1]$
3. **Preserves aspect ratio**
4. **Clips geometry** outside the view frustum

---

### Step 2: Define the Viewing Frustum

We define a frustum (truncated pyramid) where:

* Near plane is at $z = z_{\text{near}} > 0$
* Far plane is at $z = z_{\text{far}} > z_{\text{near}}$
* The camera looks down the negative $z$-axis (OpenGL convention)

Assume a vertical field of view angle $\theta$, and screen aspect ratio $a = \frac{\text{width}}{\text{height}}$.

From the vertical FOV, the top of the near plane is at:

$$
t = z_{\text{near}} \cdot \tan\left(\frac{\theta}{2}\right)
$$

The bottom is $-t$, and the left/right edges are:

$$
r = t \cdot a, \quad l = -r
$$

---

### Step 3: Construct the Perspective Matrix

We aim to map the frustum into a **canonical cube** from $[-1, 1]$ in all three axes. Here's the standard OpenGL-style perspective matrix:

$$
P =
\begin{bmatrix}
\frac{z_{\text{near}}}{r} & 0 & 0 & 0 \\
0 & \frac{z_{\text{near}}}{t} & 0 & 0 \\
0 & 0 & -\frac{z_{\text{far}} + z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}} & -\frac{2 z_{\text{far}} z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

Let's explain each row:

#### Row 1 & 2: X and Y Scaling

$$
\frac{z_{\text{near}}}{r} \quad \text{and} \quad \frac{z_{\text{near}}}{t}
$$

These scale $x$ and $y$ based on FOV and aspect ratio, so that the near rectangle maps to $[-1, 1] \times [-1, 1]$ after division by $w = -z$.

#### Row 3: Z Mapping

We want to map $[z_{\text{near}}, z_{\text{far}}] \rightarrow [-1, 1]$. The form:

$$
z' = A z + B
$$

becomes:

$$
A = -\frac{z_{\text{far}} + z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}}, \quad
B = -\frac{2 z_{\text{far}} z_{\text{near}}}{z_{\text{far}} - z_{\text{near}}}
$$

So depth can be tested in NDC space without needing to know original $z$.

#### Row 4: Perspective Divide

The $-1$ in the fourth row ensures:

$$
w = -z
\Rightarrow
\left( \frac{x'}{-z}, \frac{y'}{-z}, \frac{z'}{-z} \right)
$$

This is the **perspective divide** that gives the depth-correct behavior.


## Using FOV

Let's break down **how the perspective projection matrix using field of view (FOV)** is derived.

### Step 1: From FOV to View Volume

Let's start with the vertical field of view $\theta$ and the near plane at $z = z_n$.

From the definition of FOV:

$$
\tan\left(\frac{\theta}{2}\right) = \frac{\text{top}}{z_n}
\Rightarrow
\text{top} = t = z_n \cdot \tan\left(\frac{\theta}{2}\right)
$$

Aspect ratio $a = \frac{\text{width}}{\text{height}}$ gives:

$$
r = t \cdot a
$$

So the view frustum at the near plane spans:

* $x \in [-r, r]$
* $y \in [-t, t]$

We now need to scale and normalize this into $[-1, 1]$ in $x$, $y$, and $z$.

---

### Step 2: Scaling X and Y

To map $[-r, r]$ to $[-1, 1]$, we scale $x$ by $\frac{1}{r}$. Same for $y$ with $\frac{1}{t}$.

So the top-left 2×2 part of the matrix becomes:

$$
\begin{bmatrix}
\frac{1}{r} & 0 \\
0 & \frac{1}{t}
\end{bmatrix}
=
\begin{bmatrix}
\frac{1}{a \cdot \tan(\theta/2)} & 0 \\
0 & \frac{1}{\tan(\theta/2)}
\end{bmatrix}
$$

This ensures points at the edge of the frustum map to $-1$ or $1$ after division by $w = -z$.

---

### Step 3: Perspective and Depth Mapping

We want $w = -z$ to enable perspective divide, and $z_{\text{view}} \in [z_n, z_f]$ to map to $z_{\text{ndc}} \in [-1, 1]$.

We need:

$$
z' = A z + B, \quad w = -z
\Rightarrow z_{\text{ndc}} = \frac{z'}{w} = \frac{A z + B}{-z}
$$

Set conditions:

* When $z = z_n$: $\frac{A z_n + B}{-z_n} = -1$
* When $z = z_f$: $\frac{A z_f + B}{-z_f} = 1$

Solving these:

$$
\text{(1)} \quad \frac{A z_n + B}{-z_n} = -1 \Rightarrow A z_n + B = z_n \\
\text{(2)} \quad \frac{A z_f + B}{-z_f} = 1 \Rightarrow A z_f + B = -z_f
$$

Subtracting (1) from (2):

$$
A(z_f - z_n) = -z_f - z_n \Rightarrow A = -\frac{z_f + z_n}{z_f - z_n}
$$

Plug into (1) to find $B$:

$$
B = z_n - A z_n = z_n + z_n \cdot \frac{z_f + z_n}{z_f - z_n}
= \frac{2 z_f z_n}{z_n - z_f}
$$

So:

$$
z' = A z + B = \frac{z_f + z_n}{z_n - z_f} z + \frac{2 z_f z_n}{z_n - z_f}
$$

---

### Step 4: Final Matrix

$$
P =
\begin{bmatrix}
\frac{1}{a \tan(\theta/2)} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(\theta/2)} & 0 & 0 \\
0 & 0 & \frac{z_f + z_n}{z_n - z_f} & \frac{2 z_f z_n}{z_n - z_f} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$

Multiplied with $\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$, it outputs a 4D vector $(x', y', z', w = -z)$, which after division gives normalized screen coordinates.
