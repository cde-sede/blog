# **Deriving 3D Projection Matrices from First Principles**

Understanding how 3D graphics work behind the scenes can seem intimidating, but at its core, 3D projection is just math that mimics how our eyes see the world. This article will guide you through building a complete perspective projection matrix step by step, starting from the most basic concepts and working up to a full-featured implementation.

## **Part 1: The Basic Concept**

### What Is Perspective Projection?

In the real world, objects appear smaller the farther away they are. A simple way to simulate this mathematically is:

$$x_{\text{screen}} = \frac{x}{z}, \quad y_{\text{screen}} = \frac{y}{z}$$

This formula takes a 3D point $(x, y, z)$ and projects it onto a 2D screen by dividing the $x$ and $y$ coordinates by the depth $z$. The farther away something is (larger $z$), the smaller it appears on screen.

### Moving to Matrix Form

In computer graphics, we use **homogeneous coordinates** - representing 3D points as 4D vectors - to make both affine and linear transformations work seamlessly with matrix multiplication. A 3D point $(x, y, z)$ becomes:

$\mathbf{v} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$

We want to find a $4 \times 4$ matrix $P$ that transforms this vector so that after a **perspective divide** (dividing by the fourth component), we get our desired projection.

### The Simplest Projection Matrix

To achieve $x/z$ and $y/z$, we need the fourth component $w$ to equal $z$. Here's the simplest matrix that does this:

$P = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}$

When we multiply this with our point vector:

$P \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ z \end{bmatrix}$

After dividing by $w = z$, we get $\left(\frac{x}{z}, \frac{y}{z}, 1\right)$ - exactly the perspective projection we wanted!

This basic matrix demonstrates the core principle, but it's missing several crucial features needed for real 3D graphics.

---

## **Part 2: Adding Essential Features**

### What's Missing?

Our basic matrix works, but real 3D applications need:

1. **Field of view control** - determining how "wide" the view is
2. **Aspect ratio handling** - ensuring circles stay circular on rectangular screens  
3. **Depth clipping** - rejecting objects too close or too far from the camera
4. **Depth normalization** - mapping depth values to a standard range for depth testing

### Defining the View Frustum

A **view frustum** is the 3D region that's visible to the camera - shaped like a truncated rectangular pyramid. We define it with:

- **Near plane** at distance $z_n$ (closest visible distance)
- **Far plane** at distance $z_f$ (farthest visible distance)  
- **Vertical field of view** angle $\theta$
- **Aspect ratio** $a = \frac{\text{width}}{\text{height}}$

From these parameters, we can calculate the frustum dimensions at the near plane:

$$\text{top} = t = z_n \cdot \tan\left(\frac{\theta}{2}\right)$$
$$\text{right} = r = t \cdot a$$

The visible region at the near plane spans $x \in [-r, r]$ and $y \in [-t, t]$.

---

## **Part 3: Building the Complete Matrix**

### Scaling for Field of View and Aspect Ratio

To map the visible region $[-r, r] \times [-t, t]$ to the normalized range $[-1, 1] \times [-1, 1]$, we need to scale by $\frac{1}{r}$ and $\frac{1}{t}$:

$$\frac{1}{r} = \frac{1}{t \cdot a} = \frac{1}{a \cdot \tan(\theta/2)}$$
$$\frac{1}{t} = \frac{1}{\tan(\theta/2)}$$

### Handling Depth

For depth, we need two things:
1. **Perspective divide**: Make $w = -z$ (negative because we're looking down the negative z-axis)
2. **Depth mapping**: Map the visible depth range $[z_n, z_f]$ to $[-1, 1]$ for depth testing

The depth mapping takes the form $z' = Az + B$, and after perspective divide becomes:

$$z_{\text{ndc}} = \frac{z'}{w} = \frac{Az + B}{-z}$$

We want:
- When $z = z_n$: $z_{\text{ndc}} = -1$ 
- When $z = z_f$: $z_{\text{ndc}} = 1$

Solving these conditions:

$$\frac{Az_n + B}{-z_n} = -1 \Rightarrow Az_n + B = z_n$$
$$\frac{Az_f + B}{-z_f} = 1 \Rightarrow Az_f + B = -z_f$$

Subtracting the first equation from the second:

$$A(z_f - z_n) = -z_f - z_n \Rightarrow A = -\frac{z_f + z_n}{z_f - z_n}$$

Substituting back:

$$B = z_n - Az_n = \frac{2z_f z_n}{z_n - z_f}$$

### The Complete Perspective Matrix

Putting it all together:

$P = \begin{bmatrix}
\frac{1}{a \tan(\theta/2)} & 0 & 0 & 0 \\
0 & \frac{1}{\tan(\theta/2)} & 0 & 0 \\
0 & 0 & \frac{z_f + z_n}{z_n - z_f} & \frac{2z_f z_n}{z_n - z_f} \\
0 & 0 & -1 & 0
\end{bmatrix}$

Let's understand each component:

- **Row 1**: Scales $x$ based on aspect ratio and horizontal field of view
- **Row 2**: Scales $y$ based on vertical field of view  
- **Row 3**: Maps depth from $[z_n, z_f]$ to $[-1, 1]$ after perspective divide
- **Row 4**: Sets $w = -z$ to enable perspective divide

---

## **How It All Works Together**

When you multiply this matrix with a 3D point $\begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$:

1. You get a 4D result $\begin{bmatrix} x' \\ y' \\ z' \\ w \end{bmatrix}$ where $w = -z$
2. The graphics pipeline performs perspective divide: $\left(\frac{x'}{w}, \frac{y'}{w}, \frac{z'}{w}\right)$
3. This gives you normalized device coordinates where:
   - $x$ and $y$ are in $[-1, 1]$ and represent screen position
   - $z$ is in $[-1, 1]$ and represents depth for depth testing

Points outside this normalized cube are automatically clipped by the graphics hardware.

---

## **Conclusion**

The perspective projection matrix isn't magic - it's a systematic solution to the problem of converting 3D world coordinates into 2D screen coordinates while preserving depth information. By understanding how each component works:

- The basic $x/z, y/z$ division creates the perspective effect
- Homogeneous coordinates and matrix multiplication make it efficient
- Field of view and aspect ratio scaling ensure proper proportions
- Depth mapping enables hardware depth testing and clipping

This foundation will help you understand more advanced graphics concepts and debug projection-related issues in your 3D applications. Modern graphics APIs provide functions to generate these matrices, but knowing the underlying math gives you the power to customize and optimize when needed.
