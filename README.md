# Foundation Models for Manipulation: Step-by-Step Guide  

<div>
    <img src="./images/foundation_models.png" alt="Global Trajectory">
</div><br>

Robotics is undergoing a revolution as foundation models (large AI models trained on diverse datasets) are enabling robots to perform complex manipulation tasks with unprecedented generalization. Unlike traditional approaches that require task-specific programming, these models leverage transformers and imitation learning to allow robots to adapt flexibly to new tasks and environments. This shift is driving advances in critical applications, from industrial automation and precision manufacturing to home robot assistants.  

In this blog post, we’ll explore state-of-the-art methods like ACT, Helix from Figure AI, OpenVLA, Octo, RT-2 from Google, and Diffusion Policies from Toyota Research Institute. We'll break down each method into its core components: the datasets they use, how they process inputs and outputs, their model architectures, training procedures (including loss functions and fine-tuning strategies), and the metrics and benchmarks used to evaluate performance.  

Whether you're an AI researcher, a roboticist, or simply curious about the future of autonomous robots, this guide will provide a clear and structured overview of how modern foundation models are shaping the next generation of robotic manipulation!

## 1) Action Chunk Transformer (ACT)

<div>
    <img src="./images/act.PNG" alt="Global Trajectory">
</div><br>

ACT leverages **transformer-based action chunking** and end-to-end imitation learning to enable low-cost robotic arms to perform complex tasks with high success rates. Developed as part of the ALOHA project, ACT learns from real-world demonstrations collected via a custom teleoperation setup. The model generates action sequences in a chunked manner, improving stability and reducing compounding errors over time. With only 10 minutes of demonstrations, ACT enables robots to achieve 80-90% success rates on fine manipulation tasks.

## Dataset  

ACT uses a dataset collected from real-world bimanual teleoperation experiments. The dataset consists of **human demonstrations**, meaning they gather their own data rather than relying on pre-existing datasets. The demonstration dataset consists of trajectories of **image observations, joint positions and executed actions**.  

## Input & Output  

### Input  
- **4 RGB images** (480×640 resolution) processed through ResNet18  
- **Joint positions** for the two robot arms (7+7=14 DOF)  

### Output  

- **Absolute joint positions** in chunks (e.g., next 100 timesteps)

## Model Architecture  

<div>
    <img src="./images/act1.PNG" alt="Global Trajectory">
</div><br>

<div>
    <img src="./images/act2.PNG" alt="Global Trajectory">
</div>

### Training Phase

#### Step 1: Sample Data  
From the demonstration dataset, we sample:  
- A sequence of **RGB images** from four 480×640 webcams  
- **Joint positions** of two 7-DOF robot arms (14-dimensional vector)  
- A target **action sequence** over the next $k$ time steps  

#### Step 2: Infer Latent Style Variable $z$  
The encoder is a **BERT-style transformer encoder** that receives:  
- A learned **[CLS]** token  
- The **current joint positions**, projected to the embedding dimension  
- The **target action sequence**, also linearly embedded  

These inputs form a $(k + 2) \times d_\text{embed}$ sequence. After passing through the transformer encoder, the **[CLS] token output** is used to predict the **mean and variance** of the latent style variable $z$, modeled as a diagonal Gaussian. Using the **reparameterization trick**, a sample of $z$ is drawn, enabling gradient backpropagation.

#### Step 3: Decode Predicted Action Sequence  
The decoder — the actual **policy** — takes as input:
- **Image features**: Each image is processed with a **ResNet18** to get a 15×20×512 feature map, flattened into a sequence of 300×512. For 4 cameras, this gives a total of 1200×512.
- **2D sinusoidal position embeddings** are added to preserve spatial structure.
- **Joint positions** and **$z$**, both projected to the same embedding dimension.

These inputs are concatenated into a **1202×512** sequence and passed through a **transformer encoder**. A **transformer decoder** uses **cross-attention** to generate a sequence of **$k \times 14$** outputs, representing joint positions for each time step.

---

### Inference Phase

At test time, the model uses only the **CVAE Decoder** as the policy. The encoder is discarded.

- The robot receives a new observation: **RGB images + joint positions**  
- These are processed exactly as during training (ResNet18 → flattened features → transformer encoder)  
- The **style variable $z$** is fixed to a **zero vector** (i.e., mean of the prior distribution)  
- The transformer decoder outputs a deterministic **$k \times 14$** tensor, corresponding to the next $k$ joint positions

This deterministic decoding provides **stable, repeatable behavior**, which is especially valuable for evaluation and deployment.

---

## 2) Octo: An Open-Source Generalist Robot Policy

<div>
    <img src="./images/octo.PNG" alt="Global Trajectory">
</div><br>

Octo is a large, transformer-based policy pretrained on 800k demonstrations from the Open X-Embodiment dataset. Designed for flexibility, it supports multiple robots, sensor setups, and task types — including language commands and goal images. Octo can be finetuned quickly on new environments and is fully open-source, making it a powerful foundation for scalable, general-purpose robotic learning.

## Dataset  

Octo is trained on a massive dataset of **800,000 robot trajectories** collected from the Open X-Embodiment dataset - the largest and most diverse robot manipulation dataset to date. This dataset brings together demonstrations from nine different robotic platforms, spanning a wide variety of manipulation tasks such as pick-and-place, tool use, button pressing and drawer opening or closing. The data is highly heterogeneous, featuring a mix of camera perspectives (e.g., wrist-mounted and third-person views), robots with different degrees of freedom, and task-conditioning signals in the form of either language instructions or goal images.

## Input & Output  

### **Input:**  

  - **RGB images** from multiple viewpoints (wrist cam, third-person).  
  - **Proprioceptive states** (joint positions, velocities).  
  - **Task conditioning**:  
    - **Text commands** (e.g., "Pick up the red cup").  
    - **Goal images** (e.g., "Make the scene look like this").  

### **Output:**  
- **Delta Cartesian position actions** in chunks.  

## 3) Open VLA Meta

## 4) VLAM Helix Figure

## 5) Diffusion Vs Bins Vs FAST ????

----> but tipically Diffusion is used with transformers or can be used alone? ASK GPT + CODE



A standard Denoising Diffusion Probabilistic Model (DDPM) works by starting from a noisy version of the data and iteratively “denoising” it. For a conditional version, we modify the denoising step so that the noise prediction is conditioned on the observation.

The modified denoising update equation is given by:

$$
A_{t-1} = \alpha \Bigl( A_t - \gamma\, \varepsilon_\theta\bigl(O_t, A_t, k\bigr) + \mathcal{N}(0, \sigma^2 I) \Bigr)
$$

where:

- $A_t$ is the current (noisy) action sequence at iteration $k$.
- $\varepsilon_\theta\bigl(O_t, A_t, k\bigr)$ is a neural network (the noise prediction network) that estimates the noise present in $A_t$ given the observation $O_t$ and the denoising iteration $k$.
- $\alpha$ and $\gamma$ are schedule parameters (think of them as similar to a learning rate schedule).
- $\mathcal{N}(0, \sigma^2 I)$ is Gaussian noise added at each step.

This update can be seen as a single step of gradient descent on an energy function, where $\varepsilon_\theta$ approximates the gradient:

$$
-\nabla_{A_t}\log p\bigl(A_t \mid O_t\bigr).
$$

---

## 3. Training Loss

The training loss is defined as the Mean Squared Error (MSE) between the true noise $\varepsilon_k$ (that we add during training) and the predicted noise:

$$
L = \text{MSE}\Bigl( \varepsilon_k,\; \varepsilon_\theta\Bigl(O_t,\; A_{0t} + \varepsilon_k,\; k\Bigr) \Bigr)
$$

### Explanation

- $A_{0t}$ is the original (clean) action sequence.
- We add a noise $\varepsilon_k$ (with a variance appropriate for the iteration $k$) to obtain a noisy version $A_{0t} + \varepsilon_k$.
- The network $\varepsilon_\theta$ is then asked to predict the noise that was added.
- Minimizing the MSE loss teaches the network to “undo” the noise, i.e., to recover the clean action sequence from the noisy one while being conditioned on the observation $O_t$.

```python
# Training loop
for epoch in range(epochs):
    for O, A0 in dataloader:  # Loop through dataset batches
        # Randomly choose an iteration k for each sample
        k = torch.randint(low=1, high=K, size=(O.shape[0],), dtype=torch.float32)

        # Sample noise with variance sigma
        epsilon_k = sigma * torch.randn_like(A0)

        # Create noisy actions
        A_noisy = A0 + epsilon_k

        # Predict noise using the network
        epsilon_pred = net(O, A_noisy, k)

        # Compute MSE loss
        loss = loss_fn(epsilon_pred, epsilon_k)

        # Backpropagation step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 4. Inference Mechanism

At inference time, you start with a sample $A_K$ from a Gaussian distribution (i.e., pure noise) and then iteratively apply the denoising update:

$$
A_{k-1} = \alpha \Bigl( A_k - \gamma\, \varepsilon_\theta\bigl(O_t, A_k, k\bigr) \Bigr)
$$

> *(Note: For inference, you might use a method like DDIM, which does not add extra noise in each step.)*

After $K$ iterations, you obtain $A_0$, which is the final predicted action sequence to be executed by the robot.

```python
def generate_action(O, num_steps=K):
    """
    Given an observation O, generate the corresponding action using diffusion denoising.
    """
    # Initialize action as pure noise
    A_k = torch.randn(1, action_dim)  # Start from Gaussian noise
    O = O.unsqueeze(0)  # Ensure observation has batch dimension

    # Iteratively denoise
    for k in range(num_steps, 0, -1):
        k_tensor = torch.tensor([[k]], dtype=torch.float32)  # Current diffusion step
        epsilon_pred = net(O, A_k, k_tensor)  # Predict noise
        A_k = alpha * (A_k - gamma * epsilon_pred)  # Update action

    return A_k.squeeze(0)  # Remove batch dimension for final output
```






Input: 
- ultima image
- Stato (variabile deve essere, uno ha 5dof, l altro vuole includere anche speeds, altro FTS, …)
- Ultima azione
- latent variable

Input latent net: (figure style)
- ultime K images
- Ultime K actions
- Ultimi K states
- Text command

Output: 
- azioni cartesiane continue con diffusion (delta o assolute?)

Backbone: llama2, siglip, dino



## Key Works and Citations

- **T. Zhao, V. Kumar, S. Levine, C. Finn**: [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/pdf/2304.13705)
- **D. Ghosh, H. Walke, K. Pertsch**: [*Octo: An Open-Source Generalist Robot Policy*](https://arxiv.org/pdf/2405.12213)
- **A. Brohan, N. Brown, J. Carbajal**: [*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*](https://arxiv.org/pdf/2307.15818)
- **M. Kim, K. Pertsch, S. Karamcheti**: [*OpenVLA: An Open-Source Vision-Language-Action Model*](https://arxiv.org/pdf/2406.09246)
- **K. Black, N. Brown, D. Driess**: [*π0: A Vision-Language-Action Flow Model for General Robot Control*](https://arxiv.org/pdf/2410.24164)
- **Figure AI**: [*Helix: A Vision-Language-Action Model for Generalist Humanoid Control*](https://www.figure.ai/news/helix)
- **C. Chi, Z. Xu, S. Feng, E. Cousineau**: [*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://arxiv.org/pdf/2303.04137)
- **K. Pertsch, K. Stachowicz, B. Ichter**: [*FAST: Efficient Action Tokenization for Vision-Language-Action Models*](https://arxiv.org/pdf/2501.09747)