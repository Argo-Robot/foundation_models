# Foundation Models for Manipulation: Step-by-Step Guide  

<div>
    <img src="./images/foundation_models.png" alt="Global Trajectory">
</div><br>

Robotics is undergoing a revolution as foundation models (large AI models trained on diverse datasets) are enabling robots to perform complex manipulation tasks with unprecedented generalization. Unlike traditional approaches that require task-specific programming, these models leverage transformers and imitation learning to allow robots to adapt flexibly to new tasks and environments. This shift is driving advances in critical applications, from industrial automation and precision manufacturing to home robot assistants.  

In this blog post, we’ll explore state-of-the-art methods like ACT, Helix from Figure AI, OpenVLA, Octo, RT-2 from Google, and Diffusion Policies from Toyota Research Institute. We'll break down each method into its core components: the datasets they use, how they process inputs and outputs, their model architectures, training procedures (including loss functions and fine-tuning strategies), and the metrics and benchmarks used to evaluate performance.  

Whether you're an AI researcher, a roboticist, or simply curious about the future of autonomous robots, this guide will provide a clear and structured overview of how modern foundation models are shaping the next generation of robotic manipulation!

## 1) Action Chunk Transformer (ACT)

<div>
    <img src="./images/act.png" alt="Global Trajectory">
</div><br>

ACT leverages **transformer-based action chunking** and **end-to-end imitation learning** to enable low-cost robotic arms to perform complex tasks with high success rates. Developed as part of the ALOHA project, ACT learns from real-world demonstrations collected via a **custom teleoperation setup**. The model generates action sequences in a **chunked** manner, improving stability and reducing compounding errors over time. With only **10 minutes of demonstrations**, ACT enables robots to achieve **80-90% success rates** on fine manipulation tasks.

## Dataset  

ACT uses a dataset collected from real-world **bimanual teleoperation** experiments. The dataset consists of **human demonstrations**, meaning they gather their own data rather than relying on pre-existing datasets. The demonstration dataset consists of trajectories of **image observations, joint positions, and executed actions**.  

## Input & Output  
#### Input  
- **4 RGB images** (480×640 resolution) processed through **ResNet18**  
- **Joint positions** for the **two robot arms** (7+7=14 DOF)  

### Output  

- **Absolute joint positions** in chunks (e.g., next 100 timesteps)

## Architecture

<div>
    <img src="./images/act1.png" alt="Global Trajectory">
</div><br>

<div>
    <img src="./images/act2.png" alt="Global Trajectory">
</div><br>

## 2) Octo & RT2 & OpenX Embodiment

## Dataset  
- **Octo** is trained on **800k robot trajectories** from the **Open X-Embodiment dataset**.  
- This dataset aggregates **robotic demonstrations from 9 different robotic platforms**, covering a wide range of tasks:  
  - **Pick-and-place**  
  - **Tool use**  
  - **Button pressing**  
  - **Drawer opening/closing**  
- The dataset contains **heterogeneous data**, including:  
  - **Different camera views** (wrist cameras, third-person).  
  - **Robots with varying DoFs**.  
  - **Task-conditioning signals** (language commands & goal images).  

---

## Training Procedure  
### **Model Architecture**  
- **Transformer-based policy**, operating on **tokenized representations** of state, action, and task information.  
- **Pretraining on Open X-Embodiment dataset** for generalization across multiple robotic platforms.  
- **Finetuning on specific robots** with only a **few thousand demonstrations** for adaptation.  
- **Number of Parameters** ????

### **Training Strategy**  
- **Pretraining:**  
  - Trained on **800k demonstrations** across **9 robot setups**.  
  - Uses **sequence modeling** to predict the next action given past states.  
  - **Masked modeling objective**: Predict missing actions given observations.  
- **Finetuning:**  
  - Enables **zero-shot generalization** and **better adaptation with finetuning**.  

---

## Input & Output  
### **Input:**  
- **Observations**:  
  - **RGB images** from multiple viewpoints (wrist cam, third-person).  
  - **Proprioceptive states** (joint positions, velocities).  
  - **Task conditioning**:  
    - **Text commands** (e.g., "Pick up the red cup").  
    - **Goal images** (e.g., "Make the scene look like this").  

### **Output:**  
- **Delta position actions**, predicting relative movement rather than absolute positions.  
- **Continuous control commands**, refining robotic motion dynamically.  
- **Diffusion Models** for **action generation**, improving smoothness and adaptability.  
- **Action Chunking**

---

## Metrics & Evaluation  
### **Metrics Used:**  
- **Success Rate ($S$)**: Measures the percentage of completed tasks.  
- **Task Completion Time ($T$)**: Evaluates efficiency in execution.  
- **Generalization Score ($G$)**: Measures adaptability to unseen tasks.  

### **Evaluation Experiments:**  
- Tested on **9 robotic platforms** across different institutions.  
- Evaluated on **zero-shot performance** and **finetuning efficiency**.  
- Experiments conducted in **both real-world and simulation settings**.  

### **Baselines Compared Against:**  
- **RT-1** (Google DeepMind’s robotics transformer policy).  
- **Gato** (DeepMind’s multi-modal policy for vision, text, and robotics).  
- **BC (Behavior Cloning)**, a standard supervised learning approach for action prediction.  
- **LMP (Latent Motor Policies)**, a method utilizing latent representations.  

---


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

- **Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn**: [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://arxiv.org/pdf/2304.13705)
- **Dibya Ghosh, Homer Walke, Karl Pertsch**: [*Octo: An Open-Source Generalist Robot Policy*](https://arxiv.org/pdf/2405.12213)
- **Anthony Brohan, Noah Brown, Justice Carbajal**: [*RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*](https://arxiv.org/pdf/2307.15818)
- **Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti**: [*OpenVLA: An Open-Source Vision-Language-Action Model*](https://arxiv.org/pdf/2406.09246)
- **Kevin Black, Noah Brown, Danny Driess**: [*π0: A Vision-Language-Action Flow Model for General Robot Control*](https://arxiv.org/pdf/2410.24164)
- **Figure AI**: [*Helix: A Vision-Language-Action Model for Generalist Humanoid Control*](https://www.figure.ai/news/helix)
- **Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau**: [*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*](https://arxiv.org/pdf/2303.04137)
- **Karl Pertsch, Kyle Stachowicz, Brian Ichter**: [*FAST: Efficient Action Tokenization for Vision-Language-Action Models*](https://arxiv.org/pdf/2501.09747)










tabs:

- diffusion?
- control (joints, delta, absolute)
- action chunk
- n_params
- latent space?
- history images used?
- history actions used?
- inference speed?
- current state used? (joint pos, joint speed, FTS)
- Can I add new data? (FTS, ...)

What do they use for action? (discretization in bins, continuous actions in delta EE, absoulute EE, diffusion models)




Dataset  
Input & Output
Model Architecture + model dim? 
Training procedure (loss function? which dataset used? they gathered their own experiments? they did finetuning somehow?) 
Metrics + Evaluaton experiments used + algorithms used as benchamarks


want you to address all these topics for the attached PDF on Action Chunk Transformer and asnwer with  .md format, using $, $$, ###!!. I want only info from the attached pdf and do NOT invent things!



## 6) Next Steps (take inspiration from guy on github)

Which part of a transformer are freezed to fineruning?
How do you do post training RL for GPTlike (humans feedback?) and manioulation like?
How do you tokenize actions?
Trasnformer encoder-decoder. Decoder quando usato? Figure lo usa, Octo no
Sembra che elix e ACT usano latent encoder, ma a test time ACT mette z=0.
Redout token di Octo Vs CLS di ACT Vs last_layer di Bile locomotion. Da dove prendo output transformer?

How much time to train OpenX? Where? Lets say I want to investigate new architecture with memory from past like locomotion Bike. What project can be done to get attention? Paper insieme?


trade-off: vuoi fare foundation model usable per diversi robots o killer manipulation pipeline that cna handle difficult tasks?

rete che usa primitive? però task come piagare maglie, ecc non riesci con primitive e basta, mentre industrial setting si (definire tasks industriali)






## Metrics, Evaluation, and Benchmarks  
- **Primary metric**: **Success rate**  
- **Experiments conducted on**:  
  - **2 simulated fine manipulation tasks** in **MuJoCo**  
  - **6 real-world bimanual tasks** using **ALOHA**  

### **Baseline algorithms compared**  
- **BC-ConvMLP**: Simple **behavior cloning** with a convolutional network  
- **BeT (Behavior Transformers)**: Uses **history-conditioned transformers**  
- **RT-1**: A transformer-based model that **discretizes actions**

### **Results**  
- **ACT outperforms all baselines**, particularly in **tasks requiring precision**  
- **Highest success rates**:  
  - **88%** (Ziploc sliding task)  
  - **96%** (Battery slotting task)  
- **Action chunking significantly improves performance** by reducing compounding errors  