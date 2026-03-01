# Computer Use with a SLM using GRPO and OpenEnv
 
this project involves fine-tuning a Small Language Model using Reinforcement Learning (GRPO) to independently learn to navigate environments and interact with browser elements, essentially enabling Computer Use. I've taught the model to click a button element but it could be extended to more complex tasks like filling a form and booking a flight based on environmental rewards. I used A100 40GB GPU via Modal requiring around ~$3 worth of compute for the ```click-test``` task. You can find the Hugging Face link to my trained model [here](https://huggingface.co/DingoBeast/LFM2-350M-ComputerUse).
 
<img width="3872" height="1088" alt="computer use with slm" src="https://github.com/user-attachments/assets/b0ad81ab-47f5-401d-84bd-263821009a1d" />

## system architecture & tools

we connect a local development environment to serverless cloud GPUs, where an AI agent learns by interacting with a sandboxed browser.

* **Model:** ```LiquidAI/LFM2-350M```. A very efficient Language Model that requires significantly less compute than LLMs. [Model Link on HF](https://huggingface.co/LiquidAI/LFM2-350M)
* **vLLM:** a high-throughput inference engine. During the reinforcement learning loop, vLLM serves the model to generate trajectory rollouts (the agent's attempts at the task) exponentially faster than standard PyTorch.
* **Hugging Face TRL & PEFT:** the ```trl``` library handles the GRPO (Group Relative Policy Optimization) training loop. To optimize memory, ```peft``` applies LoRA, fine-tuning only a tiny subset of adapter weights (~10MB) instead of the full 350M parameters.
* **Weights & Biases:** dashboard to tracks the agent's reward metrics, success rates, and loss curves throughout the training process.
* **BrowserGym & OpenEnv:** the web sandbox. BrowserGym renders the actual web tasks (like [click-test](https://miniwob.farama.org/environments/click-test/)) and translates the webpage's DOM into a text-based Accessibility Tree (AXTree) that the SLM can read. [OpenEnv](https://github.com/meta-pytorch/OpenEnv) provides the standard API wrapper to send actions to the browser and calculate task success.
* **Hugging Face Spaces:** the remote server actively hosting the BrowserGym environment. During both training and evaluation, the script sends HTTP requests to this Space to execute the agent's actions and fetch the updated webpage state.
* **Docker:** packages all heavy machine learning dependencies (PyTorch, vLLM, CUDA drivers) into a standardized container image. This ensures the environment runs perfectly in the cloud without requiring massive local installations.
* **Modal:** the serverless GPU platform. Modal takes the defined Docker image and local Python scripts, instantly spins up cloud compute (e.g., A100/L40S GPUs) to run the training job, and shuts down the second it finishes.

## training configs
the training parameters are managed via ```configs/lfm2_350m_lora.yaml```:

* **Steps/Dataset Size:** 100 training iterations.
* **Max Steps per Episode:** 10 browser actions.
* **Generations per Prompt:** 4.
* **LoRA Configuration:** Rank (r) = 8, Alpha (α) = 16.


## training results

| Reward Mean | Completion Length | Entropy |
| :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/b7cf6f6a-29b4-434c-aa55-463c205c9e38" alt="Reward Mean" width="100%">  | <img src="https://github.com/user-attachments/assets/5a575da2-bcfe-4b1f-9f7f-2fbf3c613d91" alt="Completion Length" width="100%"> | <img src="https://github.com/user-attachments/assets/3d88f758-84de-4ead-a512-3d651b3c6311" alt="Entropy" width="100%"> |

## usage
### 1. local setup
clone the repo and install all dependencies using uv:
```
git clone https://github.com/shayanamir0/computer-use-with-slm.git
cd "computer use with slm"
uv sync
```

### 2. cloud authentication
ensure you are authenticated with the Hugging Face CLI (for pulling base models and pushing adapters) and Modal (for cloud compute):
```
uv run huggingface-cli login
uv run modal setup
```
Go to your Modal Dashboard Secrets and create a new custom secret named wandb-secret. Add the following two key-value pairs to it:
* **WANDB_API_KEY:** Your Weights & Biases API key.
* **HF_TOKEN:** Your Hugging Face access token (ensure it has **Write** permissions to create repositories).

### 3. run the training 
```
uv run modal run -m src.cua.finetune --config-file-name lfm2_350m_lora.yaml
```
once the 100 iterations complete, the script will automatically push the trained LoRA weights directly to your Hugging Face account.

### 4. local eval
to visually test the model's performance on your local machine:
 1. Open ```src/cua/evaluate.py```.
 2. Update the model_name variable to point to your newly created Hugging Face repository (e.g., "your-username/LFM2-350M-ComputerUse").
run the evaluation script:
```
uv run python -m src.cua.evaluate
```
this script will run the model against the BrowserGym sandbox locally. It captures the agent's step-by-step navigation and saves the visual output as screenshots in the ```./media/``` directory.

