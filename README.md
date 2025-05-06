- [Pretrain](#pretrain)
  - [Dense Models](#dense-models)
  - [MoE](#moe)
  - [Scaling Laws \& Emergent Analysis](#scaling-laws--emergent-analysis)
- [Instruction Tuning / Supervised Fine-tuning](#instruction-tuning--supervised-fine-tuning)
- [Fine-Tuning](#fine-tuning)
- [Alignment  /  RLHF](#alignment----rlhf)
- [Inference](#inference)
- [Reasoning](#reasoning)
- [AI Infra](#ai-infra)
- [AI4Code](#ai4code)
- [AIGC](#aigc)
- [LLM Agent](#llm-agent)
- [MLLMs](#mllms)



### Pretrain

#### Dense Models

* [2024/07] **The Llama 3 Herd of Models** | [paper](http://arxiv.org/abs/2407.21783) | [code](https://github.com/meta-llama/llama3)  
* [2024/05] **The Road Less Scheduled** | [paper](http://arxiv.org/abs/2405.15682) | [code](https://github.com/facebookresearch/schedule_free)
* [2023/08] **Continual Pre-Training of Large Language Models: How to (re)warm your model?** | [paper](http://arxiv.org/abs/2308.04014) 
* [2023/07] **Llama 2: Open Foundation and Fine-Tuned Chat Models** | [paper](http://arxiv.org/abs/2307.09288)
* [2023/03] **GPT-4 Technical Report** | [paper](http://arxiv.org/abs/2303.08774)
* [2023/02] **LLaMA: Open and Efficient Foundation Language Models** | [paper](http://arxiv.org/abs/2302.13971)  
* [2022/05] **OPT: Open Pre-trained Transformer Language Models** | [paper](http://arxiv.org/abs/2205.01068) 
* [2022/04] **PaLM: Scaling Language Modeling with Pathways** | [paper](http://arxiv.org/abs/2204.02311) 
* [2021/04] **RoFormer: Enhanced Transformer with Rotary Position Embedding** | [paper](http://arxiv.org/abs/2104.09864) 
* [2020/05] **Language Models are Few-Shot Learners(GPT-3)** | [paper](http://arxiv.org/abs/2005.14165) 
* [2019/02] **Language Models are Unsupervised Multitask Learners(GPT-2)** | [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [2018/10] **BERT: Pre-Training of Deep Bidirectional Transformers** | [paper](http://arxiv.org/abs/1810.04805) 
* [2018/06] **Improving Language Understanding by Generative Pre-Training(GPT-1)** | [paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 
* [2017/06] **Attention Is All You Need** | [paper](http://arxiv.org/abs/1706.03762) 
* [2016/07] **Layer Normalization** | [paper](http://arxiv.org/abs/1607.06450) 
* [2015/08] **Neural Machine Translation of Rare Words with Subword Units** | [paper](http://arxiv.org/abs/1508.07909) 
* [2013/01] **Efficient Estimation of Word Representations in Vector Space** | [paper](http://arxiv.org/abs/1301.3781) 

#### MoE

* [2024/05] **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** | [paper](http://arxiv.org/abs/2405.04434) | [code](https://github.com/deepseek-ai/DeepSeek-V2)
* [2024/01] **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models** | [paper](http://arxiv.org/abs/2401.06066) | [code](https://github.com/deepseek-ai/DeepSeek-MoE)
* [2024/01] **Mixtral of Experts** | [paper](http://arxiv.org/abs/2401.04088) 
* [2022/02] **ST-MoE: Designing Stable and Transferable Sparse Expert Models** | [paper](http://arxiv.org/abs/2202.08906)
* [2021/12] **GLaM: Efficient Scaling of Language Models with Mixture-of-Experts** | [paper](http://arxiv.org/abs/2112.06905)
* [2021/01] **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** | [paper](http://arxiv.org/abs/2101.03961) 
* [2020/06] **GShard: Scaling Giant Models with Conditional Computation** | [paper](http://arxiv.org/abs/2006.16668) 
* [2017/01] **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** | [paper](http://arxiv.org/abs/1701.06538) 

#### Scaling Laws & Emergent Analysis

* [2024/05] **Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations** | [paper](http://arxiv.org/abs/2405.18392)
* [2023/05] **Scaling Data-Constrained Language Models** | [paper](http://arxiv.org/abs/2305.16264) 
* [2023/04] **Are Emergent Abilities of Large Language Models a Mirage?** | [paper](http://arxiv.org/abs/2304.15004)
* [2022/06] **Emergent Abilities of Large Language Models** | [paper](http://arxiv.org/abs/2206.07682)
* [2022/03] **Training Compute-Optimal Large Language Models** | [paper](http://arxiv.org/abs/2203.15556)
* [2022/02] **Compute Trends Across Three Eras of Machine Learning** | [paper](http://arxiv.org/abs/2202.05924) 
* [2020/01] **Scaling Laws for Neural Language Models** | [paper](http://arxiv.org/abs/2001.08361) 


### Instruction Tuning / Supervised Fine-tuning

* [2024/12] **Instruction Tuning for Large Language Models: A Survey** | [paper](http://arxiv.org/abs/2308.10792)| [code](https://github.com/xiaoya-li/Instruction-Tuning-Survey)
* [2024/03] **COIG-CQIA: Quality Is All You Need for Chinese Instruction Fine-Tuning** | [paper](http://arxiv.org/abs/2403.18058)
* [2023/05] **LIMA: Less Is More for Alignment** | [paper](http://arxiv.org/abs/2305.11206)
* [2023/04] **WizardLM: Empowering Large Language Models to Follow Complex Instructions** | [paper](http://arxiv.org/abs/2304.12244)
* [2023/03] **Alpaca: A Strong, Replicable Instruction-Following Model** | [paper](https://crfm.stanford.edu/2023/03/13/alpaca.html) | [code](https://github.com/tatsu-lab/stanford_alpaca)
* [2022/12] **Self-Instruct: Aligning Language Models with Self-Generated Instructions** | [paper](http://arxiv.org/abs/2212.10560) | [code](https://github.com/yizhongw/self-instruct)
* [2021/09] **Finetuned Language Models Are Zero-Shot Learners** | [paper](http://arxiv.org/abs/2109.01652)

### Fine-Tuning

* [2023/12] **AdaLoRA – Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning** | [paper](http://arxiv.org/abs/2303.10512) | [code](https://github.com/QingruZhang/AdaLoRA)
* [2023/05] **QLoRA – Efficient Finetuning of Quantized LLMs** | [paper](http://arxiv.org/abs/2305.14314) | [code](https://github.com/artidoro/qlora)
* [2022/05] **Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning** | [paper](http://arxiv.org/abs/2205.05638) | [code](https://github.com/r-three/t-few)
* [2021/06] **BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models** | [paper](http://arxiv.org/abs/2106.10199) | [code](https://github.com/EladBenZaken/BitFit)
* [2021/06] **LoRA – Low-Rank Adaptation of Large Language Models** | [paper](http://arxiv.org/abs/2106.09685) | [code](https://github.com/microsoft/LoRA)
* [2021/04] **The Power of Scale for Parameter-Efficient Prompt Tuning** | [paper](http://arxiv.org/abs/2104.08691) | [code](https://github.com/google-research/prompt-tuning)
* [2021/01] **Prefix-Tuning – Optimizing Continuous Prompts for Generation** | [paper](http://arxiv.org/abs/2101.00190) | [code](https://github.com/XiangLi1999/PrefixTuning)
* [2020/12] **Parameter-Efficient Transfer Learning with Diff Pruning** | [paper](http://arxiv.org/abs/2012.07463)
* [2019/02] **Parameter-Efficient Transfer Learning for NLP** | [paper](http://arxiv.org/abs/1902.00751) | [code](https://github.com/google-research/adapter-bert)



### Alignment  /  RLHF 

* [2025/03] **Self-Rewarding Language Models** | [paper](http://arxiv.org/abs/2401.10020)  
* [2024/12] **The PRISM Alignment Dataset: What Participatory, Representative and Individualised Human Feedback Reveals About the Subjective and Multicultural Alignment of Large Language Models** | [paper](http://arxiv.org/abs/2404.16019)  
* [2024/11] **ReST-MCTS\*: LLM Self-Training via Process-Reward-Guided Tree Search** | [paper](http://arxiv.org/abs/2406.03816) | [code](https://github.com/THUDM/ReST-MCTS)  
* [2024/07] **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** | [paper](http://arxiv.org/abs/2305.18290) | [code](https://github.com/huggingface/trl)
* [2024/06] **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models** | [paper](http://arxiv.org/abs/2401.01335) | [code](https://github.com/uclaml/SPIN)
* [2024/06] **Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning** | [paper](http://arxiv.org/abs/2405.00451) | [code](https://github.com/YuxiXie/MCTS-DPO)  
* [2024/01] **Secrets of RLHF in Large Language Models Part II: Reward Modeling** | [paper](http://arxiv.org/abs/2401.06080)  
* [2023/07] **Secrets of RLHF in Large Language Models Part I: PPO** | [paper](http://arxiv.org/abs/2307.04964)  
* [2022/03] **Training language models to follow instructions with human feedback(InstructGPT)** | [paper](http://arxiv.org/abs/2203.02155)  
* [2017/08] **Proximal Policy Optimization Algorithms** | [paper](http://arxiv.org/abs/1707.06347) | [code](https://github.com/openai/baselines/tree/master/baselines/ppo1)  

### Inference 
* [2024/08] **RULER: What’s the Real Context Size of Your Long-Context Language Models?** | [paper](http://arxiv.org/abs/2404.06654)
* [2024/04] **Better & Faster Large Language Models via Multi-token Prediction** | [paper](http://arxiv.org/abs/2404.19737)
* [2024/01] **The What, Why, and How of Context Length Extension Techniques in Large Language Models — A Detailed Survey** | [paper](http://arxiv.org/abs/2401.07872)
* [2023/07] **Lost in the Middle: How Language Models Use Long Contexts** | [paper](http://arxiv.org/abs/2307.03172)
* [2022/12] **A Length-Extrapolatable Transformer** | [paper](https://arxiv.org/abs/2212.10554) 
* [2022/11] **Fast Inference from Transformers via Speculative Decoding** | [paper](http://arxiv.org/abs/2211.17192)
* [2022/10] **Contrastive Decoding: Open-ended Text Generation as Optimization** | [paper](https://arxiv.org/abs/2210.15097)
* [2018/11] **Blockwise Parallel Decoding for Deep Autoregressive Models** | [paper](http://arxiv.org/abs/1811.03115)
    
### Reasoning
* [2023/05] **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | [paper](http://arxiv.org/abs/2305.10601) | [code](https://github.com/princeton-nlp/tree-of-thought-llm) 
* [2022/11] **Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks** | [paper](http://arxiv.org/abs/2211.12588) | [code]()
* [2022/05] **Large Language Models are Zero-Shot Reasoners** | [paper](http://arxiv.org/abs/2205.11916)
* [2022/03] **Self-Consistency Improves Chain of Thought Reasoning in Language Models** | [paper](http://arxiv.org/abs/2203.11171)
* [2022/01] **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | [paper](http://arxiv.org/abs/2201.11903)

### AI Infra

* [2024/07] **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** | [paper](http://arxiv.org/abs/2407.08608) | [code](https://github.com/Dao-AILab/flash-attention)
* [2023/10] **Ring Attention with Blockwise Transformers for Near-Infinite Context** | [paper](http://arxiv.org/abs/2310.01889) 
* [2023/09] **DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models** | [paper](http://arxiv.org/abs/2309.14509) | [code](https://github.com/microsoft/DeepSpeed)
* [2023/07] **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** | [paper](http://arxiv.org/abs/2307.08691) | [code](https://github.com/Dao-AILab/flash-attention)
* [2022/05] **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | [paper](http://arxiv.org/abs/2205.14135) | [code](https://github.com/HazyResearch/flash-attention)
* [2022/01] **Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model** | [paper](http://arxiv.org/abs/2201.11990) | [code](https://github.com/microsoft/DeepSpeed)
* [2021/04] **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** | [paper](http://arxiv.org/abs/2104.04473) | [code](https://github.com/NVIDIA/Megatron-LM)
* [2019/10] **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** | [paper](http://arxiv.org/abs/1910.02054)


### AI4Code 

* [2025/04] **QiMeng-GEMM: Automatically Generating High-Performance Matrix Multiplication Code by Exploiting Large Language Models** | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/34461)
* [2025/01] **A Survey of Neural Code Intelligence: Paradigms, Advances and Beyond** | [paper](http://arxiv.org/abs/2403.14734)
* [2024/12] **AGON: Automated Design Framework for Customizing Processors from ISA Documents** | [paper](http://arxiv.org/abs/2412.20954)
* [2024/11] **OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models** | [paper](http://arxiv.org/abs/2411.04905) | [code](https://github.com/OpenCoder-llm/OpenCoder-llm)
* [2024/11] **From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge** | [paper](http://arxiv.org/abs/2411.16594) 
* [2024/11] **A Survey on Large Language Models for Code Generation** | [paper](http://arxiv.org/abs/2406.00515)
* [2024/11] **Divide-and-Conquer Meets Consensus: Unleashing the Power of Functions in Code Generation** | [paper](http://arxiv.org/abs/2405.20092) | [code](https://github.com/cometeme/funcoder)
* [2024/11] **SelfCodeAlign: Self-Alignment for Code Generation** | [paper](http://arxiv.org/abs/2410.24198) | [code](https://github.com/bigcode-project/selfcodealign)
* [2024/09] **ComBack: A Versatile Dataset for Enhancing Compiler Backend Development Efficiency** | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cbc9ba6ccf2db854fc8de25a2741e021-Abstract-Datasets_and_Benchmarks_Track.html)
* [2024/08] **DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search** | [paper](http://arxiv.org/abs/2408.08152) | [code](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)
* [2024/07] **CodeV: Empowering LLMs for Verilog Generation through Multi-Level Summarization** | [paper](http://arxiv.org/abs/2407.10424) | [code](https://github.com/IPRC-DIP/CodeV)
* [2024/06] **Magicoder: Empowering Code Generation with OSS-Instruct** | [paper](http://arxiv.org/abs/2312.02120) | [code](https://github.com/ise-uiuc/magicoder)
* [2024/06] **Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code** | [paper](http://arxiv.org/abs/2311.07989)
* [2024/06] **McEval: Massively Multilingual Code Evaluation** | [paper](http://arxiv.org/abs/2406.07436) | [code](https://github.com/MCEVAL/McEval)
* [2024/06] **LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code** | [paper](http://arxiv.org/abs/2403.07974) | [code](https://github.com/LiveCodeBench/LiveCodeBench)
* [2024/05] **DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data** | [paper](http://arxiv.org/abs/2405.14333) | [code](https://github.com/deepseek-ai/DeepSeek-Prover)
* [2024/04] **ChipNeMo: Domain-Adapted LLMs for Chip Design** | [paper](http://arxiv.org/abs/2311.00176)
* [2024/01] **A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends** | [paper](http://arxiv.org/abs/2311.10372) 
* [2023/10] **LILO: Learning Interpretable Libraries by Compressing and Documenting Code** | [paper](http://arxiv.org/abs/2310.19791) | [code](https://github.com/gabegrand/lilo)
* [2023/12] **VerilogEval: Evaluating Large Language Models for Verilog Code Generation** | [paper](http://arxiv.org/abs/2309.07544) | [code](https://github.com/NVlabs/verilog-eval)
* [2023/11] **Chip-Chat: Challenges and Opportunities in Conversational Hardware Design** | [paper](http://arxiv.org/abs/2305.13243)| [code](https://github.com/MJoergen/ChipChatData) 
* [2023/11] **ANPL: Towards Natural Programming with Interactive Decomposition** | [paper](http://arxiv.org/abs/2305.18498) | [code](https://github.com/IPRC-DIP/ANPL)
* [2023/10] **CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code** | [paper](http://arxiv.org/abs/2302.05527)
* [2023/10] **WizardCoder: Empowering Code Large Language Models with Evol-Instruct** | [paper](http://arxiv.org/abs/2306.08568) | [code](https://github.com/nlpxucan/WizardLM)
* [2023/09] **Measuring Coding-Challenge Competence with APPS** | [paper](http://arxiv.org/abs/2105.09938) | [code](https://github.com/hendrycks/apps)
* [2023/06] **Program Synthesis with Large Language Models** | [paper](http://arxiv.org/abs/2108.07732)
* [2022/11] **CodeT: Code Generation with Generated Tests** | [paper](http://arxiv.org/abs/2207.10397) | [code](https://github.com/microsoft/CodeT/tree/main/CodeT)
* [2022/07] **Efficient Training of Language Models to Fill in the Middle** | [paper](http://arxiv.org/abs/2207.14255)
* [2022/04] **InCoder: A Generative Model for Code Infilling and Synthesis** | [paper](http://arxiv.org/abs/2204.05999) | [code](https://github.com/dpfried/incoder)
* [2021/11] **Measuring Coding Challenge Competence With APPS** | [paper](http://arxiv.org/abs/2105.09938) | [code](https://github.com/hendrycks/apps)
* [2021/08] **CodeBLEU: A Method for Automatic Evaluation of Code Synthesis** | [paper](http://arxiv.org/abs/2009.10297)
* [2020/09] **Evaluating Large Language Models Trained on Code** | [paper](http://arxiv.org/abs/2107.03374) 

### AIGC

* [2023/02] **Adding Conditional Control to Text-to-Image Diffusion Models** | [paper](http://arxiv.org/abs/2302.05543) | [code](https://github.com/lllyasviel/ControlNet)
* [2023/03] **Scalable Diffusion Models with Transformers (DiT)** | [paper](http://arxiv.org/abs/2212.09748) | [code](https://github.com/facebookresearch/DiT)
* [2022/09] **DreamFusion: Text-to-3D using 2D Diffusion** | [paper](http://arxiv.org/abs/2209.14988) 
* [2022/05] **Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)** | [paper](http://arxiv.org/abs/2205.11487) 
* [2022/04] **Hierarchical Text-Conditional Image Generation with CLIP Latents(DALL-E 2)** | [paper](https://cdn.openai.com/papers/dall-e-2.pdf) 
* [2021/12] **High-Resolution Image Synthesis with Latent Diffusion Models** | [paper](http://arxiv.org/abs/2112.10752) | [code](https://github.com/CompVis/latent-diffusion)
* [2020/06] **Denoising Diffusion Probabilistic Models(Diffusion)** | [paper](http://arxiv.org/abs/2006.11239) | [code](https://github.com/hojonathanho/diffusion)


### LLM Agent

* [2025/03] **A Survey on Large Language Model based Autonomous Agents** | [paper](http://arxiv.org/abs/2308.11432) 
* [2023/10] **MemGPT: Towards LLMs as Operating Systems** | [paper](http://arxiv.org/abs/2310.08560) | [code](https://github.com/cpacker/MemGPT)
* [2023/09] **The Rise and Potential of Large Language Model Based Agents: A Survey** | [paper](http://arxiv.org/abs/2309.07864) 
* [2023/05] **Voyager: An Open-Ended Embodied Agent with Large Language Models** | [paper](http://arxiv.org/abs/2305.16270) | [code](https://github.com/MineDojo/Voyager)
* [2023/03] **ReAct: Synergizing Reasoning and Acting in Language Models** | [paper](http://arxiv.org/abs/2210.03629) 
* [2023/02] **Toolformer: Language Models Can Teach Themselves to Use Tools** | [paper](http://arxiv.org/abs/2302.04761) 
* [2023/04] **Generative Agents: Interactive Simulacra of Human Behavior** | [paper](http://arxiv.org/abs/2304.03442) | [code](https://github.com/joonspk-research/generative_agents)
* [2023/04] **LLM+P: Empowering Large Language Models with Optimal Planning Proficiency** | [paper](http://arxiv.org/abs/2304.11477) | [code](https://github.com/Cranial-XIX/llm-pddl)
* [2023/03] **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace** | [paper](http://arxiv.org/abs/2303.17580) | [code](https://github.com/AI-Chef/HuggingGPT)
* [2023/03] **PaLM-E: An Embodied Multimodal Language Model** | [paper](http://arxiv.org/abs/2303.03378) | [code](https://github.com/kyegomez/PALM-E)
* [2023/03] **Reflexion: Language Agents with Verbal Reinforcement Learning** | [paper](http://arxiv.org/abs/2303.11366) | [code](https://github.com/noahshinn/reflexion)
* [2022/12] **LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models** | [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_LLM-Planner_Few-Shot_Grounded_Planning_for_Embodied_Agents_with_Large_Language_ICCV_2023_paper.pdf) | [code](https://github.com/OSU-NLP-Group/LLM-Planner)
* [2022/11] **PAL: Program-Aided Language Models** | [paper](http://arxiv.org/abs/2211.10435) | [code](https://github.com/reasoning-machines/pal)
* [2022/10] **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances** | [paper](http://arxiv.org/abs/2204.01691) | [code](https://github.com/kyegomez/SayCan)
* [2022/09] **Code as Policies: Language Model Programs for Embodied Control** | [paper](http://arxiv.org/abs/2209.07753) | [code](https://code-as-policies.github.io)
* [2022/05] **TALM: Tool Augmented Language Models** | [paper](http://arxiv.org/abs/2205.12255)
* [2015/06] **Language Understanding for Text-based Games Using Deep Reinforcement Learning** | [paper](http://arxiv.org/abs/1506.08941)


### MLLMs
* [2023/10] **Improved Baselines with Visual Instruction Tuning** | [paper](http://arxiv.org/abs/2310.03744) | [code](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file)
* [2023/04] **Visual Instruction Tuning(LLaVA)** | [paper](http://arxiv.org/abs/2304.08485) | [code](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file)
* [2023/01] **BLIP-2: Bootstrapping Language-Image Pre-training** | [paper](http://arxiv.org/abs/2301.12597) | [code](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
* [2022/04] **Flamingo: a Visual Language Model for Few-Shot Learning** | [paper](http://arxiv.org/abs/2204.14198)
* [2022/01] **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** | [paper](http://arxiv.org/abs/2201.12086) | [code](https://github.com/salesforce/BLIP)
* [2021/07] **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation** | [paper](http://arxiv.org/abs/2107.07651) | [code](https://github.com/salesforce/ALBEF)
* [2021/01] **Learning Transferable Visual Models From Natural Language Supervision** | [paper](https://proceedings.mlr.press/v139/radford21a.html) | [code](https://github.com/openai/CLIP)

