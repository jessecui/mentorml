"""
Where the production diagram corpus comes from.

Edit this file to widen coverage, then re-run `python scripts/curate_corpus.py`.
Pages and images already fetched are cached, and images already scored are
not scored again, so adding a few entries is cheap.

Each source is a dict:
    key        short stable identifier, stored on every candidate
    name       shown to users as the source of an image
    pages      explicit page URLs to pull figures from
    crawl      seed URLs; same-site links matching `link_pattern` are followed
               one level deep (used for books/courses with many short pages)
    link_pattern   regex a crawled URL must match
    skip_pattern   regex; matching URLs are not fetched
    arxiv_ids  arXiv paper IDs; the HTML version is used when one exists
    delay      seconds between requests to this source's host
"""

SOURCES: list[dict] = [
    {
        "key": "jalammar",
        "name": "Jay Alammar",
        "delay": 0.5,
        "pages": [
            "https://jalammar.github.io/illustrated-transformer/",
            "https://jalammar.github.io/illustrated-bert/",
            "https://jalammar.github.io/illustrated-gpt2/",
            "https://jalammar.github.io/illustrated-word2vec/",
            "https://jalammar.github.io/illustrated-stable-diffusion/",
            "https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/",
            "https://jalammar.github.io/how-gpt3-works-visualizations-animations/",
            "https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/",
            "https://jalammar.github.io/illustrated-retrieval-transformer/",
            "https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/",
        ],
    },
    {
        "key": "lilianweng",
        "name": "Lil'Log (Lilian Weng)",
        "delay": 0.5,
        "pages": [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2018-02-19-rl-overview/",
            "https://lilianweng.github.io/posts/2018-04-08-policy-gradient/",
            "https://lilianweng.github.io/posts/2021-07-11-diffusion-models/",
            "https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/",
            "https://lilianweng.github.io/posts/2018-06-24-attention/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
            "https://lilianweng.github.io/posts/2018-08-12-vae/",
            "https://lilianweng.github.io/posts/2017-08-20-gan/",
            "https://lilianweng.github.io/posts/2019-01-31-lm/",
            "https://lilianweng.github.io/posts/2018-10-13-flow-models/",
            "https://lilianweng.github.io/posts/2020-10-29-odqa/",
            "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
            "https://lilianweng.github.io/posts/2025-05-01-thinking/",
            "https://lilianweng.github.io/posts/2017-06-21-overview/",
            "https://lilianweng.github.io/posts/2017-10-15-word-embedding/",
            "https://lilianweng.github.io/posts/2021-09-25-train-large/",
            "https://lilianweng.github.io/posts/2023-01-10-inference-optimization/",
            "https://lilianweng.github.io/posts/2022-06-09-vlm/",
            "https://lilianweng.github.io/posts/2020-06-07-exploration-drl/",
            "https://lilianweng.github.io/posts/2018-11-30-meta-learning/",
            "https://lilianweng.github.io/posts/2021-12-05-semi-supervised/",
        ],
    },
    {
        "key": "d2l",
        "name": "Dive into Deep Learning",
        "delay": 0.4,
        "crawl": [
            "https://d2l.ai/chapter_linear-regression/index.html",
            "https://d2l.ai/chapter_linear-classification/index.html",
            "https://d2l.ai/chapter_multilayer-perceptrons/index.html",
            "https://d2l.ai/chapter_convolutional-neural-networks/index.html",
            "https://d2l.ai/chapter_convolutional-modern/index.html",
            "https://d2l.ai/chapter_recurrent-neural-networks/index.html",
            "https://d2l.ai/chapter_recurrent-modern/index.html",
            "https://d2l.ai/chapter_optimization/index.html",
            "https://d2l.ai/chapter_attention/index.html",
            "https://d2l.ai/chapter_transformers/index.html",
            "https://d2l.ai/chapter_reinforcement-learning/index.html",
            "https://d2l.ai/chapter_deep-reinforcement-learning/index.html",
            "https://d2l.ai/chapter_generative-adversarial-networks/index.html",
            "https://d2l.ai/chapter_diffusion-models/index.html",
            "https://d2l.ai/chapter_lm-pretrained/index.html",
            "https://d2l.ai/chapter_lm-encoders/index.html",
            "https://d2l.ai/chapter_lm-pretraining/index.html",
            "https://d2l.ai/chapter_lm-finetuning/index.html",
            "https://d2l.ai/chapter_lm-rl/index.html",
            "https://d2l.ai/chapter_lm-reasoning/index.html",
            "https://d2l.ai/chapter_lm-agents/index.html",
            "https://d2l.ai/chapter_lm-compression/index.html",
            "https://d2l.ai/chapter_lm-inference/index.html",
            "https://d2l.ai/chapter_lm-architectures/index.html",
            "https://d2l.ai/chapter_computer-vision/index.html",
            "https://d2l.ai/chapter_introduction/index.html",
            "https://d2l.ai/chapter_preliminaries/index.html",
            "https://d2l.ai/chapter_builders-guide/index.html",
            "https://d2l.ai/chapter_computational-performance/index.html",
            "https://d2l.ai/chapter_hyperparameter-optimization/index.html",
            "https://d2l.ai/chapter_language-models/index.html",
            "https://d2l.ai/chapter_lm-data/index.html",
            "https://d2l.ai/chapter_lm-hardware/index.html",
            "https://d2l.ai/chapter_lm-parallelism/index.html",
            "https://d2l.ai/chapter_lm-serving/index.html",
            "https://d2l.ai/chapter_image-models/index.html",
            "https://d2l.ai/chapter_mdl-optimization/index.html",
            "https://d2l.ai/chapter_mdl-calculus/index.html",
        ],
        "link_pattern": r"^https://d2l\.ai/chapter_[\w-]+/[\w-]+\.html$",
        "skip_pattern": r"(further-reading|exercises|summary)\.html$",
    },
    {
        "key": "hf-deep-rl",
        "name": "Hugging Face Deep RL Course",
        "delay": 0.4,
        "crawl": [
            f"https://huggingface.co/learn/deep-rl-course/{unit}/introduction"
            for unit in ("unit1", "unit2", "unit3", "unit4", "unit6", "unit8")
        ],
        "link_pattern": r"^https://huggingface\.co/learn/deep-rl-course/unit\d+/[\w-]+$",
        "skip_pattern": r"/(quiz\d*|hands-on|conclusion|glossary|additional-readings|summary)$",
    },
    {
        "key": "hf-agents",
        "name": "Hugging Face Agents Course",
        "delay": 0.4,
        "crawl": ["https://huggingface.co/learn/agents-course/unit1/introduction"],
        "link_pattern": r"^https://huggingface\.co/learn/agents-course/unit1/[\w-]+$",
        "skip_pattern": r"/(quiz\d*|final-quiz|conclusion|dummy-agent-library|tutorial|onboarding)$",
    },
    {
        "key": "hf-llm",
        "name": "Hugging Face LLM Course",
        "delay": 0.4,
        "crawl": ["https://huggingface.co/learn/llm-course/chapter1/1"],
        "link_pattern": r"^https://huggingface\.co/learn/llm-course/chapter1/\d+$",
    },
    {
        "key": "colah",
        "name": "colah's blog (Christopher Olah)",
        "delay": 0.5,
        "pages": [
            "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
            "https://colah.github.io/posts/2015-08-Backprop/",
            "https://colah.github.io/posts/2014-07-Conv-Nets-Modular/",
            "https://colah.github.io/posts/2014-07-Understanding-Convolutions/",
            "https://colah.github.io/posts/2015-09-NN-Types-FP/",
        ],
    },
    {
        "key": "cs231n",
        "name": "Stanford CS231n notes",
        "delay": 0.5,
        "pages": [
            "https://cs231n.github.io/neural-networks-1/",
            "https://cs231n.github.io/neural-networks-2/",
            "https://cs231n.github.io/neural-networks-3/",
            "https://cs231n.github.io/optimization-1/",
            "https://cs231n.github.io/optimization-2/",
            "https://cs231n.github.io/convolutional-networks/",
            "https://cs231n.github.io/linear-classify/",
            "https://cs231n.github.io/transfer-learning/",
        ],
    },
    {
        "key": "distill",
        "name": "Distill",
        "delay": 0.5,
        "pages": [
            "https://distill.pub/2016/augmented-rnns/",
            "https://distill.pub/2019/computing-receptive-fields/",
            "https://distill.pub/2016/deconv-checkerboard/",
            "https://distill.pub/2017/momentum/",
            "https://distill.pub/2021/gnn-intro/",
        ],
    },
    {
        "key": "google-mlcc",
        "name": "Google Machine Learning Crash Course",
        "delay": 0.5,
        "crawl": [
            f"https://developers.google.com/machine-learning/crash-course/{module}"
            for module in ("linear-regression", "logistic-regression", "classification", "neural-networks",
                           "embeddings", "overfitting", "categorical-data", "numerical-data", "llm")
        ],
        "link_pattern": r"^https://developers\.google\.com/machine-learning/crash-course/[\w-]+/[\w-]+$",
        "skip_pattern": r"/(quiz|test-your-knowledge|programming-exercise|whats-next|conclusion)[\w-]*$",
    },
    {
        "key": "arxiv",
        "name": "arXiv",
        "delay": 3.0,  # arXiv asks automated clients to go slowly
        "arxiv_ids": [
            # Foundations
            "1301.3781",   # word2vec
            "1409.3215",   # seq2seq
            "1409.0473",   # Bahdanau attention
            "1512.03385",  # ResNet
            "1406.2661",   # GAN
            "1312.6114",   # VAE
            "1503.02531",  # knowledge distillation
            # Transformers and language models
            "1706.03762",  # Attention Is All You Need
            "1810.04805",  # BERT
            "1910.10683",  # T5
            "2005.14165",  # GPT-3
            "2104.09864",  # RoPE
            "2106.09685",  # LoRA
            "2305.14314",  # QLoRA
            "2312.00752",  # Mamba
            # Vision and multimodal
            "2010.11929",  # ViT
            "2103.00020",  # CLIP
            "2002.05709",  # SimCLR
            "2111.06377",  # MAE
            "2103.14030",  # Swin
            # Diffusion
            "2006.11239",  # DDPM
            "2112.10752",  # latent diffusion
            "2204.06125",  # unCLIP / DALL-E 2
            # Reinforcement learning
            "1312.5602",   # DQN
            "1602.01783",  # A3C
            "1509.02971",  # DDPG
            "1707.06347",  # PPO
            "1801.01290",  # SAC
            "2106.01345",  # Decision Transformer
            # Alignment and human feedback
            "1706.03741",  # deep RL from human preferences
            "2203.02155",  # InstructGPT (RLHF)
            "2305.18290",  # DPO
            "2212.08073",  # Constitutional AI
            # Retrieval, reasoning and agents
            "2005.11401",  # RAG
            "2004.04906",  # DPR
            "2201.11903",  # chain-of-thought
            "2210.03629",  # ReAct
            "2303.11366",  # Reflexion
            "2305.10601",  # Tree of Thoughts
            "2302.04761",  # Toolformer
            "2304.03442",  # Generative Agents
            "2308.11432",  # survey of LLM-based autonomous agents
            "2308.08155",  # AutoGen
            # Added to fill gaps found by scripts/audit_coverage.py
            "1803.08494",  # Group Normalization (batch vs layer vs instance vs group norm)
            "1508.04025",  # Luong attention (global and local attention in seq2seq)
            "1803.10122",  # World Models
            "1912.01603",  # Dreamer
            "2207.12598",  # classifier-free guidance
            "1712.05877",  # integer quantization
            "2204.14198",  # Flamingo (cross-attention between vision and language)
            "2103.03206",  # Perceiver
            "1502.03167",  # batch normalization
            "1607.06450",  # layer normalization
            "1503.03832",  # FaceNet triplet loss
        ],
    },
]

# Images that must never be accepted. The build step fails loudly if one is,
# which means the rubric has drifted and needs attention.
KNOWN_REJECTS: list[str] = [
    # Five model names drawn as Sesame Street characters: looks clean, teaches nothing.
    "https://jalammar.github.io/images/transformer-ber-ulmfit-elmo.png",
]
