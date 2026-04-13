import copy
import json
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights

from clip import clip
from clip.model import QuickGELU
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .hicropl import CoPrompt_dataset_name_mapping, gpt_clip_classifier


_tokenizer = _Tokenizer()


def _get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n_layers)])


def load_clip_to_cpu_teacher(cfg):
    backbone_name = cfg.TRAINER.HICROSPLIP.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "IVLP",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if zero_shot_model:
        design_details = {
            "trainer": "IVLP",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
    else:
        design_details = {
            # Reuse the HiCroPL-capable CLIP blocks while reading HiCroSPLIP config.
            "trainer": "HiCroPL",
            "vision_depth": cfg.TRAINER.HICROSPLIP.PROMPT_DEPTH,
            "language_depth": cfg.TRAINER.HICROSPLIP.PROMPT_DEPTH,
            "vision_ctx": cfg.TRAINER.HICROSPLIP.N_CTX,
            "language_ctx": cfg.TRAINER.HICROSPLIP.N_CTX,
        }

    return clip.build_model(state_dict or model.state_dict(), design_details)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.transformer([x, cross_prompts_text_deeper])
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, token_query, sequence_key, sequence_value):
        token_query = token_query + self.attn(
            self.ln_1(token_query),
            self.ln_1(sequence_key),
            self.ln_1(sequence_value),
            need_weights=False,
        )[0]
        return self.ln_2(token_query)


class CrossPromptAttention(nn.Module):
    def __init__(self, hidden_size, encoder_hidden_size, num_attention_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_attention_heads)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(encoder_hidden_size, hidden_size)
        self.linear_v = nn.Linear(encoder_hidden_size, hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(hidden_size, hidden_size * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(hidden_size * 4, hidden_size)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(hidden_size)

    def forward(self, q, k, v):
        q_proj = self.linear_q(q)
        k_proj = self.linear_k(k)
        v_proj = self.linear_v(v)
        q_proj = q_proj + self.attn(
            self.ln_1(q_proj),
            self.ln_1(k_proj),
            self.ln_1(v_proj),
            need_weights=False,
        )[0]
        q_proj = q_proj + self.ffn(self.ln_2(q_proj))
        return q_proj


class TokenPromptMapper(nn.Module):
    def __init__(self, input_dim, output_dim, n_tokens, hidden_dim=None, num_heads=8):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        self.query_tokens = nn.Parameter(torch.randn(n_tokens, input_dim) * 0.02)
        self.ln_q = nn.LayerNorm(input_dim)
        self.ln_kv = nn.LayerNorm(input_dim)
        self.attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            QuickGELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.ln_out = nn.LayerNorm(output_dim)

    def forward(self, tokens):
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)

        batch_size = tokens.shape[0]
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled = self.attn(
            self.ln_q(query),
            self.ln_kv(tokens),
            self.ln_kv(tokens),
            need_weights=False,
        )[0]
        prompts = self.mlp(query + pooled)
        return self.ln_out(prompts)


class HiCroSPLIPPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.HICROSPLIP.N_CTX
        self.cross_prompts_depth = cfg.TRAINER.HICROSPLIP.PROMPT_DEPTH
        self.cross_layer = cfg.TRAINER.HICROSPLIP.CROSS_LAYER
        self.dtype = clip_model.dtype

        assert self.cross_prompts_depth >= 1, "HICROSPLIP requires PROMPT_DEPTH >= 1"
        assert 0 <= self.cross_layer <= self.cross_prompts_depth, "CROSS_LAYER must be within prompt depth"

        ctx_init = cfg.TRAINER.HICROSPLIP.CTX_INIT
        ctx_dim = clip_model.ln_final.weight.shape[0]
        v_dim = clip_model.visual.conv1.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and self.n_ctx <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        print("HiCroSPLIP design: HiCroPL with SpLiP-style input token injection")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of HiCroSPLIP context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        cross_prompts_text = nn.ParameterList(
            [self.ctx] + [nn.Parameter(torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)) for _ in range(self.cross_prompts_depth - 1)]
        )
        for single_para in cross_prompts_text[1:]:
            nn.init.normal_(single_para, std=0.02)
        self.cross_prompts_text = cross_prompts_text

        visual_vectors = torch.empty(self.n_ctx, v_dim, dtype=self.dtype)
        nn.init.normal_(visual_vectors, std=0.02)
        self.cross_prompts_visual = nn.ParameterList(
            [nn.Parameter(visual_vectors.clone()) for _ in range(self.cross_prompts_depth)]
        )

        self.text2visual_net = CrossPromptAttention(hidden_size=v_dim, encoder_hidden_size=ctx_dim, num_attention_heads=8)
        self.visual2text_net = CrossPromptAttention(hidden_size=ctx_dim, encoder_hidden_size=v_dim, num_attention_heads=8)

        attn_pooling_text = AttentionPooling(hidden_size=ctx_dim, num_attention_heads=8)
        self.attn_pooling_text_nets = _get_clones(attn_pooling_text, self.cross_layer)
        attn_pooling_visual = AttentionPooling(hidden_size=v_dim, num_attention_heads=8)
        self.attn_pooling_visual_nets = _get_clones(
            attn_pooling_visual, max(self.cross_prompts_depth - self.cross_layer, 0)
        )

        self.text_proxy_token = nn.ParameterList(
            [nn.Parameter(torch.randn(1, ctx_dim, dtype=self.dtype)) for _ in range(self.cross_layer)]
        )
        self.visual_proxy_token = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, v_dim, dtype=self.dtype))
                for _ in range(self.cross_layer, self.cross_prompts_depth)
            ]
        )

        self.image_to_text_anchor = TokenPromptMapper(v_dim, ctx_dim, self.n_ctx, hidden_dim=v_dim)
        self.text_to_visual_anchor = TokenPromptMapper(ctx_dim, v_dim, self.n_ctx, hidden_dim=ctx_dim)
        self.text_anchor_gates = nn.Parameter(torch.zeros(self.cross_prompts_depth, 1, 1, dtype=self.dtype))
        self.visual_anchor_gates = nn.Parameter(torch.zeros(self.cross_prompts_depth, 1, 1, dtype=self.dtype))

        if cfg.TRAINER.HICROSPLIP.PREC == "fp16":
            self.text2visual_net = self.text2visual_net.half()
            self.visual2text_net = self.visual2text_net.half()
            self.attn_pooling_text_nets = self.attn_pooling_text_nets.half()
            self.attn_pooling_visual_nets = self.attn_pooling_visual_nets.half()
            self.image_to_text_anchor = self.image_to_text_anchor.half()
            self.text_to_visual_anchor = self.text_to_visual_anchor.half()

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])
        self.register_buffer("seed_prefix", embedding[:1, :1, :].clone())
        self.register_buffer("seed_suffix", embedding[:, 1 + self.n_ctx :, :].mean(dim=0, keepdim=True))
        self.tokenized_prompts = tokenized_prompts

        conv1 = clip_model.visual.conv1
        self.register_buffer("image_patch_proj_weight", conv1.weight.detach().clone())
        self.patch_stride = conv1.stride
        self.register_buffer(
            "image_patch_positional_embedding",
            clip_model.visual.positional_embedding[1:].detach().clone(),
        )

        clip_model_temp = load_clip_to_cpu(cfg, zero_shot_model=True).float()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg)
        if torch.cuda.is_available():
            clip_model_temp = clip_model_temp.cuda()
            clip_model_temp_image = clip_model_temp_image.cuda()

        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual

        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)

        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME)
        self.register_buffer("fixed_embeddings", clip_weights)

    def encode_patch_tokens(self, image):
        x = F.conv2d(
            image.type(self.image_patch_proj_weight.dtype),
            self.image_patch_proj_weight,
            bias=None,
            stride=self.patch_stride,
        )
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = x + self.image_patch_positional_embedding.to(x.dtype)
        return x

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        return torch.cat([prefix, ctx, suffix], dim=1)

    def construct_batch_prompts(self, ctx_batch):
        batch_size = ctx_batch.shape[0]
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ctx = ctx_batch.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=2)

    def construct_seed_prompts(self, ctx_batch):
        batch_size = ctx_batch.shape[0]
        prefix = self.seed_prefix.expand(batch_size, -1, -1)
        suffix = self.seed_suffix.expand(batch_size, -1, -1)
        return torch.cat([prefix, ctx_batch, suffix], dim=1)

    def repeat_tokenized_prompts(self, batch_size):
        return self.tokenized_prompts.unsqueeze(0).expand(batch_size, -1, -1).reshape(
            -1, self.tokenized_prompts.shape[-1]
        )

    def repeat_text_prompts(self, text_prompt_layers):
        repeated = []
        for prompt in text_prompt_layers:
            repeated.append(
                prompt.unsqueeze(1).expand(-1, self.n_cls, -1, -1).reshape(-1, self.n_ctx, prompt.shape[-1])
            )
        return repeated

    def _compute_shared_hicro_prompts(self):
        text_prompt_layers = [prompt for prompt in self.cross_prompts_text]
        visual_prompt_layers = [prompt for prompt in self.cross_prompts_visual]

        if self.cross_layer > 0:
            proxy_text_tokens = []
            for i in range(self.cross_layer):
                text_proxy_token = self.attn_pooling_text_nets[i](
                    token_query=self.text_proxy_token[i],
                    sequence_key=text_prompt_layers[i],
                    sequence_value=text_prompt_layers[i],
                )
                proxy_text_tokens.append(text_proxy_token)

            proxy_text_prompts = torch.cat(proxy_text_tokens, dim=0)
            visual_prompts = torch.stack(visual_prompt_layers[: self.cross_layer], dim=0).view(-1, visual_prompt_layers[0].shape[-1])
            proxy_text_prompts = proxy_text_prompts.view(-1, proxy_text_prompts.shape[-1])
            updated_visual_prompts = self.text2visual_net(visual_prompts, proxy_text_prompts, proxy_text_prompts)
            updated_visual_prompts = updated_visual_prompts.view(self.cross_layer, self.n_ctx, -1)
            for i in range(self.cross_layer):
                visual_prompt_layers[i] = updated_visual_prompts[i]

        tail_depth = self.cross_prompts_depth - self.cross_layer
        if tail_depth > 0:
            proxy_visual_tokens = []
            for i in range(self.cross_layer, self.cross_prompts_depth):
                visual_proxy_token = self.attn_pooling_visual_nets[i - self.cross_layer](
                    token_query=self.visual_proxy_token[i - self.cross_layer],
                    sequence_key=visual_prompt_layers[i],
                    sequence_value=visual_prompt_layers[i],
                )
                proxy_visual_tokens.append(visual_proxy_token)

            proxy_visual_prompts = torch.cat(proxy_visual_tokens, dim=0)
            text_prompts = torch.stack(text_prompt_layers[self.cross_layer :], dim=0).view(-1, text_prompt_layers[0].shape[-1])
            proxy_visual_prompts = proxy_visual_prompts.view(-1, proxy_visual_prompts.shape[-1])
            updated_text_prompts = self.visual2text_net(text_prompts, proxy_visual_prompts, proxy_visual_prompts)
            updated_text_prompts = updated_text_prompts.view(tail_depth, self.n_ctx, -1)
            for idx, i in enumerate(range(self.cross_layer, self.cross_prompts_depth)):
                text_prompt_layers[i] = updated_text_prompts[idx]

        return text_prompt_layers, visual_prompt_layers

    def forward(self, image):
        batch_size = image.shape[0]

        shared_text_prompts, shared_visual_prompts = self._compute_shared_hicro_prompts()

        patch_tokens = self.encode_patch_tokens(image)
        image_anchor = self.image_to_text_anchor(patch_tokens)

        shallow_text_ctx = shared_text_prompts[0].unsqueeze(0) + self.text_anchor_gates[0].type_as(image_anchor) * image_anchor
        seed_prompts = self.construct_seed_prompts(shallow_text_ctx)
        text_anchor = self.text_to_visual_anchor(seed_prompts)

        text_prompt_layers = []
        visual_prompt_layers = []
        for i in range(self.cross_prompts_depth):
            text_gate = self.text_anchor_gates[i].type_as(image_anchor)
            visual_gate = self.visual_anchor_gates[i].type_as(text_anchor)
            text_prompt_layers.append(shared_text_prompts[i].unsqueeze(0) + text_gate * image_anchor)
            visual_prompt_layers.append(shared_visual_prompts[i].unsqueeze(0) + visual_gate * text_anchor)

        text_input = self.construct_batch_prompts(text_prompt_layers[0])
        repeated_text_prompts = self.repeat_text_prompts(text_prompt_layers[1:])

        return (
            text_input,
            visual_prompt_layers[0],
            repeated_text_prompts,
            visual_prompt_layers[1:],
        )


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = HiCroSPLIPPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)
        self.lambd = cfg.TRAINER.HICROSPLIP.LAMBD

    def forward(self, image, label=None):
        batch_size = image.shape[0]
        tokenized_prompts = self.prompt_learner.repeat_tokenized_prompts(batch_size)
        logit_scale = self.logit_scale.exp()

        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner(image)
        text_features = self.text_encoder(
            text_input.reshape(-1, text_input.shape[2], text_input.shape[3]),
            tokenized_prompts,
            cross_prompts_text_deeper,
        )
        text_features = text_features.reshape(batch_size, self.n_cls, -1)

        image_features = self.image_encoder(image.type(self.dtype), visual_ctx, cross_prompts_visual_deeper)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed.type(self.dtype)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        fixed_text = self.prompt_learner.fixed_embeddings.type(self.dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + fixed_text.unsqueeze(0).expand(batch_size, -1, -1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = logit_scale * torch.einsum("bd,bcd->bc", image_features, text_features)

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)

            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)
            fixed_text_expanded = fixed_text.unsqueeze(0).expand(batch_size, -1, -1)
            loss_distill_text = 1.0 - torch.mean(
                cos(
                    text_features.reshape(-1, text_features.shape[-1]),
                    fixed_text_expanded.reshape(-1, fixed_text_expanded.shape[-1]),
                )
            )
            loss_distill_image = 1.0 - torch.mean(cos(image_features, image_features_fixed.type(self.dtype)))

            return loss_cls + self.lambd * (loss_distill_text + loss_distill_image)

        return logits


@TRAINER_REGISTRY.register()
class HiCroSPLIP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.HICROSPLIP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.HICROSPLIP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name and "ZS_image_encoder" not in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("HiCroSPLIP", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.HICROSPLIP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.HICROSPLIP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return image, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            for buffer_name in [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
                "prompt_learner.seed_prefix",
                "prompt_learner.seed_suffix",
            ]:
                if buffer_name in state_dict:
                    del state_dict[buffer_name]

            print('Loading weights to {} from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
