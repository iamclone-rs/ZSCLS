import json
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.model import QuickGELU
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()


CoPrompt_dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}


def load_clip_to_cpu_teacher(cfg):
    backbone_name = cfg.TRAINER.SPLIPDISTILL.TEACHER_NAME
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


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Load a prompt-free CLIP and implement SpLiP-style sharing in this trainer.
    design_details = {
        "trainer": "IVLP",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


class VisualToTextMapper(nn.Module):
    def __init__(self, vision_dim, text_dim, n_tokens, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(n_tokens, vision_dim) * 0.02)
        self.ln_q = nn.LayerNorm(vision_dim)
        self.ln_kv = nn.LayerNorm(vision_dim)
        self.attn = nn.MultiheadAttention(vision_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            QuickGELU(),
            nn.Linear(vision_dim, vision_dim),
            QuickGELU(),
            nn.Linear(vision_dim, text_dim),
        )
        self.ln_out = nn.LayerNorm(text_dim)

    def forward(self, patch_tokens):
        batch_size = patch_tokens.shape[0]
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled = self.attn(
            self.ln_q(query),
            self.ln_kv(patch_tokens),
            self.ln_kv(patch_tokens),
            need_weights=False,
        )[0]
        tokens = query + pooled
        tokens = self.mlp(tokens)
        return self.ln_out(tokens)


class SemanticToVisualMapper(nn.Module):
    def __init__(self, text_dim, vision_dim):
        super().__init__()
        self.proj = nn.Linear(text_dim, vision_dim)
        self.ln = nn.LayerNorm(vision_dim)

    def forward(self, text_tokens):
        return self.ln(self.proj(text_tokens))


class TextToVisualMapper(nn.Module):
    def __init__(self, text_dim, vision_dim, n_tokens, bottleneck_dim=256, num_heads=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(n_tokens, text_dim) * 0.02)
        self.ln_q = nn.LayerNorm(text_dim)
        self.ln_kv = nn.LayerNorm(text_dim)
        self.attn = nn.MultiheadAttention(text_dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, vision_dim),
        )
        self.ln_out = nn.LayerNorm(vision_dim)

    def forward(self, text_tokens):
        batch_size = text_tokens.shape[0]
        query = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled = self.attn(
            self.ln_q(query),
            self.ln_kv(text_tokens),
            self.ln_kv(text_tokens),
            need_weights=False,
        )[0]
        tokens = query + pooled
        tokens = self.mlp(tokens)
        return self.ln_out(tokens)


class SpLiPTextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx_text, prompt_depth):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_ctx_text = n_ctx_text
        self.prompt_depth = min(prompt_depth, len(self.transformer.resblocks))

    def _replace_text_prompts(self, x, prompt_tokens):
        prefix = x[:1, :, :]
        suffix = x[1 + self.n_ctx_text :, :, :]
        prompt_tokens = prompt_tokens.permute(1, 0, 2)
        return torch.cat([prefix, prompt_tokens, suffix], dim=0)

    def forward(self, prompts, tokenized_prompts, prompt_tokens):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        hidden_states = []
        for i, block in enumerate(self.transformer.resblocks):
            if 0 < i < self.prompt_depth:
                x = self._replace_text_prompts(x, prompt_tokens)
            x = block(x)
            hidden_states.append(x.permute(1, 0, 2))

        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x, hidden_states


class SpLiPVisionEncoder(nn.Module):
    def __init__(self, clip_visual, prompt_depth, visual_prompt_len):
        super().__init__()
        self.conv1 = clip_visual.conv1
        self.class_embedding = clip_visual.class_embedding
        self.positional_embedding = clip_visual.positional_embedding
        self.ln_pre = clip_visual.ln_pre
        self.transformer = clip_visual.transformer
        self.ln_post = clip_visual.ln_post
        self.proj = clip_visual.proj
        self.dtype = clip_visual.conv1.weight.dtype
        self.prompt_depth = min(prompt_depth, len(self.transformer.resblocks))
        self.visual_prompt_len = visual_prompt_len

    def encode_patch_tokens(self, image):
        x = self.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding[1:].to(x.dtype)
        return x

    def _replace_visual_prompts(self, x, prompt_tokens):
        prefix = x[:1, :, :]
        suffix = x[1 + self.visual_prompt_len :, :, :]
        prompt_tokens = prompt_tokens.permute(1, 0, 2)
        return torch.cat([prefix, prompt_tokens, suffix], dim=0)

    def forward_with_patches(self, patch_tokens, visual_prompts):
        batch_size = patch_tokens.shape[0]
        cls_token = self.class_embedding.to(patch_tokens.dtype)
        cls_token = cls_token + self.positional_embedding[:1].to(patch_tokens.dtype).squeeze(0)
        cls_token = cls_token.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)

        x = torch.cat([cls_token, visual_prompts[0].type(patch_tokens.dtype), patch_tokens], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        for i, block in enumerate(self.transformer.resblocks):
            if 0 < i < len(visual_prompts) and i < self.prompt_depth:
                x = self._replace_visual_prompts(x, visual_prompts[i])
            x = block(x)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        return x


class SpLiPDistillPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx_text = cfg.TRAINER.SPLIPDISTILL.N_CTX_TEXT
        self.n_ctx_vision = cfg.TRAINER.SPLIPDISTILL.N_CTX_VISION
        self.prompt_depth = cfg.TRAINER.SPLIPDISTILL.PROMPT_DEPTH
        self.dtype = clip_model.dtype

        text_dim = clip_model.ln_final.weight.shape[0]
        vision_dim = clip_model.visual.conv1.weight.shape[0]

        ctx_init = cfg.TRAINER.SPLIPDISTILL.CTX_INIT.replace("_", " ").strip()
        ctx_words = ctx_init.split(" ") if ctx_init else []
        if ctx_init and len(ctx_words) != self.n_ctx_text:
            raise ValueError(
                f"SPLIPDISTILL expects N_CTX_TEXT={self.n_ctx_text} to match "
                f"the number of words in CTX_INIT ({len(ctx_words)})."
            )

        if ctx_init:
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            semantic_ctx = embedding[0, 1 : 1 + self.n_ctx_text, :]
            prompt_prefix = ctx_init
        else:
            semantic_ctx = torch.empty(self.n_ctx_text, text_dim, dtype=self.dtype)
            nn.init.normal_(semantic_ctx, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx_text)

        print("SpLiPDistill design: SpLiP-style prompt sharing with teacher distillation")
        print(f'Initial text template: "{prompt_prefix}"')
        print(f"Number of text prompt tokens: {self.n_ctx_text}")
        print(f"Number of text-guided visual tokens: {self.n_ctx_vision}")

        self.register_buffer("semantic_ctx", semantic_ctx)

        self.bt = VisualToTextMapper(vision_dim, text_dim, self.n_ctx_text)
        self.bv = SemanticToVisualMapper(text_dim, vision_dim)
        self.bvt = TextToVisualMapper(text_dim, vision_dim, self.n_ctx_vision)

        if self.dtype == torch.float16:
            self.bt.half()
            self.bv.half()
            self.bvt.half()

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx_text :, :])
        self.tokenized_prompts = tokenized_prompts

        clip_model_temp = load_clip_to_cpu(cfg).float()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg)
        if torch.cuda.is_available():
            clip_model_temp = clip_model_temp.cuda()
            clip_model_temp_image = clip_model_temp_image.cuda()

        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual

        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)

        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME
        )
        self.register_buffer("fixed_embeddings", clip_weights)

    @property
    def visual_prompt_len(self):
        return self.semantic_ctx.shape[0] + self.n_ctx_vision

    def construct_prompts(self, text_ctx):
        batch_size = text_ctx.shape[0]
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ctx = text_ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)
        prompts = torch.cat([prefix, ctx, suffix], dim=2)
        return prompts

    def repeat_text_ctx(self, text_ctx):
        return text_ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1).reshape(
            -1, self.n_ctx_text, text_ctx.shape[-1]
        )

    def repeat_tokenized_prompts(self, batch_size):
        return self.tokenized_prompts.unsqueeze(0).expand(batch_size, -1, -1).reshape(
            -1, self.tokenized_prompts.shape[-1]
        )

    def build_visual_prompts(self, text_hidden_states, batch_size):
        semantic_visual = self.bv(self.semantic_ctx).unsqueeze(0).expand(batch_size, -1, -1)
        visual_prompts = []

        for layer_tokens in text_hidden_states[: self.prompt_depth]:
            layer_tokens = layer_tokens.reshape(batch_size, self.n_cls, layer_tokens.shape[1], layer_tokens.shape[2])
            layer_tokens = layer_tokens.reshape(batch_size, -1, layer_tokens.shape[-1])
            shared_visual = self.bvt(layer_tokens)
            visual_prompts.append(torch.cat([semantic_visual, shared_visual], dim=1))

        return visual_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = SpLiPDistillPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = SpLiPTextEncoder(
            clip_model,
            n_ctx_text=cfg.TRAINER.SPLIPDISTILL.N_CTX_TEXT,
            prompt_depth=cfg.TRAINER.SPLIPDISTILL.PROMPT_DEPTH,
        )
        self.image_encoder = SpLiPVisionEncoder(
            clip_model.visual,
            prompt_depth=cfg.TRAINER.SPLIPDISTILL.PROMPT_DEPTH,
            visual_prompt_len=self.prompt_learner.visual_prompt_len,
        )
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = len(classnames)
        self.lambd = cfg.TRAINER.SPLIPDISTILL.LAMBD

    def forward(self, image, label=None):
        batch_size = image.shape[0]
        image = image.type(self.dtype)

        patch_tokens = self.image_encoder.encode_patch_tokens(image)
        text_ctx = self.prompt_learner.bt(patch_tokens)
        text_prompts = self.prompt_learner.construct_prompts(text_ctx)
        repeated_text_ctx = self.prompt_learner.repeat_text_ctx(text_ctx)
        repeated_tokenized_prompts = self.prompt_learner.repeat_tokenized_prompts(batch_size)

        text_features, text_hidden_states = self.text_encoder(
            text_prompts.reshape(-1, text_prompts.shape[2], text_prompts.shape[3]),
            repeated_tokenized_prompts,
            repeated_text_ctx,
        )
        text_features = text_features.reshape(batch_size, self.n_cls, -1)

        visual_prompts = self.prompt_learner.build_visual_prompts(text_hidden_states, batch_size)
        image_features = self.image_encoder.forward_with_patches(patch_tokens, visual_prompts)

        with torch.no_grad():
            image_features_fixed = self.prompt_learner.ZS_image_encoder(image)
            image_features_fixed = image_features_fixed / image_features_fixed.norm(dim=-1, keepdim=True)

        fixed_text = self.prompt_learner.fixed_embeddings.type(self.dtype)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features + image_features_fixed.type(self.dtype)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features + fixed_text.unsqueeze(0).expand(batch_size, -1, -1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = self.logit_scale.exp() * torch.einsum("bd,bcd->bc", image_features, text_features)

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)

            fixed_text_expanded = fixed_text.unsqueeze(0).expand(batch_size, -1, -1)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)
            loss_distill_text = 1.0 - torch.mean(
                cos(
                    text_features.reshape(-1, text_features.shape[-1]),
                    fixed_text_expanded.reshape(-1, fixed_text_expanded.shape[-1]),
                )
            )
            loss_distill_image = 1.0 - torch.mean(cos(image_features, image_features_fixed.type(self.dtype)))

            return loss_cls + self.lambd * (loss_distill_text + loss_distill_image)

        return logits


def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os

    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t for t in gpt_prompts[classname]]
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=0)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_splipdistill.pt")

    return clip_weights


@TRAINER_REGISTRY.register()
class SpLiPDistill(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.SPLIPDISTILL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SPLIPDISTILL.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients except SpLiP blocks and CLIP LayerNorms")
        for name, param in self.model.named_parameters():
            trainable = False
            if "ZS_image_encoder" in name:
                trainable = False
            elif "prompt_learner" in name:
                trainable = True
            elif any(tag in name for tag in ["ln_1", "ln_2", "ln_pre", "ln_post", "ln_final"]):
                trainable = True
            param.requires_grad_(trainable)

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
        self.register_model("SpLiPDistill", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SPLIPDISTILL.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.SPLIPDISTILL.PREC
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
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            for key in [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
                "prompt_learner.semantic_ctx",
                "prompt_learner.fixed_embeddings",
            ]:
                if key in state_dict:
                    del state_dict[key]

            print(f'Loading weights to {name} from "{model_path}" (epoch = {epoch})')
            self._models[name].load_state_dict(state_dict, strict=False)
