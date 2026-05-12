import os
import sys
import json
import torch
import torch.nn as nn
import pytorch_lightning as pl
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    SwinModel,
)

from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor

from peft import get_peft_model, LoraConfig, TaskType


# =========================================================
# HF TOKEN
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN no encontrado. "
        "Asegúrate de exportarlo correctamente."
    )
else:
    print("Token de Hugging Face cargado correctamente.")


# =========================================================
# DEBUG INFO
# =========================================================

print("PYTHON  :", sys.executable)
print("PATH    :", ":".join(sys.path)[:200], "…")
print("TRANSFORMERS", transformers.__version__, transformers.__file__)


# =========================================================
# MODEL
# =========================================================

class R2GenGPT(pl.LightningModule):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.save_hyperparameters(args)

        # =====================================================
        # VISION ENCODER
        # =====================================================

        print(f"Loading vision encoder: {args.vision_model}")

        self.visual_encoder = SwinModel.from_pretrained(
            args.vision_model
        )

        if args.vis_use_lora:

            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )

            self.visual_encoder = get_peft_model(
                self.visual_encoder,
                peft_config_visual
            )

            self.visual_encoder.print_trainable_parameters()

            print("Loading vision encoder with LoRA -- Done")

        elif args.freeze_vm:

            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False

            print(
                f"Loading Frozen vision encoder: "
                f"{args.vision_model} -- Done"
            )

        else:

            print(
                f"Loading Trainable vision encoder: "
                f"{args.vision_model} -- Done"
            )

        # =====================================================
        # LLAMA TOKENIZER
        # =====================================================

        print("Loading LLAMA tokenizer")

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            args.llama_model,
            trust_remote_code=True,
            token=HF_TOKEN,
            use_fast=False,
        )

        self.llama_tokenizer.pad_token_id = 0

        # =====================================================
        # QUANTIZATION CONFIG
        # =====================================================

        quant_config = None

        if getattr(args, "load_in_4bit", False):

            from transformers import BitsAndBytesConfig

            cdtype = getattr(
                torch,
                args.bnb_4bit_compute_dtype,
                None,
            )

            if cdtype is None:
                cdtype = torch.float16

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=cdtype,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            )

            print(f"Using 4-bit quantization: {quant_config}")

        # =====================================================
        # LLAMA MODEL
        # =====================================================

        print("Loading LLAMA model")

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=HF_TOKEN,
            quantization_config=quant_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        # =====================================================
        # LORA LLM
        # =====================================================

        self.embed_tokens = (
            self.llama_model.get_input_embeddings()
        )

        if args.llm_use_lora:

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
            )

            self.llama_model = get_peft_model(
                self.llama_model,
                peft_config,
            )

            self.llama_model.print_trainable_parameters()

            print("Loading LLAMA LoRA Done")

        else:

            for _, param in self.llama_model.named_parameters():
                param.requires_grad = False

            print("Loading LLAMA Done")

        # =====================================================
        # PROJECTION
        # =====================================================

        self.llama_proj = nn.Linear(
            self.visual_encoder.num_features,
            self.llama_model.config.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(
            self.llama_model.config.hidden_size
        )

        self.end_sym = args.end_sym

        self.prompt = (
            "Generate a comprehensive and detailed "
            "diagnosis report for this chest xray image."
        )

        self.val_step_outputs = []
        self.test_step_outputs = []

        self.val_score = 0.0

        # =====================================================
        # LOAD CHECKPOINT
        # =====================================================

        if args.delta_file is not None:

            state_dict = torch.load(
                args.delta_file,
                map_location=torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                ),
            )["model"]

            self.load_state_dict(
                state_dict=state_dict,
                strict=False,
            )

            print(f"Load checkpoint from {args.delta_file}")

    # =========================================================
    # METRICS
    # =========================================================

    def score(self, ref, hypo):

        scorers = [
            (
                Bleu(4),
                ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"],
            ),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]

        try:
            scorers.append((Meteor(), "METEOR"))

        except Exception as e:
            print(f"WARNING: METEOR disabled ({e})")

        final_scores = {}

        for scorer, method in scorers:

            score, _ = scorer.compute_score(ref, hypo)

            if isinstance(score, list):

                for m, s in zip(method, score):
                    final_scores[m] = s

            else:
                final_scores[method] = score

        return final_scores

    # =========================================================
    # IMAGE ENCODING
    # =========================================================

    def encode_img(self, images):

        image_embeds = []

        for image in images:

            device = image.device

            if self.hparams.global_only:

                image_embed = (
                    self.visual_encoder(image)["pooler_output"]
                    .unsqueeze(1)
                    .to(device)
                )

            else:

                image_embed = (
                    self.visual_encoder(image)["last_hidden_state"]
                    .to(device)
                )

            image_embeds.append(image_embed)

        image_embeds = torch.stack(image_embeds).mean(0)

        inputs_llama = self.llama_proj(image_embeds)

        atts_llama = torch.ones(
            inputs_llama.size()[:-1],
            dtype=torch.long,
            device=image.device,
        )

        return inputs_llama, atts_llama

    # =========================================================
    # PROMPT WRAP
    # =========================================================

    def prompt_wrap(self, img_embeds, atts_img):

        prompt = (
            "Human: <Img><ImageHere></Img> "
            f"{self.prompt} \nAssistant:"
        )

        batch_size = img_embeds.shape[0]

        p_before, p_after = prompt.split("<ImageHere>")

        p_before_tokens = self.llama_tokenizer(
            p_before,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(
            p_after,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(img_embeds.device)

        p_before_embeds = self.embed_tokens(
            p_before_tokens.input_ids
        ).expand(batch_size, -1, -1)

        p_after_embeds = self.embed_tokens(
            p_after_tokens.input_ids
        ).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat(
            [p_before_embeds, img_embeds, p_after_embeds],
            dim=1,
        )

        wrapped_atts_img = atts_img[:, :1].expand(
            -1,
            wrapped_img_embeds.shape[1],
        )

        return wrapped_img_embeds, wrapped_atts_img

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(self, samples):

        image = samples["image"]

        img_embeds, atts_img = self.encode_img(image)

        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(
            img_embeds,
            atts_img,
        )

        self.llama_tokenizer.padding_side = "right"

        text = [
            t + self.end_sym
            for t in samples["input_text"]
        ]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0,
            -100,
        )

        empty_targets = (
            torch.ones(
                [atts_img.shape[0], atts_img.shape[1] + 1],
                dtype=torch.long,
            )
            .to(image[0].device)
            .fill_(-100)
        )

        targets = torch.cat(
            [empty_targets, targets],
            dim=1,
        )

        batch_size = img_embeds.shape[0]

        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.embed_tokens(bos.long())

        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(
            to_regress_tokens.input_ids
        )

        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds, to_regress_embeds],
            dim=1,
        )

        attention_mask = torch.cat(
            [
                atts_bos,
                atts_img,
                to_regress_tokens.attention_mask,
            ],
            dim=1,
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        loss = outputs.loss

        return {"loss": loss}

    # =========================================================
    # TRAINING
    # =========================================================

    def training_step(self, batch, batch_idx):

        result = self(batch)

        self.log_dict(
            result,
            prog_bar=True,
            sync_dist=True,
        )

        return result["loss"]

    # =========================================================
    # VALIDATION
    # =========================================================

    def validation_step(self, samples, batch_idx):

        self.llama_tokenizer.padding_side = "right"

        to_regress_tokens = self.llama_tokenizer(
            samples["input_text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        )

        image = samples["image"]

        img_embeds, atts_img = self.encode_img(image)

        img_embeds = self.layer_norm(img_embeds)

        img_embeds, atts_img = self.prompt_wrap(
            img_embeds,
            atts_img,
        )

        batch_size = img_embeds.shape[0]

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.long,
            device=atts_img.device,
        ) * self.llama_tokenizer.bos_token_id

        bos_embeds = self.embed_tokens(bos)

        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat(
            [bos_embeds, img_embeds],
            dim=1,
        )

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )

        hypo = [self.decode(i) for i in outputs]

        ref = [
            self.decode(i)
            for i in to_regress_tokens["input_ids"]
        ]

        self.val_step_outputs.append(
            {
                "hypo": hypo,
                "ref": ref,
                "id": samples["id"],
            }
        )

        return hypo, ref

    # =========================================================
    # DECODE
    # =========================================================

    def decode(self, output_token):

        if output_token[0] == 0:
            output_token = output_token[1:]

        if output_token[0] == 1:
            output_token = output_token[1:]

        output_text = self.llama_tokenizer.decode(
            output_token,
            add_special_tokens=False,
        )

        output_text = output_text.split("</s>")[0].strip()

        output_text = output_text.replace("<unk>", "")

        return output_text

    # =========================================================
    # VALIDATION END
    # =========================================================

    def on_validation_epoch_end(self):

        ref, hypo, ids = [], [], []

        for i in self.val_step_outputs:

            ref.extend(i["ref"])
            hypo.extend(i["hypo"])
            ids.extend(i["id"])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}

        eval_res = self.score(ref=ref, hypo=hypo)

        self.log_dict(
            eval_res,
            sync_dist=True,
            logger=True,
        )

        self.val_step_outputs.clear()

    # =========================================================
    # OPTIMIZER
    # =========================================================

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=1e-6,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    # =========================================================
    # PROGRESS BAR
    # =========================================================

    def get_progress_bar_dict(self):

        items = super().get_progress_bar_dict()

        items.pop("v_num", None)

        return items

    # =========================================================
    # ZERO GRAD
    # =========================================================

    def optimizer_zero_grad(
        self,
        epoch,
        batch_idx,
        optimizer,
    ):
        optimizer.zero_grad()