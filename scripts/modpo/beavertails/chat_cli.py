#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys

# --gpus 선파싱: torch 로드 전에 CUDA_VISIBLE_DEVICES 설정
def _preparse_gpus(argv):
    for i, a in enumerate(argv):
        if a == "--gpus" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = argv[i + 1]  # 예: "5" 또는 "5,7"
            break
_preparse_gpus(sys.argv)

import argparse
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    from peft import PeftModel
except Exception as e:
        print("peft 패키지가 필요합니다. pip install peft", file=sys.stderr)
        raise


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model_name", type=str, required=True)
    p.add_argument("--adapter_path", type=str, default=None, help="학습된 LoRA 어댑터 경로 (예: ./output/.../best_checkpoint)")
    p.add_argument("--system", type=str, default=None, help="선택: system 메시지")
    p.add_argument("--prompt_template", type=str, default="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--no_repeat_ngram_size", type=int, default=0)

    # 디바이스/샤딩
    p.add_argument("--gpus", type=str, default=None, help="사용할 GPU ID 목록. 예: '5' 또는 '5,7' (CUDA_VISIBLE_DEVICES 설정)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--device_map", type=str, default="auto", choices=["auto", "none"], help="auto면 여러 GPU가 보일 때 모델 자동 샤딩")

    # 실행 옵션
    p.add_argument("--stream", action="store_true")
    p.add_argument("--history", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def select_dtype():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_model_and_tokenizer(args):
    torch.manual_seed(args.seed)
    dtype = select_dtype()
    use_cuda = (args.device in ["auto", "cuda"]) and torch.cuda.is_available()
    device_map = "auto" if (use_cuda and args.device_map == "auto") else None

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    tok = AutoTokenizer.from_pretrained(args.sft_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)  # 어댑터명이 기본 "default"면 자동 활성화

    model.eval()
    return model, tok


def build_messages(history_pairs, user_text, system_text=None):
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    for u, a in history_pairs:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def build_prompt(tok, history_pairs, user_text, fallback_template, system_text=None):
    # tokenizer가 chat 템플릿을 지원하면 그것을 우선 사용
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = build_messages(history_pairs, user_text, system_text)
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 아니면 과거 포맷으로 구성
    if not history_pairs:
        return fallback_template.format(raw_prompt=user_text)
    convo = "BEGINNING OF CONVERSATION:"
    for (u, a) in history_pairs:
        convo += f" USER: {u} ASSISTANT: {a}"
    convo += f" USER: {user_text} ASSISTANT:"
    return convo


class StopOnTokens(StoppingCriteria):
    def __init__(self, tok, stop_strs=None, extra_eos_tokens=None):
        self.tok = tok
        self.stop_ids_list = []
        stop_strs = stop_strs or []
        for s in stop_strs:
            ids = tok(s, add_special_tokens=False).input_ids
            if ids:
                self.stop_ids_list.append(ids)
        # 단일 토큰 eos 후보들
        self.eos_single_ids = set()
        if extra_eos_tokens:
            for t in extra_eos_tokens:
                try:
                    tid = tok.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid != tok.unk_token_id and tid != -1:
                        self.eos_single_ids.add(tid)
                except Exception:
                    pass
        if tok.eos_token_id is not None:
            self.eos_single_ids.add(tok.eos_token_id)

    def __call__(self, input_ids, scores, **kwargs):
        # 마지막 토큰이 eos 후보면 정지
        last_id = input_ids[0, -1].item()
        if last_id in self.eos_single_ids:
            return True
        # 시퀀스 매칭
        seq = input_ids[0].tolist()
        for s in self.stop_ids_list:
            L = len(s)
            if L <= len(seq) and seq[-L:] == s:
                return True
        return False


def decode_new_only(tok, full_ids, prompt_len):
    gen_ids = full_ids[0][prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True)


def generate_once(model, tok, prompt, args):
    inputs = tok(prompt, return_tensors="pt")
    use_cuda = (args.device in ["auto", "cuda"]) and torch.cuda.is_available()
    if use_cuda:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # 다음 사용자 턴/역할 마커로 멈추도록 후보 문자열을 준비
    stop_strs = [
        "\nUSER:", " USER:", "\nUser:", "User:",
        "### Human:", "\n\nHuman:",
        "<|user|>", "<|prompter|>",
    ]
    # 모델별 추가 eos 토큰 후보
    extra_eos_tokens = ["<|eot_id|>", "<|im_end|>", "<|eos|>", "<|end_of_text|>"]
    stops = StoppingCriteriaList([StopOnTokens(tok, stop_strs, extra_eos_tokens)])

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size if args.no_repeat_ngram_size > 0 else None,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        stopping_criteria=stops,
    )
    # None 값 제거
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    if args.stream:
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        t = Thread(target=model.generate, kwargs={**inputs, **gen_kwargs})
        t.start()
        out_text = ""
        for piece in streamer:
            # 안전장치: 스트리밍 중 stop_strs가 포함되면 중단
            if any(s in piece for s in ["\nUSER:", " USER:", "### Human:", "<|user|>", "<|prompter|>"]):
                break
            sys.stdout.write(piece); sys.stdout.flush()
            out_text += piece
        sys.stdout.write("\n"); sys.stdout.flush()
        t.join()
        # 출력 후 마커가 남아있으면 잘라냄
        for s in ["\nUSER:", " USER:", "### Human:", "<|user|>", "<|prompter|>"]:
            if s in out_text:
                out_text = out_text.split(s)[0]
        return out_text.strip()
    else:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        # 프롬프트 길이만큼 슬라이스 후 디코드
        prompt_len = inputs["input_ids"].shape[1]
        decoded = decode_new_only(tok, out, prompt_len)
        for s in ["\nUSER:", " USER:", "### Human:", "<|user|>", "<|prompter|>"]:
            if s in decoded:
                decoded = decoded.split(s)[0]
        return decoded.strip()


def main():
    args = build_parser().parse_args()
    model, tok = load_model_and_tokenizer(args)

    history_pairs = []
    print("대화 시작. /reset, /exit 사용 가능.")
    while True:
        try:
            user = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"/exit", "/quit"}:
            print("Goodbye!")
            break
        if user.lower() == "/reset":
            history_pairs = []
            print("히스토리 초기화.")
            continue

        prompt = build_prompt(tok, history_pairs if args.history else [], user, args.prompt_template, args.system)
        # 스트리밍이면 프리픽스만 먼저 찍고, generate_once 안에서 토큰을 실시간 출력
        if args.stream:
            print("ASSISTANT> ", end="", flush=True)

        resp = generate_once(model, tok, prompt, args)

        # 스트리밍이 아닐 때만 한 줄로 출력
        if not args.stream:
            print("ASSISTANT>", resp)

        if args.history:
            history_pairs.append((user, resp))


if __name__ == "__main__":
    main()
