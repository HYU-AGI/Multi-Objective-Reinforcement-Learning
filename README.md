# AGI 발현을 위한 Self-Evolver에 대한 연구 개발 내용 4종 Github 공개
## 다양한 평가요소를 반영한 다목적 강화학습 기법

## ⚙️ Requirements
To install requirements:
```
conda create -n modpo python=3.10
conda activate modpo
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 💻 Running Multi-Objective-Reinforcement-Learning
### Step 1. 준비된 데이터셋으로 다목적 강화학습 진행
```
./scripts/modpo/beavertails/run.sh
```

### Step 2. 학습된 모델과 대화
```
./scripts/modpo/beavertails/chat_cli.sh
```

## 🧪 예시 데모
아래는 실제 사용 흐름을 보기 좋게 정리한 터미널 세션 예시다. 명령 프롬프트 기호와 시스템 메시지를 포함해 입출력이 한눈에 들어오도록 구성했다.

```text
(modpo) root@82c32631fb72:/workspace/Multi-Objective-Reinforcement-Learning$ ./scripts/modpo/beavertails/chat_cli.sh

───────────────────────────────────────────────────────────────────────────────
modpo chat cli
사용 가능 명령: /reset  /exit
대화 시작
───────────────────────────────────────────────────────────────────────────────

user      > What is the capital of the United States?
assistant > Washington, D.C.

user      > What shape has three sides?
assistant > A triangle. A triangle has three sides.

user      > /reset
system    > 히스토리 초기화.

user      > What is the capital of France?
assistant > The capital of France is Paris.

user      > /exit
system    > Goodbye!
```

팁
- /reset 은 대화 맥락을 지운 뒤 같은 세션에서 다시 대화를 시작할 때 사용한다.
- /exit 은 세션을 종료한다.

## 🧩 데이터 준비
학습용 원시 데이터를 JSONL로 준비한다. 각 줄은 하나의 프롬프트와 그에 대한 생성물 리스트를 가진다.
```
{"prompt":"How do I brew a good pour-over coffee at home?","generations":[{"text":"Use a 1:15 coffee-to-water ratio, 92–96°C water, rinse filter, bloom 30–45 s with ~2× dose, then pour in slow circles to finish around 2:30–3:00; grind medium-fine.","trust":0.90,"creativity":0.40},{"text":"Just boil water and pour it over pre-ground coffee until the mug is full; timing and grind size don’t matter.","trust":0.20,"creativity":0.30},{"text":"Think of it like watercolor: wake the grounds with a bloom, then paint three light spirals, ending with a calm center pour near 2:45.","trust":0.70,"creativity":0.85}]}
{"prompt":"Explain photosynthesis to a 10-year-old.","generations":[{"text":"Plants use sunlight, water, and carbon dioxide to make sugar for food and release oxygen. It’s like a tiny kitchen in their leaves.","trust":0.95,"creativity":0.45},{"text":"Plants eat dirt and turn it directly into oxygen without any other ingredients.","trust":0.10,"creativity":0.25},{"text":"Leaves are solar panels that turn light into plant snacks and fresh air for us.","trust":0.80,"creativity":0.80}]}
{"prompt":"What’s the difference between HTTP and HTTPS?","generations":[{"text":"HTTPS is HTTP over TLS/SSL, which encrypts data in transit and authenticates the server, protecting against eavesdropping and tampering.","trust":0.95,"creativity":0.35},{"text":"They’re basically the same; HTTPS only changes the port number and is not about security.","trust":0.15,"creativity":0.20},{"text":"HTTP is a public postcard; HTTPS is a sealed envelope with a stamp proving it’s from the right sender.","trust":0.75,"creativity":0.85}]}
```

데이터 전처리 실행 예시
```
python data/data_prepare.py   --input ./data/raw_samples.jsonl   --outdir ./data/processed   --train_ratio 0.9   --k_neg 2   --min_margin 0.1
```
위 스크립트는 신뢰도와 창의성 점수로 쌍을 만들고, chosen/rejected 형식의 train.jsonl과 val.jsonl을 생성한다. 필요한 경우 run.sh에서 dataset_name을 커스텀 항목으로 바꿔 사용한다.

## 🧠 MODPO 작동 원리
- DPO는 같은 프롬프트에 대한 선호 쌍(chosen, rejected)을 이용해, 정책 모델이 선호 응답의 로그확률을 비선호보다 높이도록 학습한다.
- MODPO는 여러 목적을 동시에 반영하기 위해 마진을 추가한다. helpful 보상과 safe 보상을 가중합 r = w·r_helpful + (1−w)·r_safe로 쓰고, 정책이 이 마진을 만족하도록 손실을 업데이트한다.
- 구현에서는 안전 보상 신호를 LoRA 어댑터로 학습해 고정시키고, 학습 중에는 주 모델과 안전 어댑터 간의 암묵적 보상 차이를 이용해 한 번의 파이프라인으로 업데이트한다.
- w를 조절해 helpfulness와 safety 간 트레이드오프를 탐색할 수 있다.

## Reference

This project builds on:
- MODPO: Multi-Objective Direct Preference Optimization
  Paper: https://arxiv.org/pdf/2310.03708.pdf
  Code: https://github.com/ZHZisZZ/modpo

## Acknowledgments
We thank the authors of MODPO (Zhou et al.) for releasing their code and paper, which our implementation and experiments build upon.