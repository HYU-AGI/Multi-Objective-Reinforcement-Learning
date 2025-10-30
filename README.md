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

## Reference

This project builds on:
- MODPO: Multi-Objective Direct Preference Optimization
  Paper: https://arxiv.org/pdf/2310.03708.pdf
  Code: https://github.com/ZHZisZZ/modpo

## Acknowledgments
We thank the authors of MODPO (Zhou et al.) for releasing their code and paper, which our implementation and experiments build upon.