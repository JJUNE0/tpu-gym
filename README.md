# tpu-gym

CPU·GPU·Google TPU에서 실행하는 범용 JAX 강화학습 실험 프로젝트입니다. 태스크는 `tasks/<분야>/<대상>/`에 추가하며, 현재 검증된 예제는 G1 평지 보행입니다.

`g1-flat`은 MuJoCo Playground의 `G1JoystickFlatTerrain`을 MJX-JAX로 실행하는 평지 속도 추종 태스크입니다. Brax PPO와 FlashCHORD의 JAX FlashSAC을 제공합니다. PPO는 Playground의 기본 설정을 사용하고, FlashSAC은 원본 벤치마크 스크립트에 따라 외부 밀기를 끕니다. 따라서 현재 기본 실행끼리는 환경 조건이 같지 않습니다. G1에서 설치와 학습을 확인한 뒤 `tasks/locomotion/humanoid_light/`에 Humanoid Light를 추가할 수 있습니다.

새 태스크는 `train.py`에 등록하고 `make_environment(algorithm=...)`, `ppo_defaults()`, `flashsac_defaults()`를 제공합니다. 환경의 `state`는 정책 관측이고 `privileged_state`는 여기에 critic 전용 관측을 이어 붙인 배열입니다.

```text
tpu-gym/
├── pyproject.toml / uv.lock       # 독립 환경과 버전 잠금
├── train.py                      # checkout에서 실행하는 진입점
├── src/tpu_gym/
│   ├── train.py                  # 인자, 장치 선택, 태스크 등록
│   ├── algorithm/                # PPO, FlashSAC; SAC++는 아직 미구현
│   └── tasks/locomotion/g1/flat.py
├── third_party/                  # 커밋을 고정한 Git submodule
├── patches/flash_chord-jax.patch  # 학습기를 Newton 없이 사용하는 수정
├── scripts/bootstrap.py          # submodule·패치·모델 준비
├── tests/jax/                    # 학습기 연결 회귀 테스트
└── outputs/                      # 로컬 로그와 체크포인트, Git 제외
```

## 소스 준비

```bash
git clone git@github.com:JJUNE0/tpu-gym.git
cd tpu-gym
python3 scripts/bootstrap.py --assets unitree_g1
```

`bootstrap.py`가 submodule을 초기화하고 FlashCHORD 패치를 적용합니다. 반복 실행할 수 있으며, 이미 수정된 파일과 충돌하면 덮어쓰지 않고 실패합니다. FlashCHORD는 NVIDIA `video_to_data` 저장소의 `robotic_grounding/flash_chord`만 sparse checkout하고 LFS 데이터는 받지 않습니다. 단순 `git submodule update` 뒤에도 패치는 별도로 적용해야 하므로 이 스크립트를 사용합니다.

| Submodule | 고정 버전 | 용도 |
|---|---|---|
| `mujoco_playground` | `v0.2.0` / `124a73f` | 실제 사용하는 MJX 환경, editable 설치 |
| `flash_chord` | `5226e34` | JAX FlashSAC 학습기, 로컬 패치 적용 |
| `FlashSAC` | `87edc90` | 원본 설정·공개 CSV·수치 비교의 기준 |
| `sacpp` | `172c117` | SAC++ PyTorch 참조 구현, 아직 JAX 학습기에 미등록 |

`sacpp`는 비공개 저장소이므로 clone하는 계정에 접근 권한이 필요합니다. 공개 프로젝트에는 submodule 주소와 커밋만 등록하며 SAC++ 소스 자체를 복사하지 않습니다. Brax와 JAX 등은 `uv.lock`으로 관리합니다. FlashCHORD 전체 패키지의 Newton·CUDA 의존성과 SAC++의 PyTorch 의존성은 이 프로젝트의 JAX 환경에 설치하지 않습니다.

SAC++ 접근 권한이 없는 사용자는 `python3 scripts/bootstrap.py --skip-sacpp --assets unitree_g1`로 PPO와 FlashSAC을 준비할 수 있습니다.

새 Menagerie 모델은 `python3 scripts/bootstrap.py --assets unitree_go2 franka_emika_panda`처럼 추가할 수 있습니다. 위 G1 준비 명령은 해당 모델만 받으므로 다른 로봇 태스크를 추가할 때 필요한 모델 폴더도 추가합니다. 태스크의 시뮬레이션은 `impl="jax"`를 사용합니다. TPU 사용은 TPU VM 안에서 동일 코드를 실행하는 방식이며, 실제 TPU 하드웨어 검증은 아직 하지 않았습니다.

## 로컬 Linux x86-64

저장소 루트에서 실행합니다. `uv`가 Python 3.13과 `.venv`를 준비합니다. FlashCHORD는 전체 시뮬레이터의 의존성을 설치하지 않고, 패치한 JAX 학습기 소스만 불러옵니다. 학습에는 잠금 파일의 `playground==0.2.0` 패키지와 `third_party/flash_chord/robotic_grounding/flash_chord`의 학습기 소스를 사용합니다. 첫 G1 실행 때 Playground가 MuJoCo Menagerie에서 모델 에셋을 받으므로 Git과 네트워크 연결이 필요합니다.

```bash
python3 scripts/bootstrap.py --assets unitree_g1
uv sync --locked
uv run --locked python train.py \
  --task g1-flat --algorithm ppo --platform cpu --smoke
uv run --locked python train.py \
  --task g1-flat --algorithm flashsac --platform cpu --smoke
```

`--smoke`는 작은 네트워크와 배치로 실제 업데이트 및 체크포인트 저장을 확인합니다. NVIDIA GPU가 있는 로컬 머신에서는 `uv run`에 `--extra cuda`를, 학습 명령에 `--platform gpu`를 사용합니다. `--platform auto`는 현재 머신의 JAX 기본 장치를 사용합니다. 장치를 명시하면 해당 장치를 초기화할 수 없을 때 실행이 실패합니다. 실행 시작 시 선택된 장치가 출력됩니다.

첫 실행에는 MJX rollout과 학습 함수의 XLA 컴파일 시간이 포함됩니다.

같은 GPU에서 다른 학습도 실행 중이라면 아래처럼 JAX 메모리 선점을 끄고 allocator의 메모리 비율을 설정할 수 있습니다. 환경 8,192개, 기본 PPO 네트워크·batch·minibatch를 사용한 로컬 검증에서도 이 값을 사용했습니다.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.20 \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm ppo --platform gpu \
  --num-envs 8192 --steps 200000000 --num-evals 20
```

## Google Cloud TPU VM

`--platform tpu`는 현재 머신에 연결된 TPU를 선택합니다. 먼저 준비된 TPU VM에 SSH로 접속하고, 그 VM 안에서 소스를 clone한 뒤 아래 명령을 실행합니다. VM 생성과 가용 리전은 [Google Cloud TPU VM 가이드](https://docs.cloud.google.com/tpu/docs/quickstart-create-tpu-instance)를 따릅니다.

```bash
PROJECT_ID=your-project-id
ZONE=your-tpu-zone
VM=your-tpu-vm
gcloud compute ssh "$VM" --project="$PROJECT_ID" --zone="$ZONE"
```

VM에 이 저장소를 clone하고 `python3 scripts/bootstrap.py --assets unitree_g1`로 submodule과 패치를 준비한 뒤, `uv`를 설치합니다. VM의 저장소 루트에서 실행합니다. [uv 설치 문서](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked --extra tpu
uv run --locked --extra tpu python train.py \
  --task g1-flat --algorithm ppo --platform tpu --smoke
uv run --locked --extra tpu python train.py \
  --task g1-flat --algorithm flashsac --platform tpu --smoke
```

두 명령이 완료되면 `--smoke`를 빼고 학습합니다. G1 PPO의 기본값은 Playground 설정을 따르는 환경 8,192개, 전체 2억 transition, 최대 에피소드 길이 1,000스텝(20초), batch size 256, minibatch 32개, 평가 20회입니다. 이 값은 전체 학습 예산이며, 중간 평가는 학습 추세를 확인하는 데 사용합니다.

`--num-envs`, `--steps`, `--batch-size`, `--seed`, `--output`으로 실험을 조절할 수 있습니다. `--steps`는 전체 환경의 transition 수입니다. PPO는 rollout 단위로 올림하고 FlashSAC은 `--num-envs`의 배수를 사용합니다. `--num-evals`는 평가 및 체크포인트 횟수를 설정하고, `metrics.jsonl`에는 각 평가의 보상·에피소드 길이·세부 보상 항목이 저장됩니다. FlashSAC에는 `--replay-capacity`와 `--replay-warmup`도 있습니다. 현재 FlashSAC은 단일 JAX 장치에서 실행합니다.

## FlashSAC 벤치마크 설정

이전 GPU 학습의 설정과 전체 평가 곡선은 [G1 seed 0 결과](benchmarks/g1-flat/flashsac-seed0.json)에 보관합니다. 원본 로그와 체크포인트는 로컬 `outputs/jax/`로 이전하며 Git에는 올리지 않습니다.

`third_party/FlashSAC/scripts/run_mujoco_playground.sh`의 override를 기준으로 환경 1,024개, 50,000,896 transition, γ=0.97, n-step=1, replay 10,000,000개, warmup 100,000개, batch 2,048개, collection당 업데이트 2회, actor 관측 103차원과 critic 관측 216차원을 사용합니다. 외부 밀기와 domain randomization은 끄고, 종료 시 초기 data/obs를 재사용하며 task info/RNG를 유지합니다. 평가는 50개 환경으로 진행합니다. GPU에서는 FP16 행렬 연산과 동적 loss scaling을 사용하고 CPU에서는 FP32를 사용합니다. 기본 seed는 PPO 42, FlashSAC 0입니다.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.90 \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm flashsac --platform gpu
```

1천만 replay에는 관측·다음 관측·action 등을 저장하므로 약 17.2 GiB가 필요하며, 모델·시뮬레이터·컴파일 임시 메모리는 별도입니다. GPU 공유 시 PPO용 메모리 비율 0.20을 그대로 적용할 수 없습니다. `--replay-capacity`를 줄일 수 있지만 그 실행은 공개 설정과 다른 실험입니다.

실행의 `config.json`에 실제 하이퍼파라미터, 환경 설정, 패키지 버전, 관측 차원, replay 메모리 예상치를 기록합니다. 원본은 Playground 0.0.5 / MuJoCo 3.3.3을 사용하고 이 환경은 0.2.0 / 3.11.0을 사용합니다. 원본 공개 CSV의 G1 평지 결과는 5천만 스텝에서 평균 약 39.94입니다. 설정 수정 후 seed 0 실행(`outputs/jax/flashsac_benchmark_fix/20261111-165331/flashsac`)은 50,000,896 transition에서 리턴 40.744, 평가 50개 환경의 평균 에피소드 길이 1,000스텝을 기록했습니다(총 45분 25초). 단일 seed와 시뮬레이터 버전 차이 때문에 엄밀한 논문 재현까지 검증한 결과는 아닙니다. 이전 γ=0.99 / n-step=3 / replay 100만 / 밀기 활성화 실행은 2억 스텝에서 26.35였으므로 재현 실패로 보존합니다.

원본 PyTorch 기준값과의 수치 비교 및 rollout 회귀 검증:

```bash
uv sync --locked --extra cuda --group dev
JAX_PLATFORMS=cpu PYTHONPATH=src:third_party/flash_chord/robotic_grounding/flash_chord/src \
  .venv/bin/python -m pytest -q tests/jax/test_flashsac_benchmark.py
```

기본 출력 경로는 `outputs/jax/g1-flat/<algorithm>/<timestamp>/`입니다. PPO는 Brax 체크포인트를 저장하고, FlashSAC은 평가 시점마다 정책과 학습 상태를 safetensors로 저장합니다. 두 학습기 모두 `metrics.jsonl`에 평가 결과를 기록합니다. VM에 `~/tpu-gym`로 배치했다면 로컬 터미널에서 결과를 복사할 수 있습니다.

```bash
gcloud compute scp --recurse "$VM:~/tpu-gym/outputs/jax" ./tpu-jax-outputs \
  --project="$PROJECT_ID" --zone="$ZONE"
```

## SAC++ 참조 소스와 JAX 이식 범위

`third_party/sacpp`에는 [nerdbamboo/sacpp](https://github.com/nerdbamboo/sacpp)의 비공개 원본 저장소를 submodule로 등록했습니다. 현재는 PyTorch + IsaacLab 참조 구현이며, `train.py --algorithm`에 SAC++를 등록하거나 JAX 학습기를 구현하지 않았습니다. 설정과 알고리즘의 상세 내용은 접근 권한이 있는 계정에서 해당 저장소의 문서와 코드를 확인합니다.

후속 구현은 `src/tpu_gym/algorithm/sacpp/`에 두고 기존 MJX 환경과 rollout 연결을 재사용합니다. 원본과의 수치 비교 및 GPU 메모리 검증을 먼저 진행한 뒤 장기 학습합니다. MuJoCo Playground에 연결한 결과는 원본 IsaacLab 태스크의 리턴과 직접 비교할 수 없습니다.

## PPO 이후 FlashSAC 연속 실행

`scripts/run_jax_sequence.py`는 G1 PPO를 환경 8,192개·2억 transition으로 실행한 뒤, 정상 종료되면 FlashSAC을 환경 1,024개·50,000,896 transition으로 실행합니다. 최대 에피소드 길이는 둘 다 1,000스텝입니다. 위에 설명한 대로 밀기와 reset 설정은 알고리즘별 기본 벤치마크를 따릅니다.

```bash
python3 scripts/run_jax_sequence.py --output outputs/jax/g1_sequence/my-run
# FlashSAC만 실행
python3 scripts/run_jax_sequence.py --algorithms flashsac --output outputs/jax/flashsac/my-run
```

이 명령은 이 프로젝트의 `.venv`를 사용합니다. 실행 상태와 PID는 `<output>/status.json`, 각 학습기의 출력은 `<output>/<algorithm>/train.log`, 평가 수치는 `<output>/<algorithm>/metrics.jsonl`에서 확인합니다. 한 단계가 실패하면 상태를 `failed`로 기록합니다.
