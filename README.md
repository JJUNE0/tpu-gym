# tpu-gym

JAX와 MuJoCo Playground(MJX)를 사용해 강화학습을 실험하는 프로젝트입니다. 로컬 GPU에서 먼저 학습을 검증하고, 같은 코드를 Google Cloud TPU로 옮겨 실행하는 흐름을 목표로 합니다.

## 1. 프로젝트 목표

- 휴머노이드에 한정하지 않고 locomotion, manipulation 등 여러 분야의 태스크를 추가합니다.
- 환경과 학습 알고리즘을 분리해 PPO, FlashSAC, SAC++ 등을 같은 프로젝트에서 실험합니다.
- 의존성과 third-party 커밋을 고정해 로컬 GPU와 TPU에서 실험 설정을 재사용합니다.

현재 PPO와 FlashSAC의 G1 평지 예제를 GPU에서 검증했습니다. SAC++는 참조 소스만 등록되어 있으며 JAX 이식은 후속 작업입니다. 실제 TPU 하드웨어에서의 학습 검증도 아직 진행하지 않았습니다.

## 2. Google Cloud 계정 연결

[Google Cloud CLI](https://docs.cloud.google.com/sdk/docs/install)를 설치하고, 결제가 활성화된 프로젝트와 사용할 zone의 TPU 할당량을 준비합니다. 프로젝트에 API 활성화와 VM 생성·접속 권한도 필요합니다. 로컬 터미널에서 실행합니다.

```bash
PROJECT_ID=your-project-id

gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com --project="$PROJECT_ID"
```

`gcloud auth login`으로 Google 계정을 연결하고 `PROJECT_ID`로 사용할 프로젝트를 선택합니다. [Google Cloud 인증 가이드](https://docs.cloud.google.com/sdk/docs/authenticate)

로컬 GPU만 사용할 때는 이 단계를 생략할 수 있습니다. TPU VM 생성과 접속은 아래 5번에서 진행합니다.

## 3. 코드 구조

```text
tpu-gym/
├── train.py                          # 학습 진입점
├── pyproject.toml / uv.lock           # Python 및 CPU·CUDA·TPU 의존성
├── src/tpu_gym/
│   ├── train.py                      # CLI, 장치 선택, 태스크 등록
│   ├── algorithm/
│   │   ├── ppo.py                    # Brax PPO
│   │   └── flashsac.py               # FlashCHORD JAX FlashSAC 연결
│   └── tasks/locomotion/g1/flat.py    # 현재 검증된 G1 예제
├── third_party/                      # 커밋을 고정한 Git submodule
│   ├── mujoco_playground/            # MJX 환경, v0.2.0
│   ├── flash_chord/                  # video_to_data 안의 JAX 학습기
│   ├── FlashSAC/                     # 원본 설정 및 비교 자료
│   └── sacpp/                        # SAC++ 참조 구현, 비공개
├── scripts/bootstrap.py              # submodule·패치·모델 준비
├── patches/                          # third-party에 적용하는 수정
├── tests/jax/                        # 학습기 연결 회귀 테스트
├── benchmarks/                       # 기존 검증 결과와 설정
└── outputs/                          # 로그·체크포인트, Git 제외
```

새 태스크는 `src/tpu_gym/tasks/<분야>/<대상>/`에 두고 `src/tpu_gym/train.py`의 `TASKS`에 등록합니다. 기존 G1 모듈처럼 `make_environment(algorithm=...)`, `ppo_defaults()`, `flashsac_defaults()`를 제공합니다.

`bootstrap.py`는 FlashCHORD에 필요한 패치도 적용합니다. SAC++ 접근 권한이 있으면 아래 명령의 `--skip-sacpp`를 생략할 수 있습니다. 다른 Menagerie 모델은 `--assets`에 모델 폴더 이름을 추가해 준비합니다.

## 4. 먼저 로컬 GPU에서 검증하기

Linux x86-64와 NVIDIA GPU 기준입니다. [uv](https://docs.astral.sh/uv/getting-started/installation/)를 설치한 뒤 소스와 전용 `.venv`를 준비합니다. Python 3.13은 uv가 준비합니다.

```bash
git clone https://github.com/JJUNE0/tpu-gym.git
cd tpu-gym
python3 scripts/bootstrap.py --skip-sacpp --assets unitree_g1
uv sync --locked --extra cuda
```

**본인 GPU에서 아래 짧은 실행을 먼저 완료하는 것을 권장합니다.** `--smoke`는 환경·네트워크·학습량을 줄여 실제 업데이트와 체크포인트 저장을 확인합니다. 첫 실행에는 XLA 컴파일 시간이 포함됩니다.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm ppo --platform gpu --smoke

XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm flashsac --platform gpu --smoke
```

두 실행이 완료되면 `--smoke`를 빼고 본학습을 진행합니다. 아래 PPO 예제의 기본 학습량은 환경 8,192개, 2억 transition입니다.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm ppo --platform gpu
```

FlashSAC의 기본 학습량은 환경 1,024개, 약 5천만 transition입니다. 기본 replay 1천만 개만 약 17.2 GiB를 사용하므로 본학습 전 GPU 메모리를 확인합니다. 24 GiB GPU에서 검증한 실행은 다음과 같습니다.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_CLIENT_MEM_FRACTION=0.90 \
uv run --locked --extra cuda python train.py \
  --task g1-flat --algorithm flashsac --platform gpu
```

`--num-envs`, `--steps`, `--batch-size`, `--seed`, `--output`으로 실험을 조절합니다. 결과는 기본적으로 `outputs/jax/<task>/<algorithm>/<timestamp>/`에 저장합니다. 기존 FlashSAC GPU 검증은 5천만 transition에서 리턴 40.744를 기록했으며, [설정과 평가 곡선](benchmarks/g1-flat/flashsac-seed0.json)을 보관합니다. 단일 seed 결과이고 PPO와 FlashSAC의 기본 밀기·reset 조건도 다르므로 비교 시 환경 설정을 함께 확인합니다.

## 5. Google TPU에서 학습하기

**TPU VM에 SSH로 접속한 뒤, VM 안에서 같은 학습 명령에 `--tpu`를 붙여 실행합니다.** 이 옵션이 학습 프로세스가 사용할 장치를 TPU로 선택합니다.

먼저 로컬 터미널에서 VM을 만들고 접속합니다. 아래는 v6e 1칩 VM 예시입니다. `TPU_ZONE`은 할당량이 있는 zone으로 바꾸고, 머신 유형과 생성 옵션은 [Google의 TPU VM 생성 가이드](https://docs.cloud.google.com/compute/docs/tpus/quickstart-create-tpu-instance)와 [머신 유형](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines)을 확인합니다. 이미 VM이 있으면 생성 명령은 생략합니다.

```bash
TPU_ZONE=your-tpu-zone
TPU_VM=tpu-gym

gcloud compute instances create "$TPU_VM" \
  --project="$PROJECT_ID" --zone="$TPU_ZONE" \
  --machine-type=ct6e-standard-1t \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --maintenance-policy=TERMINATE

gcloud compute ssh "$TPU_VM" --project="$PROJECT_ID" --zone="$TPU_ZONE"
```

이제 **TPU VM 안에서** Git과 uv를 준비하고 실행합니다. `--extra tpu`가 TPU용 의존성을 설치하며, 로컬의 CUDA 환경과 별도로 관리합니다.

```bash
git clone https://github.com/JJUNE0/tpu-gym.git
cd tpu-gym
python3 scripts/bootstrap.py --skip-sacpp --assets unitree_g1
uv sync --locked --extra tpu

# TPU 장치와 실제 업데이트 확인
uv run --locked --extra tpu python train.py \
  --task g1-flat --algorithm ppo --tpu --smoke

# 검증 후 본학습
uv run --locked --extra tpu python train.py \
  --task g1-flat --algorithm ppo --tpu
```

FlashSAC은 `--algorithm flashsac`으로 선택하고 TPU에서도 먼저 `--smoke`로 확인합니다. 현재 FlashSAC은 단일 JAX 장치에서 실행하며, 여러 TPU VM에 걸친 분산 학습은 지원하지 않습니다.

실험이 끝나면 로컬 터미널에서 결과를 복사한 뒤 VM을 정리합니다. 아래 복사 경로는 VM에 `~/tpu-gym`으로 clone한 경우입니다. VM 삭제는 연결된 부팅 디스크의 결과도 삭제할 수 있으므로 복사 완료 후 실행합니다.

```bash
gcloud compute scp --recurse "$TPU_VM:~/tpu-gym/outputs/jax" ./tpu-results \
  --project="$PROJECT_ID" --zone="$TPU_ZONE"

gcloud compute instances delete "$TPU_VM" \
  --project="$PROJECT_ID" --zone="$TPU_ZONE"
```
