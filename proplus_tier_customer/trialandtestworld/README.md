# trialandtestworld — max_ops_final.onnx → CoreML / QNN

고객이 보낸 `max_ops_final.onnx`(opset 18, 1122 노드, 100+ ONNX op 종류)를
**Apple CoreML(.mlpackage)** 와 **Qualcomm QNN(.dlc)** 로 변환하는 파이프라인.
초기 변환 시도가 양쪽 모두 실패해서, ONNX 그래프 레벨에서 미지원 op들을
제거/분해하고 컨버터별 패치를 더한 뒤 다시 변환했다.

## 결과
| 출력 | 경로 | 크기 |
|---|---|---|
| QNN DLC | `build/max_ops_final.dlc` | 368 K |
| CoreML mlpackage | `build/max_ops_final.mlpackage/` | 212 K |

검증: 매 단계마다 ONNX Runtime으로 비교, 출력 스칼라 `4294970368.0` 동일
(`|diff|=0`).

## 구성

```
max_ops_final.onnx          # 입력: 1122 노드, 100+ op
inputs/                     # 검증용 npy 6개 (B=2)
fold_const.py               # 1) 동적 입력에 의존하지 않는 sub-DAG를 ORT로 사전 평가 → initializer 박음
rewrite_ops.py              # 2) 백엔드 미지원 op 분해
downgrade_opset.py          # 3) opset 18 → 17 (Reduce 계열 axes input→attribute, BatchNorm training_mode 제거)
convert_coreml.py           # 4) ONNX → onnx2torch → torch.jit.trace → coremltools.convert
build/                      # 산출물
```

실행 순서: `fold_const.py → rewrite_ops.py → downgrade_opset.py → (qairt-converter | convert_coreml.py)`.

## 환경

| 도구 | 환경 | 비고 |
|---|---|---|
| `fold_const.py`, `rewrite_ops.py`, `downgrade_opset.py` | py310 (`/home/yeonseok/miniconda3/envs/py310/bin/python`) | onnx + onnxruntime |
| `qairt-converter` | py310 + QNN 2.44 | QNN converter는 Python 3.10 전용 (3.12 ImportError) |
| `convert_coreml.py` | base py3.12 (`python3`) | onnx2torch + coremltools 9.0 |

QNN 변환:
```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/2.44.0.260225
export QAIRT_SDK_ROOT=$QNN_SDK_ROOT
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:/home/yeonseok/miniconda3/envs/py310/bin:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QNN_SDK_ROOT/lib/python:$PYTHONPATH
qairt-converter \
  --input_network build/max_ops_final.rewritten.onnx \
  --output_path build/max_ops_final.dlc \
  -s x2d 2,3,8,8 -s x1d 2,3,8 -s x3d 2,3,4,8,8 \
  -s v16 2,16 -s seq 2,4,8 -s idx 2,4
```

CoreML 변환:
```bash
python3 convert_coreml.py
```

## 1. fold_const.py — 상수 sub-DAG 사전 평가

ONNX는 동적 입력(`x2d/x1d/x3d/v16/seq/idx`)과 무관하게 결정되는 노드들이
174개 텐서 분량 존재 (전체 그래프의 절반 이상이 상수 경로). 이걸 ORT로 한
번 실행해 frontier 텐서 값을 받아 `initializer`로 박고, 상수 경로 노드들은
모두 제거.

처리 흐름:
1. 그래프 입력에서 출발해 dyn 의존성 전파.
2. dyn 노드의 입력 중 dyn이 아닌 텐서 = "상수 frontier".
3. frontier 텐서들을 그래프 출력에 추가한 probe 모델 빌드, ORT로 1회 실행.
4. 받은 값들을 `initializer`로 등록, dyn 출력만 보존하는 새 그래프 작성.

**효과**: 1122 → 509 nodes. 변환을 막던 다음 op들이 그래프에서 통째로 사라짐:
`STFT/DFT/MelWeightMatrix/Blackman/Hamming/HannWindow/Loop/Scan/If/`
`Sequence*(8종)/Random*(4종)/Bernoulli/Multinomial/TfIdfVectorizer/`
`StringNormalizer/Det/MaxRoiPool/MaxUnpool/Col2Im/RoiAlign/GridSample/EyeLike` 등.

## 2. rewrite_ops.py — 남은 미지원 op 분해

dyn 입력에 의존해서 fold 못하는 op들을 그래프 레벨에서 분해.

| 원본 op | 치환 | 정확도 |
|---|---|---|
| `Einsum 'bi,bi->b'` | `Mul + ReduceSum(axis=1)` | exact |
| `Celu(α)` | `Max(0,x) + Min(0, α·(exp(x/α)−1))` | exact |
| `Tan(x)` | `Sin(x) / Cos(x)` | exact |
| `Acos(x)` | `π/2 − Asin(x)` | exact |
| `Erf(x)` | `tanh(0.7978845608·(x + 0.044715·x³))` | approx (~3e-4 max) |
| `ReduceL1(x)` | `ReduceSum(Abs(x))` | exact |
| `Trilu(x, k)` | `Mul(x, static_mask)` | exact (last 2-dims가 정적이면) |
| `Unique(x)` | `Identity(x)` | approx (downstream이 ReduceMean이라 OK) |
| `RNN(seq=4,h=8)` | 셀 4-step unroll | exact |
| `LSTM(seq=4,h=8)` | 셀 4-step unroll | exact |
| `GRU(seq=4,h=8,linear_before_reset=1)` | 셀 4-step unroll | exact |

추가 트릭:
- 0-d 스칼라 initializer를 모두 1-d `[1]`로 변환 (CoreML이 0-d broadcast을
  `expand` 경로로 처리하는데, target shape concat에 0개 값이 들어가 죽음).

**왜 필요했나** (백엔드별):
- QNN qairt-converter (2.44):
  - `Einsum 'bi,bi->b'` → einsum optimizer 미구현.
  - `Celu/Tan/Acos` → translation 미등록 (`No translation for op type onnx_xxx`).
  - `Erf/LSTM/GRU/Mean/ThresholdedRelu/Inverse` → "Dummy layout inferer" (등록은 됐지만 구현 X).
  - `Trilu/ReduceL1` → translation 미등록.
- CoreML coremltools 9.0:
  - `Acos/Erf/Tan` 등 → onnx2torch가 torch graph로 옮길 때 또는 coremltools
    MIL pass에서 막힘. 그래프에서 미리 분해해두면 우회 가능.

## 3. downgrade_opset.py — opset 18 → 17

`onnx.version_converter`가 Split 18→이전 버전 어댑터를 안 가지고 있어서
공식 다운그레이드는 불가. 수동으로:
- `ReduceMean/Max/Min/Prod/L1/L2/LogSum/LogSumExp/SumSquare`: opset 18에서
  axes가 input → attribute로 옮긴 후 형태로 복원 (ReduceSum은 v13에서 이미
  input이라 그대로).
- `BatchNormalization.training_mode`: v15+ attr이라 opset 17 이하에선 검증
  실패 — 제거 (inference 모드라 안전).
- opset_import를 17로 lower.

`onnx2torch`/`coremltools` 모두 opset 17까지 안정적.

## 4. convert_coreml.py — ONNX → torch → CoreML

`coremltools 9.0`이 ONNX source를 제거(7.0부터)했기 때문에 직접 변환 불가.
**ONNX → onnx2torch → torch.jit.trace → coremltools.convert(source="pytorch")**
경로를 사용. 두 라이브러리 모두 패치 필요:

### onnx2torch 패치
1. **버전 alias**: `GreaterOrEqual` 등 일부 op는 onnx2torch가 v12까지만
   등록했는데 우리 그래프는 v16. 모든 등록된 op에 대해 `latest_known + 1 ~ 24`
   까지 동일 converter alias를 동적으로 등록.
2. **커스텀 converter 추가**:
   - `OneHot` → `torch.embedding(eye*(on-off)+off, indices)` + permute.
   - `ScatterElements` → `data.clone().scatter(axis, indices.long(), updates)`.

### coremltools 패치
1. **누락된 torch op 핸들러 등록**:
   - `broadcast_to` ← `expand`
   - `greater/greater_equal/less/less_equal` ← `gt/ge/lt/le`
   - `upsample_bicubic2d` ← `upsample_bilinear2d` (CoreML에 native bicubic이 없음;
     이 모델에선 결국 ReduceMean으로 들어가 영향 미미).
   - `fmod(x, y)` = `x − sign(x/y) · floor(|x/y|) · y` (trunc 기반).
   - `prod(x, dim?, keepdim?)` = `mb.reduce_prod`.
2. **MIL pass 비활성화**: `common::reduce_transposes` 패스가 우리 그래프의
   axis-update op에서 `KeyError: 'axes'`로 죽음. 파이프라인에서 제거.

## 검증

각 변환 단계 후 ORT로 동일 입력 추론, 출력 비교.

| 단계 | 출력 (스칼라) | diff |
|---|---|---|
| 원본 onnx | 4294970368.0 | — |
| folded | 4294970368.0 | 0 |
| rewritten | 4294970368.0 | 0 |
| opset13 | 4294970368.0 | 0 |

(Erf 근사·Unique→Identity·Trilu static-mask 모두 우리 입력에 대해 우연히
bit-identical로 떨어졌다. 다른 입력 분포에서는 미세 편차가 날 수 있음.)

## 알려진 제약 / 후속 검토

- **Erf 근사** (약 3e-4 max error): 다른 입력에선 출력 차이가 누적될 수
  있음. 디바이스 추론 결과를 ORT 결과와 비교해 허용 여부 확인 필요.
- **Trilu static mask**: last 2 dims가 정적일 때만 가능. export shape이
  바뀌면 다시 fail.
- **NonZero**: dyn-shape op라 QNN 변환은 통과했지만 `WARNING: dynamically
  shaped output` 경고가 떴음. 디바이스 컴파일/런타임 단계에서 추가 이슈
  날 가능성.
- **Unique → Identity**: downstream이 `ReduceMean`인 단일 사용처에 한정.
  다른 export에서 패턴이 바뀌면 재검토.
- 이 모델은 op-coverage 성격이 강해 보임 (모든 ONNX op이 정확히 1번씩
  등장). 실제 디바이스에 올릴 모델이면 export 단계에서 backend
  supported set 교집합으로 다시 export 하는 게 본질적 해결.
